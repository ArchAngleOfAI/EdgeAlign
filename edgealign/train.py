"""Training loop (spec section 4) wiring the harness, injection,
generator front-end, and losses together.

`training_step` / `run_training` are pure functions over already-built
objects, deliberately kept separate from `main()`'s config-driven
construction -- this is what lets scripts/run_smoke_test.py exercise the
exact same training loop code against a tiny local model, rather than a
separate, unverified code path.
"""
import argparse
import os
import sys
import traceback
from typing import Dict, Iterable, List, Optional

import torch

from .config import ExperimentConfig, GeneratorConfig, load_config
from .data import wikipedia
from .data.packing import pack_texts
from .data.pipeline import build_mixed_dataset
from .data.synthetic import synthetic_batches
from .evaluate import evaluate
from .frozen_model import FrozenLLMHarness
from .generators.base import GeneratorFrontend
from .generators.self_embedding import SelfEmbeddingGenerator
from .generators.stub import MeanPoolStubGenerator
from .init_utils import compute_embedding_scale_stats, rescale_output_layer_to_target_stats
from .losses import embedding_scale_aux_loss, kl_distillation_loss

# Process exit code main() uses when it catches a confirmed-sticky CUDA
# context fault (see MEMORY.md) -- scripts/train_with_restart.sh checks
# for exactly this code to decide whether to restart a fresh process.
STICKY_CUDA_FAULT_EXIT_CODE = 42

# Optimizer registry key -> torch optimizer class. Extend as needed;
# every entry must accept (params, lr=..., **kwargs).
OPTIMIZER_REGISTRY = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}


def build_optimizer(train_cfg, params) -> torch.optim.Optimizer:
    cls = OPTIMIZER_REGISTRY[train_cfg.optimizer.lower()]
    return cls(params, lr=train_cfg.learning_rate, **train_cfg.optimizer_kwargs)


def build_scheduler(train_cfg, optimizer: torch.optim.Optimizer):
    from transformers import get_scheduler

    if train_cfg.lr_scheduler != "constant" and train_cfg.max_steps is None:
        raise ValueError(
            f"lr_scheduler={train_cfg.lr_scheduler!r} needs a concrete max_steps to compute "
            f"its decay curve against -- got max_steps=None"
        )
    return get_scheduler(
        name=train_cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_cfg.warmup_steps,
        num_training_steps=train_cfg.max_steps or 1,
        **train_cfg.lr_scheduler_kwargs,
    )

# "dummy_stub" is a smoke-test-only placeholder (see generators/stub.py),
# not a real variant. "self_embedding" is v2. v1/v3 are not implemented
# yet.
GENERATOR_REGISTRY = {
    "dummy_stub": MeanPoolStubGenerator,
    "self_embedding": SelfEmbeddingGenerator,
}


def build_generator(cfg: GeneratorConfig, embedding_dim: int, frozen_model_config=None) -> GeneratorFrontend:
    cls = GENERATOR_REGISTRY[cfg.type]
    kwargs = dict(cfg.extra)
    if getattr(cls, "needs_frozen_hidden_states", False) and frozen_model_config is not None:
        # hidden_size/num_hidden_layers are properties of the frozen model
        # actually being loaded, not free hyperparameters -- default to
        # its real config rather than risking a config-file typo that
        # silently mismatches the loaded checkpoint. cfg.extra can still
        # override explicitly if ever needed.
        kwargs.setdefault("hidden_size", frozen_model_config.hidden_size)
        kwargs.setdefault("num_hidden_layers", frozen_model_config.num_hidden_layers)
    return cls(n_soft_tokens=cfg.n_soft_tokens, embedding_dim=embedding_dim, **kwargs)


def _batches_from_rows(rows, batch_size: int, max_batches: Optional[int] = None) -> List[List[str]]:
    """Materializes an iterable of dataset rows (dicts with a "prompt"
    key, or already-plain strings) into a fixed list of batches, so it
    can be iterated repeatedly (once per eval call) rather than exhausted
    after a single pass."""
    batches: List[List[str]] = []
    batch: List[str] = []
    for row in rows:
        batch.append(row["prompt"] if isinstance(row, dict) else row)
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []
            if max_batches is not None and len(batches) >= max_batches:
                break
    return batches


def build_data_iterators(
    data_cfg, eos_token: Optional[str] = None
) -> "tuple[Iterable[List[str]], Optional[List[List[str]]]]":
    """Returns (train_iter, eval_batches). eval_batches is a plain list
    (reusable across multiple evaluate() calls during training) or None
    if no held-out evaluation is configured."""
    if data_cfg.mode == "wikipedia":
        if not data_cfg.wikipedia_tar_path:
            raise ValueError("data.wikipedia_tar_path must be set for mode='wikipedia'")

        # Packing uses the frozen model's real EOS token as the document
        # separator when available (matching how the frozen model was
        # almost certainly pretrained on packed documents), falling back
        # to a plain blank line otherwise.
        separator = eos_token or "\n\n"

        def _packed_batches(seed: int, max_passages: Optional[int] = None):
            passages = wikipedia.iter_passages(
                data_cfg.wikipedia_tar_path,
                shuffle_buffer_size=data_cfg.wikipedia_shuffle_buffer_size,
                seed=seed,
                max_passages=max_passages,
            )
            packed = pack_texts(passages, max_seq_len=data_cfg.max_seq_len, separator=separator)
            batch = []
            for text in packed:
                batch.append(text)
                if len(batch) == data_cfg.batch_size:
                    yield batch
                    batch = []

        train_iter = _packed_batches(seed=0)
        eval_batches = None
        if data_cfg.eval_fraction > 0:
            # The corpus is streamed, not a known-length dataset, so
            # eval_fraction here means "read roughly this many packed
            # eval batches" (via a passage budget, ~5 passages/pack) from
            # a different shuffle seed -- not a true held-out split of a
            # fixed-size dataset like the "real" mode below.
            passages_budget = data_cfg.max_eval_batches * data_cfg.batch_size * 5
            eval_batches = list(_packed_batches(seed=1, max_passages=passages_budget))[: data_cfg.max_eval_batches]

        return train_iter, eval_batches

    if data_cfg.mode == "synthetic":
        train_iter = synthetic_batches(batch_size=data_cfg.batch_size, num_batches=None, seed=0)
        eval_batches = None
        if data_cfg.eval_fraction > 0:
            eval_batches = list(
                synthetic_batches(batch_size=data_cfg.batch_size, num_batches=data_cfg.max_eval_batches, seed=1)
            )
        return train_iter, eval_batches

    if data_cfg.mode == "real":
        ds = build_mixed_dataset(data_cfg.sources)
        eval_batches = None
        if data_cfg.eval_fraction > 0:
            n_eval_rows = max(1, int(len(ds) * data_cfg.eval_fraction))
            eval_ds = ds.select(range(n_eval_rows))
            train_ds = ds.select(range(n_eval_rows, len(ds)))
            eval_batches = _batches_from_rows(eval_ds, data_cfg.batch_size, data_cfg.max_eval_batches)
        else:
            train_ds = ds

        def _iter():
            batch = []
            for row in train_ds:
                batch.append(row["prompt"])
                if len(batch) == data_cfg.batch_size:
                    yield batch
                    batch = []

        return _iter(), eval_batches

    raise ValueError(f"unknown data mode: {data_cfg.mode!r}")


def training_step(
    frozen_llm: FrozenLLMHarness,
    generator: GeneratorFrontend,
    prompts: List[str],
    max_seq_len: int,
    aux_loss_weight: float,
    embedding_stats: Optional[Dict[str, torch.Tensor]],
    kl_top_k: Optional[int] = None,
    kl_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    tokenized = frozen_llm.tokenizer(
        prompts, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt"
    )
    device = frozen_llm.input_embedding_device
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    needs_hidden_states = getattr(generator, "needs_frozen_hidden_states", False)
    if needs_hidden_states:
        # Hook-based selective capture (only the layers the generator
        # actually needs), not output_hidden_states=True -- see
        # generators/base.py's required_hidden_state_layers for why this
        # matters at long sequence lengths, not just as an optimization.
        teacher_logits, hidden_states = frozen_llm.teacher_pass_selected_layers(
            input_ids, attention_mask, generator.required_hidden_state_layers
        )
        soft_prefix = generator(prompts, hidden_states=hidden_states, attention_mask=attention_mask)
    else:
        teacher_logits = frozen_llm.teacher_pass(input_ids, attention_mask)
        soft_prefix = generator(prompts)

    student_logits = frozen_llm.student_pass(input_ids, attention_mask, soft_prefix)

    loss = kl_distillation_loss(
        teacher_logits, student_logits, generator.n_soft_tokens, attention_mask,
        top_k=kl_top_k, chunk_size=kl_chunk_size,
    )

    if aux_loss_weight and embedding_stats is not None:
        aux = embedding_scale_aux_loss(soft_prefix, embedding_stats["mean_norm"])
        loss = loss + aux_loss_weight * aux

    return loss


def _checkpoint_path(train_cfg) -> str:
    return os.path.join(train_cfg.output_dir, "checkpoint.pt")


def save_checkpoint(
    path: str,
    generator: GeneratorFrontend,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    history: List[float],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path: str,
    generator: GeneratorFrontend,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> "tuple[int, List[float]]":
    """Loads generator/optimizer/scheduler state in place from `path` if it
    exists. Returns (global_step, history) to resume from -- (0, []) if
    there is no checkpoint yet."""
    if not os.path.exists(path):
        return 0, []
    device = next(generator.parameters()).device
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"[resume] loaded {path} at global_step={ckpt['global_step']}", flush=True)
    return ckpt["global_step"], ckpt["history"]


def run_training(
    frozen_llm: FrozenLLMHarness,
    generator: GeneratorFrontend,
    data_iter: Iterable[List[str]],
    train_cfg,
    max_seq_len: int,
    embedding_stats: Optional[Dict[str, torch.Tensor]] = None,
    eval_data_iter: Optional[Iterable[List[str]]] = None,
) -> List[float]:
    """max_steps/log_every/eval_every all count *optimizer* steps (after
    grad_accum_steps micro-batches have been accumulated), not raw
    micro-batches -- so a config's max_steps means the same thing
    whether grad_accum_steps is 1 or 8."""
    optimizer = build_optimizer(train_cfg, generator.parameters())
    scheduler = build_scheduler(train_cfg, optimizer)
    generator.train()

    ckpt_path = _checkpoint_path(train_cfg)
    global_step, history = load_checkpoint(ckpt_path, generator, optimizer, scheduler)
    micro_step = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    for prompts in data_iter:
        if train_cfg.max_steps is not None and global_step >= train_cfg.max_steps:
            break

        loss = training_step(
            frozen_llm, generator, prompts, max_seq_len, train_cfg.aux_loss_weight, embedding_stats,
            kl_top_k=train_cfg.kl_top_k, kl_chunk_size=train_cfg.kl_chunk_size,
        )
        (loss / train_cfg.grad_accum_steps).backward()
        accum_loss += loss.item()
        micro_step += 1

        if micro_step % train_cfg.grad_accum_steps != 0:
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        avg_loss = accum_loss / train_cfg.grad_accum_steps
        accum_loss = 0.0
        history.append(avg_loss)

        if train_cfg.log_every and global_step % train_cfg.log_every == 0:
            print(f"step {global_step}: kl_loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        if eval_data_iter is not None and train_cfg.eval_every and global_step > 0 and global_step % train_cfg.eval_every == 0:
            val_kl = evaluate(
                frozen_llm, generator, eval_data_iter, max_seq_len,
                kl_top_k=train_cfg.kl_top_k, kl_chunk_size=train_cfg.kl_chunk_size,
            )
            print(f"step {global_step}: val_kl={val_kl:.4f}")

        global_step += 1

        if train_cfg.checkpoint_every and global_step % train_cfg.checkpoint_every == 0:
            save_checkpoint(ckpt_path, generator, optimizer, scheduler, global_step, history)

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Soft prompt generator self-distillation warm-start training")
    parser.add_argument("--config", required=True, help="path to a YAML config (see configs/)")
    args = parser.parse_args()

    cfg: ExperimentConfig = load_config(args.config)

    if cfg.train.available_gpus:
        if cfg.train.num_gpus is not None and cfg.train.num_gpus != len(cfg.train.available_gpus):
            raise ValueError(
                f"train.num_gpus={cfg.train.num_gpus} does not match "
                f"len(train.available_gpus)={len(cfg.train.available_gpus)}"
            )
        # Must happen before any CUDA call (including inside
        # FrozenLLMHarness.from_pretrained below) -- CUDA_VISIBLE_DEVICES
        # can't be changed once the driver has initialized a context.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in cfg.train.available_gpus)
        print(f"Restricting to GPUs {cfg.train.available_gpus} (CUDA_VISIBLE_DEVICES set)")

    torch.manual_seed(cfg.train.seed)

    try:
        frozen_llm = FrozenLLMHarness.from_pretrained(cfg.model)
        embedding_stats = compute_embedding_scale_stats(frozen_llm)

        generator = build_generator(cfg.generator, frozen_llm.embedding_dim, frozen_llm.model.config)
        rescale_output_layer_to_target_stats(
            generator.output_layer_for_init(), embedding_stats["mean"], embedding_stats["std"]
        )
        generator.to(frozen_llm.input_embedding_device)

        data_iter, eval_batches = build_data_iterators(cfg.data, eos_token=frozen_llm.tokenizer.eos_token)

        history = run_training(
            frozen_llm, generator, data_iter, cfg.train, cfg.data.max_seq_len, embedding_stats,
            eval_data_iter=eval_batches,
        )
    except (torch.AcceleratorError, torch.OutOfMemoryError) as exc:
        # Confirmed on this cluster to be a rare, non-reproducible *sticky*
        # CUDA context fault (see MEMORY.md): once one occurs, every
        # further CUDA call in this same process fails identically --
        # confirmed even for torch.cuda.empty_cache() -- so there is no
        # safe in-process recovery. Exit immediately with a distinct code,
        # deliberately making no further CUDA calls, so an external
        # restart loop (scripts/train_with_restart.sh) can tell this apart
        # from a real bug/config error and start a fresh process, which
        # resumes from the last periodic checkpoint (train.checkpoint_every).
        traceback.print_exc()
        print(
            f"[fatal CUDA fault] {type(exc).__name__}: {str(exc)[:300]!r} -- exiting "
            f"for external restart (exit code {STICKY_CUDA_FAULT_EXIT_CODE})",
            flush=True,
        )
        sys.exit(STICKY_CUDA_FAULT_EXIT_CODE)

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(cfg.train.output_dir, "generator.pt"))
    print(f"done. final kl_loss={history[-1]:.4f}" if history else "done. no training steps ran.")


if __name__ == "__main__":
    main()
