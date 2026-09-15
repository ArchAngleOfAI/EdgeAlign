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
from typing import Dict, Iterable, List, Optional

import torch

from .config import ExperimentConfig, GeneratorConfig, load_config
from .data.pipeline import build_mixed_dataset
from .data.synthetic import synthetic_batches
from .evaluate import evaluate
from .frozen_model import FrozenLLMHarness
from .generators.base import GeneratorFrontend
from .generators.self_embedding import SelfEmbeddingGenerator
from .generators.stub import MeanPoolStubGenerator
from .init_utils import compute_embedding_scale_stats, rescale_output_layer_to_target_stats
from .losses import embedding_scale_aux_loss, kl_distillation_loss

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


def build_data_iterators(data_cfg) -> "tuple[Iterable[List[str]], Optional[List[List[str]]]]":
    """Returns (train_iter, eval_batches). eval_batches is a plain list
    (reusable across multiple evaluate() calls during training) or None
    if no held-out evaluation is configured."""
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
) -> torch.Tensor:
    tokenized = frozen_llm.tokenizer(
        prompts, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt"
    )
    device = frozen_llm.input_embedding_device
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    needs_hidden_states = getattr(generator, "needs_frozen_hidden_states", False)
    if needs_hidden_states:
        teacher_logits, hidden_states = frozen_llm.teacher_pass(input_ids, attention_mask, output_hidden_states=True)
        soft_prefix = generator(prompts, hidden_states=hidden_states, attention_mask=attention_mask)
    else:
        teacher_logits = frozen_llm.teacher_pass(input_ids, attention_mask)
        soft_prefix = generator(prompts)

    student_logits = frozen_llm.student_pass(input_ids, attention_mask, soft_prefix)

    loss = kl_distillation_loss(teacher_logits, student_logits, generator.n_soft_tokens, attention_mask)

    if aux_loss_weight and embedding_stats is not None:
        aux = embedding_scale_aux_loss(soft_prefix, embedding_stats["mean_norm"])
        loss = loss + aux_loss_weight * aux

    return loss


def run_training(
    frozen_llm: FrozenLLMHarness,
    generator: GeneratorFrontend,
    data_iter: Iterable[List[str]],
    train_cfg,
    max_seq_len: int,
    embedding_stats: Optional[Dict[str, torch.Tensor]] = None,
    eval_data_iter: Optional[Iterable[List[str]]] = None,
) -> List[float]:
    optimizer = torch.optim.AdamW(generator.parameters(), lr=train_cfg.learning_rate)
    generator.train()

    history: List[float] = []
    for step, prompts in enumerate(data_iter):
        if train_cfg.max_steps is not None and step >= train_cfg.max_steps:
            break

        loss = training_step(
            frozen_llm, generator, prompts, max_seq_len, train_cfg.aux_loss_weight, embedding_stats
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if train_cfg.log_every and step % train_cfg.log_every == 0:
            print(f"step {step}: kl_loss={loss.item():.4f}")

        if eval_data_iter is not None and train_cfg.eval_every and step > 0 and step % train_cfg.eval_every == 0:
            val_kl = evaluate(frozen_llm, generator, eval_data_iter, max_seq_len)
            print(f"step {step}: val_kl={val_kl:.4f}")

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Soft prompt generator self-distillation warm-start training")
    parser.add_argument("--config", required=True, help="path to a YAML config (see configs/)")
    args = parser.parse_args()

    cfg: ExperimentConfig = load_config(args.config)
    torch.manual_seed(cfg.train.seed)

    frozen_llm = FrozenLLMHarness.from_pretrained(cfg.model)
    embedding_stats = compute_embedding_scale_stats(frozen_llm)

    generator = build_generator(cfg.generator, frozen_llm.embedding_dim, frozen_llm.model.config)
    rescale_output_layer_to_target_stats(
        generator.output_layer_for_init(), embedding_stats["mean"], embedding_stats["std"]
    )
    generator.to(frozen_llm.input_embedding_device)

    data_iter, eval_batches = build_data_iterators(cfg.data)

    history = run_training(
        frozen_llm, generator, data_iter, cfg.train, cfg.data.max_seq_len, embedding_stats,
        eval_data_iter=eval_batches,
    )

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(cfg.train.output_dir, "generator.pt"))
    print(f"done. final kl_loss={history[-1]:.4f}" if history else "done. no training steps ran.")


if __name__ == "__main__":
    main()
