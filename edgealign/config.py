"""Config schema for the shared infrastructure (spec section: whole doc).

Plain dataclasses + YAML, deliberately not a validation framework -- the
only things read from a config file are the ones the infra spec calls out
as configurable hyperparameters (N soft tokens, dataset mixing ratios,
etc).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    # HF model id or local path for the frozen base LLM. For the
    # prototype phase this should be the Qwen 3 32B checkpoint; see
    # configs/prototype_32b.yaml. Never set to a deploy-phase (8B) model
    # here -- that is a separate, later port per the infra spec.
    name_or_path: str
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    # Per-GPU memory cap (GiB) passed to from_pretrained's max_memory,
    # applied uniformly to every visible CUDA device. None = let
    # device_map="auto" use the full reported-free memory on each GPU.
    # On a shared cluster, using every last free MB makes loading
    # fragile to momentary contention from other users' jobs (confirmed
    # directly: two consecutive load-time failures despite nvidia-smi
    # showing the GPUs fully free immediately before and after each
    # attempt -- see MEMORY.md). A small margin below the real capacity
    # trades a bit of usable memory for headroom against exactly that.
    max_memory_gib: Optional[float] = None


@dataclass
class GeneratorConfig:
    # Registry key for the generator front-end (spec section 3). No
    # v1/v2/v3 variant has been chosen yet -- "dummy_stub" is a
    # smoke-test-only placeholder, not a real variant. Replace this once
    # a variant is implemented and registered.
    type: str = "dummy_stub"
    n_soft_tokens: int = 16
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSourceConfig:
    name: str
    weight: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    # "synthetic" = small in-code corpus, local smoke testing only.
    # "real" = the actual Stack-Edu/OpenCodeInstruct/xLAM mix (spec
    # section 6), for the training cluster.
    # "wikipedia" = the packed Wikipedia passage corpus actually used for
    # the first real-data validation run on the self-embedding branch
    # (see MEMORY.md "Pretraining data actually used") -- a deliberate,
    # acknowledged departure from the code/tool-use data this project's
    # specs otherwise call for; see PROMPTS/soft_prompt_generator_spec_v2_self_embedding.md.
    mode: str = "synthetic"
    max_seq_len: int = 512
    batch_size: int = 4
    sources: List[DataSourceConfig] = field(default_factory=list)
    # Fraction of the mixed dataset held out for eval (spec section 7).
    # 0 disables held-out evaluation entirely. Ignored in "synthetic" mode
    # (a small fixed synthetic eval set is used instead when > 0).
    eval_fraction: float = 0.0
    max_eval_batches: int = 20
    # "wikipedia" mode only: path to the wiki-18.jsonl tar archive (see
    # edgealign/data/wikipedia.py), and the shuffle buffer size for its
    # approximate streaming shuffle.
    wikipedia_tar_path: Optional[str] = None
    wikipedia_shuffle_buffer_size: int = 10000


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    max_steps: Optional[int] = 1000
    eval_every: int = 0
    aux_loss_weight: float = 0.0
    log_every: int = 10
    seed: int = 0
    output_dir: str = "checkpoints/run"

    # KL loss memory controls (edgealign/losses.py kl_distillation_loss)
    # -- both None by default (exact, unchanged behavior). At the real
    # vocab_size=151936, the float32+log_softmax step alone can OOM
    # before Liger's kernel is even involved; see MEMORY.md.
    kl_top_k: Optional[int] = None
    kl_chunk_size: Optional[int] = None

    # Gradient accumulation: this many micro-batches (each of
    # data.batch_size) are accumulated before each optimizer step.
    # Effective batch size = data.batch_size * grad_accum_steps.
    # max_steps/log_every/eval_every all count optimizer steps (after
    # accumulation), not micro-batches. 1 = no accumulation, one
    # optimizer step per micro-batch (previous, still-default behavior).
    grad_accum_steps: int = 1

    # Optimizer registry key (edgealign/train.py's OPTIMIZER_REGISTRY),
    # plus any extra constructor kwargs beyond lr (e.g. weight_decay,
    # betas).
    optimizer: str = "adamw"
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    # LR schedule: any name transformers.get_scheduler supports
    # ("constant", "linear", "cosine", "cosine_with_restarts",
    # "polynomial", "constant_with_warmup", ...). "constant" with
    # warmup_steps=0 reproduces the previous fixed-LR-forever behavior
    # exactly. A non-"constant" schedule needs a concrete max_steps (not
    # None) to compute its decay curve against.
    lr_scheduler: str = "constant"
    warmup_steps: int = 0
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    # GPU restriction: if set, edgealign/train.py's main() sets
    # CUDA_VISIBLE_DEVICES to exactly this list before any CUDA call is
    # made (must happen before FrozenLLMHarness.from_pretrained -- CUDA
    # device visibility can't be changed after the driver has already
    # initialized a context). num_gpus, if set, is cross-checked against
    # len(available_gpus) as a sanity guard, not an independent control.
    num_gpus: Optional[int] = None
    available_gpus: List[int] = field(default_factory=list)

    # Save generator+optimizer+scheduler+step state to
    # "<output_dir>/checkpoint.pt" every this many optimizer steps. 0
    # disables checkpointing. run_training() auto-resumes from this file
    # if it already exists at start (the data iterator itself is NOT
    # checkpointed -- it's a streaming corpus, so a resumed run continues
    # consuming fresh batches rather than replaying the exact pre-restart
    # sequence; acceptable for this self-distillation warm-start, not a
    # correctness requirement).
    #
    # This exists because torch.AcceleratorError / torch.OutOfMemoryError
    # have been confirmed on this cluster to be rare, non-reproducible
    # *sticky* CUDA context faults (see MEMORY.md): once one occurs, every
    # further CUDA call in that same process fails identically (confirmed
    # even for torch.cuda.empty_cache()), so there is no safe in-process
    # retry -- recovery requires a fresh process, which is what
    # scripts/train_with_restart.sh does, resuming from this checkpoint.
    checkpoint_every: int = 0


@dataclass
class ExperimentConfig:
    model: ModelConfig
    generator: GeneratorConfig
    data: DataConfig
    train: TrainConfig


def load_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    model = ModelConfig(**raw["model"])
    generator = GeneratorConfig(**raw["generator"])

    data_raw = dict(raw["data"])
    sources_raw = data_raw.pop("sources", [])
    sources = [DataSourceConfig(**s) for s in sources_raw]
    data = DataConfig(sources=sources, **data_raw)

    train = TrainConfig(**raw["train"])

    return ExperimentConfig(model=model, generator=generator, data=data, train=train)
