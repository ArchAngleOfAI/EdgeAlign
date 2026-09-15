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
