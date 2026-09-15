"""Dataset mixing (infra spec section 6.4).

Combines the three prepared sources at a configurable ratio into a single
training stream, via datasets.interleave_datasets. stopping_strategy=
"all_exhausted" keeps sampling (looping smaller sources) until the
largest source is exhausted, so xLAM's much smaller row count doesn't
silently make tool-calling prompt shapes underrepresented relative to the
configured weight.
"""
from typing import List

from datasets import Dataset, load_from_disk
from datasets import interleave_datasets

from ..config import DataSourceConfig
from . import opencodeinstruct, stack_edu, xlam


def _load_stack_edu(extra: dict) -> Dataset:
    # Expects a path already produced by scripts/prepare_stack_edu.py --
    # this loader never resolves S3 content itself (spec section 6.1).
    prepared_path = extra["prepared_path"]
    ds = load_from_disk(prepared_path)
    return stack_edu.to_prompt_examples(ds)


def _load_opencodeinstruct(extra: dict) -> Dataset:
    return opencodeinstruct.to_prompt_examples(opencodeinstruct.load_raw(**extra))


def _load_xlam(extra: dict) -> Dataset:
    return xlam.to_prompt_examples(xlam.load_raw())


SOURCE_LOADERS = {
    "stack_edu": _load_stack_edu,
    "opencodeinstruct": _load_opencodeinstruct,
    "xlam": _load_xlam,
}


def build_mixed_dataset(source_configs: List[DataSourceConfig]) -> Dataset:
    if not source_configs:
        raise ValueError("no data sources configured for real-data mode")

    datasets_, probabilities = [], []
    for src in source_configs:
        loader = SOURCE_LOADERS[src.name]
        datasets_.append(loader(src.extra))
        probabilities.append(src.weight)

    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    return interleave_datasets(datasets_, probabilities=probabilities, stopping_strategy="all_exhausted")
