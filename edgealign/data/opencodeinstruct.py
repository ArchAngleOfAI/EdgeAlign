"""OpenCodeInstruct loader -- infra spec section 6.2. Content is inline,
no separate resolution step needed."""
from typing import Optional

from datasets import Dataset, load_dataset


def load_raw(require_pass: bool = True, min_test_score: Optional[float] = None) -> Dataset:
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train")
    if require_pass:
        ds = ds.filter(lambda row: row["tests_execution_status"] == "pass")
    if min_test_score is not None:
        ds = ds.filter(lambda row: row["average_test_score"] >= min_test_score)
    return ds


def to_prompt_examples(ds: Dataset) -> Dataset:
    return ds.map(lambda row: {"prompt": row["input"], "context": None}, remove_columns=ds.column_names)
