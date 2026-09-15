"""xLAM function-calling loader -- infra spec section 6.3. Content is
inline, no separate resolution step needed."""
import json

from datasets import Dataset, load_dataset


def load_raw() -> Dataset:
    return load_dataset("Salesforce/xlam-function-calling-60k", split="train")


def _format(row):
    try:
        tools = json.loads(row["tools"])
        tools_text = "\n".join(str(t) for t in tools)
    except (json.JSONDecodeError, TypeError):
        tools_text = str(row["tools"])
    prompt = f"<user>{row['query']}</user>\n\n<tools>\n{tools_text}\n</tools>"
    return {"prompt": prompt, "context": None}


def to_prompt_examples(ds: Dataset) -> Dataset:
    return ds.map(_format, remove_columns=ds.column_names)
