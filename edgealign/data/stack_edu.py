"""Stack-Edu loader/preparer -- infra spec section 6.1.

Real network/S3 operation, meant to run ONCE on the training cluster
(needs AWS credentials and a fast network link), never as a side effect
of importing this module or of running the training loop. Use
scripts/prepare_stack_edu.py to invoke `prepare()` as an offline step.
"""
import os
from typing import List, Optional

from datasets import Dataset, concatenate_datasets, load_dataset

DEFAULT_LANGUAGES = ["python", "javascript", "typescript", "shell"]


def load_raw(languages: Optional[List[str]] = None, min_int_score: Optional[int] = 3) -> Dataset:
    """Loads Stack-Edu metadata (blob_id, language, int_score, ...) for
    the given language configs. Does NOT resolve file content -- see
    resolve_content()."""
    languages = languages or DEFAULT_LANGUAGES
    ds = concatenate_datasets(
        [load_dataset("HuggingFaceTB/stack-edu", lang, split="train") for lang in languages]
    )
    if min_int_score is not None:
        ds = ds.filter(lambda row: row["int_score"] >= min_int_score)
    return ds


def resolve_content(
    ds: Dataset,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> Dataset:
    """Resolves each row's actual file content from the public Software
    Heritage S3 mirror by blob_id (SHA1), gzip-compressed. Requires
    `pip install smart_open[s3] boto3` and AWS credentials able to sign
    requests (the bucket itself is public-read)."""
    import boto3
    from smart_open import open as smart_open

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id or os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_secret_access_key or os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3 = session.client("s3")

    def _download(row):
        s3_url = f"s3://softwareheritage/content/{row['blob_id']}"
        with smart_open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
            row["content"] = fin.read().decode(row["src_encoding"], errors="replace")
        return row

    return ds.map(_download)


def to_prompt_examples(ds: Dataset) -> Dataset:
    return ds.map(lambda row: {"prompt": row["content"], "context": None}, remove_columns=ds.column_names)


def prepare(
    output_dir: str,
    languages: Optional[List[str]] = None,
    min_int_score: Optional[int] = 3,
    **aws_kwargs,
) -> Dataset:
    """Offline prep entry point: resolves content and writes it to disk
    under output_dir (as an Arrow dataset), so the training loop only
    ever does a fast local `load_from_disk`, never live S3 fetches."""
    ds = load_raw(languages, min_int_score)
    ds = resolve_content(ds, **aws_kwargs)
    ds.save_to_disk(output_dir)
    return ds
