"""Wikipedia passage corpus loader (see MEMORY.md "Pretraining data
actually used" for the full rationale).

The corpus on the training cluster ships as `wiki-18.jsonl`, but despite
the extension it is actually a **tar archive** containing one real
jsonl file inside it. This loader reads that member directly via
Python's tarfile module, streaming -- it never extracts the ~13.4GB
archive to disk (this cluster's root filesystem has already run out of
space once; do not repeat that mistake here).

Each line of the underlying jsonl is:
    {"id": "<n>", "contents": "\"<Article Title>\"\n<~100-word passage>"}
(the FlashRAG-style wiki_dpr_100w DPR passage split).
"""
import io
import json
import random
import tarfile
from typing import Iterator, Optional


def _find_jsonl_member(tar: tarfile.TarFile) -> tarfile.TarInfo:
    for member in tar.getmembers():
        if member.isfile() and member.name.endswith(".jsonl"):
            return member
    raise ValueError("no .jsonl member found inside the wiki-18.jsonl tar archive")


def iter_passages(
    tar_path: str,
    shuffle_buffer_size: int = 10000,
    seed: int = 0,
    max_passages: Optional[int] = None,
) -> Iterator[str]:
    """Streams passage text (the "contents" field) from the Wikipedia
    tar archive, with an approximate shuffle via a fixed-size reservoir
    buffer -- the corpus is far too large to shuffle in memory outright.
    """
    rng = random.Random(seed)
    buffer = []
    n_yielded = 0

    with tarfile.open(tar_path, mode="r") as tar:
        member = _find_jsonl_member(tar)
        fileobj = tar.extractfile(member)
        if fileobj is None:
            raise ValueError(f"could not stream member {member.name} from {tar_path}")

        with io.TextIOWrapper(fileobj, encoding="utf-8") as text_stream:
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                passage = json.loads(line)["contents"]

                if len(buffer) < shuffle_buffer_size:
                    buffer.append(passage)
                    continue

                idx = rng.randrange(shuffle_buffer_size)
                to_yield = buffer[idx]
                buffer[idx] = passage
                yield to_yield
                n_yielded += 1
                if max_passages is not None and n_yielded >= max_passages:
                    return

    rng.shuffle(buffer)
    for passage in buffer:
        yield passage
        n_yielded += 1
        if max_passages is not None and n_yielded >= max_passages:
            return
