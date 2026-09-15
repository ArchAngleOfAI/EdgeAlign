"""Real-data sanity check for the Wikipedia tar loader + packing.

Unlike scripts/run_smoke_test_wikipedia_packing.py (synthetic tar, runs
anywhere), this points at the ACTUAL wiki-18.jsonl tar archive and reads
a small, bounded number of real passages -- meant to run on the training
cluster only, where that file actually exists. Cheap and read-only: it
never extracts the archive, never touches the frozen model, never
trains anything. Just confirms the real file's tar/jsonl structure
matches what edgealign/data/wikipedia.py expects, and that packing
behaves sensibly on real passage lengths.

Usage: python scripts/check_wikipedia_on_cluster.py [tar_path]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgealign.data.packing import pack_texts
from edgealign.data.wikipedia import iter_passages

DEFAULT_TAR_PATH = "/data/r50058044/reskill_search/retriever/wiki-18.jsonl"
N_PASSAGES = 500
MAX_SEQ_LEN = 512


def main() -> None:
    tar_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TAR_PATH
    print(f"Reading up to {N_PASSAGES} real passages from: {tar_path}")

    passages = list(iter_passages(tar_path, shuffle_buffer_size=1000, seed=0, max_passages=N_PASSAGES))
    assert len(passages) == N_PASSAGES, f"expected {N_PASSAGES} passages, got {len(passages)}"

    lengths = [len(p) for p in passages]
    print(f"Read {len(passages)} passages.")
    print(f"Passage length (chars): min={min(lengths)} max={max(lengths)} avg={sum(lengths)/len(lengths):.0f}")
    print(f"First passage (truncated to 200 chars): {passages[0][:200]!r}")

    packed = list(pack_texts(iter(passages), max_seq_len=MAX_SEQ_LEN, separator="</s>"))
    packed_lengths = [len(p) for p in packed]
    print(f"\nPacked {len(passages)} passages -> {len(packed)} packed strings at max_seq_len={MAX_SEQ_LEN}")
    print(f"Packed string length (chars): min={min(packed_lengths)} max={max(packed_lengths)} "
          f"avg={sum(packed_lengths)/len(packed_lengths):.0f}")
    print(f"Estimated tokens per packed string (chars/4): "
          f"~{sum(packed_lengths)/len(packed_lengths)/4:.0f} (target was {MAX_SEQ_LEN})")

    print("\nREAL-DATA CHECK PASSED")


if __name__ == "__main__":
    main()
