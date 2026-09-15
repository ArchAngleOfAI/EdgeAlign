"""Smoke test for the Wikipedia tar-streaming loader + packing +
"wikipedia" data mode (see MEMORY.md "Pretraining data actually used").

Builds a tiny synthetic tar file with the exact same structure as the
real wiki-18.jsonl (a tar containing one .jsonl member, lines shaped
{"id": ..., "contents": ...}), then:
  1. Verifies edgealign.data.wikipedia.iter_passages reads it correctly.
  2. Verifies edgealign.data.packing.pack_texts actually reduces many
     short passages into fewer, longer packed strings.
  3. Runs the real training loop (edgealign.train.build_data_iterators +
     run_training) end-to-end against the tiny random Qwen3 model used
     by the other smoke tests, with mode="wikipedia" pointed at the
     synthetic tar.

No network access, no real Qwen 3 weights, no real Wikipedia data --
see scripts/check_wikipedia_on_cluster.py for the real-data check (run
on the cluster, not here).

Usage: python scripts/run_smoke_test_wikipedia_packing.py
"""
import json
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from edgealign.config import DataConfig, TrainConfig
from edgealign.data.packing import pack_texts
from edgealign.data.wikipedia import iter_passages
from edgealign.init_utils import compute_embedding_scale_stats, rescale_output_layer_to_target_stats
from edgealign.train import build_data_iterators, run_training
from edgealign.generators.stub import MeanPoolStubGenerator

from run_smoke_test import build_tiny_harness, TINY_VOCAB_SIZE

FAKE_PASSAGES = [
    f'"Fake Article {i}"\n' + " ".join([f"word{j}" for j in range(20)])
    for i in range(50)
]


def build_synthetic_wiki_tar(tmp_dir: Path) -> Path:
    """Writes a tiny tar archive with the same layout as the real
    wiki-18.jsonl: one .jsonl member, one JSON object per line."""
    jsonl_path = tmp_dir / "wiki_dump.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, passage in enumerate(FAKE_PASSAGES):
            f.write(json.dumps({"id": str(i), "contents": passage}) + "\n")

    tar_path = tmp_dir / "wiki-18.jsonl"  # matches the real (misleading) name
    with tarfile.open(tar_path, mode="w") as tar:
        tar.add(jsonl_path, arcname="data00/fake/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl")

    return tar_path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        tar_path = build_synthetic_wiki_tar(tmp_dir)

        # 1. iter_passages reads the tar correctly.
        passages = list(iter_passages(str(tar_path), shuffle_buffer_size=10, seed=0))
        assert len(passages) == len(FAKE_PASSAGES), f"expected {len(FAKE_PASSAGES)} passages, got {len(passages)}"
        assert set(passages) == set(FAKE_PASSAGES), "passage content mismatch after streaming/shuffling"
        assert passages != FAKE_PASSAGES, "shuffle produced the exact original order -- buffer shuffle isn't doing anything"
        print(f"[1/3] iter_passages: read {len(passages)} passages, order shuffled as expected")

        # 2. pack_texts actually reduces many short passages into fewer, longer strings.
        max_seq_len = 128
        packed = list(pack_texts(iter(FAKE_PASSAGES), max_seq_len=max_seq_len, separator="</s>"))
        assert len(packed) < len(FAKE_PASSAGES), (
            f"packing should merge multiple passages per output, got {len(packed)} packed from "
            f"{len(FAKE_PASSAGES)} inputs -- no reduction happened"
        )
        assert all("</s>" in p or FAKE_PASSAGES[0] in p for p in packed), "separator missing from packed output"
        avg_len = sum(len(p) for p in packed) / len(packed)
        print(f"[2/3] pack_texts: {len(FAKE_PASSAGES)} passages -> {len(packed)} packed strings (avg {avg_len:.0f} chars)")

        # 3. Full training loop via build_data_iterators(mode="wikipedia").
        torch.manual_seed(0)
        frozen_llm = build_tiny_harness()
        embedding_stats = compute_embedding_scale_stats(frozen_llm, sample_size=200)

        generator = MeanPoolStubGenerator(
            n_soft_tokens=4, embedding_dim=frozen_llm.embedding_dim,
            vocab_size=TINY_VOCAB_SIZE, embed_dim=16, hidden=32,
        )
        rescale_output_layer_to_target_stats(
            generator.output_layer_for_init(), embedding_stats["mean"], embedding_stats["std"]
        )

        data_cfg = DataConfig(
            mode="wikipedia",
            max_seq_len=32,
            batch_size=2,
            eval_fraction=0.5,
            max_eval_batches=3,
            wikipedia_tar_path=str(tar_path),
            wikipedia_shuffle_buffer_size=10,
        )
        train_iter, eval_batches = build_data_iterators(data_cfg, eos_token="</s>")
        assert eval_batches is not None and len(eval_batches) > 0, "expected non-empty eval batches"
        print(f"[3/3] build_data_iterators(wikipedia): {len(eval_batches)} eval batches materialized")

        train_cfg = TrainConfig(
            learning_rate=0.01, max_steps=5, eval_every=2, aux_loss_weight=0.01,
            log_every=1, seed=0, output_dir="/tmp/edgealign_smoke_test_wikipedia",
        )

        frozen_param_before = next(frozen_llm.model.parameters()).clone()
        generator_param_before = next(generator.parameters()).clone()

        history = run_training(
            frozen_llm, generator, train_iter, train_cfg, max_seq_len=data_cfg.max_seq_len,
            embedding_stats=embedding_stats, eval_data_iter=eval_batches,
        )

        frozen_param_after = next(frozen_llm.model.parameters())
        generator_param_after = next(generator.parameters())

        assert len(history) == train_cfg.max_steps, f"expected {train_cfg.max_steps} steps, got {len(history)}"
        assert all(torch.isfinite(torch.tensor(v)) for v in history), f"non-finite loss: {history}"
        assert torch.equal(frozen_param_before, frozen_param_after), "frozen params changed!"
        assert not torch.equal(generator_param_before, generator_param_after), "generator params did not change!"

    print("\nSMOKE TEST PASSED (wikipedia loader + packing + full training loop)")
    print(f"loss trajectory: {[round(v, 4) for v in history]}")


if __name__ == "__main__":
    main()
