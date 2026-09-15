"""Real-weights smoke test against the actual Qwen 3 32B checkpoint
cached on the training cluster.

Unlike the other smoke tests (tiny random models, run anywhere -- see
run_smoke_test.py / run_smoke_test_self_embedding.py), this loads the
REAL 32B pretrained weights via FrozenLLMHarness.from_pretrained and
must run on a machine with real GPUs and access to the checkpoint --
never locally. Uses the v2 self-embedding generator with
hidden_size/num_hidden_layers auto-derived from the loaded model's own
real config (edgealign/train.py's build_generator), not hand-specified.

Everything else is kept minimal (batch size 1, short sequences, a
handful of steps) -- the point is catching integration issues against
the real checkpoint's real dimensions and real device placement
(device_map="auto" shards it across whichever GPUs are visible), not
producing a useful checkpoint or benchmarking anything.

Usage (on the cluster, restricting to known-free GPUs):
    CUDA_VISIBLE_DEVICES=1,3,4,5,6 python scripts/run_smoke_test_qwen3_32b_real.py [model_path]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from edgealign.config import GeneratorConfig, ModelConfig, TrainConfig
from edgealign.data.synthetic import synthetic_batches
from edgealign.frozen_model import FrozenLLMHarness
from edgealign.init_utils import compute_embedding_scale_stats, rescale_output_layer_to_target_stats
from edgealign.train import build_generator, run_training

DEFAULT_MODEL_PATH = "/data/models/huggingface/qwen3-32b"


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    torch.manual_seed(0)

    print(f"Loading real Qwen3-32B from {model_path} (bfloat16, device_map=auto)...")
    model_cfg = ModelConfig(name_or_path=model_path, dtype="bfloat16", device_map="auto")
    frozen_llm = FrozenLLMHarness.from_pretrained(model_cfg)

    devices_used = sorted({str(p.device) for p in frozen_llm.model.parameters()})
    print(f"Loaded. embedding_dim={frozen_llm.embedding_dim}, devices in use: {devices_used}")

    embedding_stats = compute_embedding_scale_stats(frozen_llm)
    print(f"Real embedding stats probed: mean_norm={embedding_stats['mean_norm'].item():.4f}")

    generator_cfg = GeneratorConfig(type="self_embedding", n_soft_tokens=8, extra={})
    generator = build_generator(generator_cfg, frozen_llm.embedding_dim, frozen_llm.model.config)
    rescale_output_layer_to_target_stats(
        generator.output_layer_for_init(), embedding_stats["mean"], embedding_stats["std"]
    )
    generator.to(frozen_llm.input_embedding_device)
    print(
        f"Generator built: n_soft_tokens=8, hidden_size={generator.hidden_size}, "
        f"num_hidden_layers={generator.num_hidden_layers}, layer indices={generator._layer_indices}"
    )

    train_cfg = TrainConfig(
        learning_rate=0.01, max_steps=3, eval_every=0, aux_loss_weight=0.01,
        log_every=1, seed=0, output_dir="/tmp/edgealign_smoke_test_qwen32b_real",
    )
    data_iter = synthetic_batches(batch_size=1, num_batches=train_cfg.max_steps, seed=0)

    frozen_param_before = next(frozen_llm.model.parameters()).detach().clone()
    generator_param_before = next(generator.parameters()).detach().clone()

    history = run_training(
        frozen_llm, generator, data_iter, train_cfg, max_seq_len=64, embedding_stats=embedding_stats,
    )

    frozen_param_after = next(frozen_llm.model.parameters())
    generator_param_after = next(generator.parameters())

    assert len(history) == train_cfg.max_steps, f"expected {train_cfg.max_steps} steps, got {len(history)}"
    assert all(torch.isfinite(torch.tensor(v)) for v in history), f"non-finite loss: {history}"
    assert torch.equal(frozen_param_before, frozen_param_after), (
        "frozen Qwen3-32B parameters changed during training -- must never happen!"
    )
    assert not torch.equal(generator_param_before, generator_param_after), (
        "generator parameters did not change -- training step is not updating the generator!"
    )

    print("\nSMOKE TEST PASSED (real Qwen 3 32B weights, self-embedding generator)")
    print(f"loss trajectory: {[round(v, 4) for v in history]}")


if __name__ == "__main__":
    main()
