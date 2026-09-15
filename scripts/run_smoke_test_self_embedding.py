"""Smoke test for the v2 self-embedding generator
(PROMPTS/soft_prompt_generator_spec_v2_self_embedding.md).

Runs the real training loop (edgealign.train.run_training /
training_step) against a tiny, randomly initialized Qwen3-architecture
model and synthetic prompts -- no network access, no dataset download,
no real Qwen 3 weights.

Uses 4 transformer layers (not 2, like the baseline smoke test) because
the self-embedding generator reads layers L-3, L-2, L-1 (excluding the
pre-transformer embedding output at index 0 and the final layer L) --
that indexing only produces real, distinct transformer-layer outputs
when L >= 4.

Usage: python scripts/run_smoke_test_self_embedding.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from edgealign.config import TrainConfig
from edgealign.data.synthetic import synthetic_batches
from edgealign.frozen_model import FrozenLLMHarness
from edgealign.generators.self_embedding import SelfEmbeddingGenerator
from edgealign.init_utils import compute_embedding_scale_stats, rescale_output_layer_to_target_stats
from edgealign.train import run_training

from run_smoke_test import TinyWhitespaceTokenizer, TINY_VOCAB_SIZE

TINY_HIDDEN_LAYERS = 4


def build_tiny_harness() -> FrozenLLMHarness:
    config = Qwen3Config(
        vocab_size=TINY_VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=TINY_HIDDEN_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config)
    tokenizer = TinyWhitespaceTokenizer(vocab_size=TINY_VOCAB_SIZE)
    return FrozenLLMHarness(model, tokenizer)


def main() -> None:
    torch.manual_seed(0)

    frozen_llm = build_tiny_harness()
    embedding_stats = compute_embedding_scale_stats(frozen_llm, sample_size=200)

    generator = SelfEmbeddingGenerator(
        n_soft_tokens=4,
        embedding_dim=frozen_llm.embedding_dim,
        hidden_size=frozen_llm.model.config.hidden_size,
        num_hidden_layers=frozen_llm.model.config.num_hidden_layers,
        mlp_hidden_1=32,
        mlp_hidden_2=32,
        mlp_hidden_3=16,
    )
    rescale_output_layer_to_target_stats(
        generator.output_layer_for_init(), embedding_stats["mean"], embedding_stats["std"]
    )

    train_cfg = TrainConfig(
        learning_rate=0.01,
        max_steps=5,
        eval_every=0,
        aux_loss_weight=0.01,
        log_every=1,
        seed=0,
        output_dir="/tmp/edgealign_smoke_test_self_embedding",
    )

    frozen_param_before = next(frozen_llm.model.parameters()).clone()
    generator_param_before = next(generator.parameters()).clone()

    data_iter = synthetic_batches(batch_size=2, num_batches=train_cfg.max_steps, seed=0)
    history = run_training(
        frozen_llm, generator, data_iter, train_cfg, max_seq_len=32, embedding_stats=embedding_stats
    )

    frozen_param_after = next(frozen_llm.model.parameters())
    generator_param_after = next(generator.parameters())

    assert len(history) == train_cfg.max_steps, f"expected {train_cfg.max_steps} steps, got {len(history)}"
    assert all(torch.isfinite(torch.tensor(v)) for v in history), f"non-finite loss encountered: {history}"
    assert all(v >= -1e-4 for v in history), f"KL divergence should never be negative: {history}"
    assert torch.equal(frozen_param_before, frozen_param_after), (
        "frozen Qwen3 parameters changed during training -- the frozen model must never receive gradient updates!"
    )
    assert not torch.equal(generator_param_before, generator_param_after), (
        "generator parameters did not change -- the training step is not actually updating the generator!"
    )

    print("\nSMOKE TEST PASSED (self-embedding generator)")
    print(f"loss trajectory: {[round(v, 4) for v in history]}")
    print("- frozen Qwen3 (tiny, 4 layers) parameters: unchanged, as required")
    print("- generator parameters: updated, as expected")
    print(f"- read hidden_states at layer indices {generator._layer_indices} (L={TINY_HIDDEN_LAYERS})")


if __name__ == "__main__":
    main()
