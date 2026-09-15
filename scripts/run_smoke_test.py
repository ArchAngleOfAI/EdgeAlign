"""Smoke test for the shared infrastructure
(PROMPTS/soft_prompt_generator_infrastructure_spec.md).

Runs the REAL training loop (edgealign.train.run_training) against a
tiny, randomly initialized Qwen3-architecture model and a handful of
synthetic prompts -- no network access, no dataset download, no real
Qwen 3 weights. This only verifies the plumbing (shapes, gradient flow,
injection alignment, frozen-model isolation); it says nothing about
whether any generator variant actually learns anything useful on the
real 32B model -- that only happens on the training cluster.

Usage: python scripts/run_smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from edgealign.config import TrainConfig
from edgealign.data.synthetic import synthetic_batches
from edgealign.frozen_model import FrozenLLMHarness
from edgealign.generators.stub import MeanPoolStubGenerator
from edgealign.init_utils import compute_embedding_scale_stats, rescale_output_layer_to_target_stats
from edgealign.train import run_training

TINY_VOCAB_SIZE = 512


class TinyWhitespaceTokenizer:
    """Whitespace/hash tokenizer standing in for a real tokenizer.
    Test-only -- never used for real training, where the actual Qwen 3
    tokenizer is loaded via FrozenLLMHarness.from_pretrained()."""

    def __init__(self, vocab_size: int = TINY_VOCAB_SIZE, pad_token_id: int = 0, eos_token_id: int = 1):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __call__(self, texts, padding=True, truncation=True, max_length=32, return_tensors="pt"):
        sequences = []
        for text in texts:
            ids = [(hash(tok) % (self.vocab_size - 2)) + 2 for tok in text.split()]
            if truncation:
                ids = ids[:max_length]
            sequences.append(ids or [self.eos_token_id])

        max_len = max(len(seq) for seq in sequences)
        input_ids, attention_mask = [], []
        for seq in sequences:
            pad_len = max_len - len(seq)
            input_ids.append(seq + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(seq) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def build_tiny_harness() -> FrozenLLMHarness:
    config = Qwen3Config(
        vocab_size=TINY_VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
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

    generator = MeanPoolStubGenerator(
        n_soft_tokens=4,
        embedding_dim=frozen_llm.embedding_dim,
        vocab_size=TINY_VOCAB_SIZE,
        embed_dim=16,
        hidden=32,
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
        output_dir="/tmp/edgealign_smoke_test",
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

    print("\nSMOKE TEST PASSED")
    print(f"loss trajectory: {[round(v, 4) for v in history]}")
    print("- frozen Qwen3 (tiny) parameters: unchanged, as required")
    print("- generator parameters: updated, as expected")


if __name__ == "__main__":
    main()
