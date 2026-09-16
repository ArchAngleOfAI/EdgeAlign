"""Evaluation / success criterion (spec section 7): track KL divergence
between teacher and student output distributions on held-out data."""
from typing import Iterable, List, Optional

import torch

from .frozen_model import FrozenLLMHarness
from .generators.base import GeneratorFrontend
from .losses import kl_distillation_loss


@torch.no_grad()
def evaluate(
    frozen_llm: FrozenLLMHarness,
    generator: GeneratorFrontend,
    data_iter: Iterable[List[str]],
    max_seq_len: int,
    max_batches: Optional[int] = None,
    kl_top_k: Optional[int] = None,
    kl_chunk_size: Optional[int] = None,
) -> float:
    generator.eval()
    total_kl, n_batches = 0.0, 0
    for prompts in data_iter:
        if max_batches is not None and n_batches >= max_batches:
            break
        tokenized = frozen_llm.tokenizer(
            prompts, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt"
        )
        device = frozen_llm.input_embedding_device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        if getattr(generator, "needs_frozen_hidden_states", False):
            # Hook-based selective capture -- see train.py's training_step
            # and generators/base.py's required_hidden_state_layers for why
            # (avoids output_hidden_states=True's memory cost at long
            # sequence lengths).
            teacher_logits, hidden_states = frozen_llm.teacher_pass_selected_layers(
                input_ids, attention_mask, generator.required_hidden_state_layers
            )
            soft_prefix = generator(prompts, hidden_states=hidden_states, attention_mask=attention_mask)
        else:
            teacher_logits = frozen_llm.teacher_pass(input_ids, attention_mask)
            soft_prefix = generator(prompts)

        student_logits = frozen_llm.student_pass(input_ids, attention_mask, soft_prefix)

        kl = kl_distillation_loss(
            teacher_logits, student_logits, generator.n_soft_tokens, attention_mask,
            top_k=kl_top_k, chunk_size=kl_chunk_size,
        )
        total_kl += kl.item()
        n_batches += 1

    generator.train()
    return total_kl / max(n_batches, 1)
