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

        teacher_logits = frozen_llm.teacher_pass(input_ids, attention_mask)
        soft_prefix = generator(prompts)
        student_logits = frozen_llm.student_pass(input_ids, attention_mask, soft_prefix)

        kl = kl_distillation_loss(teacher_logits, student_logits, generator.n_soft_tokens, attention_mask)
        total_kl += kl.item()
        n_batches += 1

    generator.train()
    return total_kl / max(n_batches, 1)
