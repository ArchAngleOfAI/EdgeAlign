"""Embedding-space soft-prefix injection (spec section 2).

Identical no matter which generator front-end produced the soft vectors --
implemented once here, independent of the front-end module, and used by
FrozenLLMHarness.student_pass.
"""
import torch


def inject_soft_prefix(token_embeds: torch.Tensor, attention_mask: torch.Tensor, soft_prefix: torch.Tensor):
    """Prepend soft_prefix to token_embeds in embedding space.

    Args:
        token_embeds: (B, L, D) real prompt token embeddings, untouched.
        attention_mask: (B, L) mask for the real prompt tokens.
        soft_prefix: (B, N, D) generated soft prompt vectors.

    Returns:
        (inputs_embeds, full_attention_mask): (B, N+L, D) and (B, N+L).
    """
    soft_prefix = soft_prefix.to(device=token_embeds.device, dtype=token_embeds.dtype)
    inputs_embeds = torch.cat([soft_prefix, token_embeds], dim=1)
    prefix_mask = attention_mask.new_ones(soft_prefix.shape[:2])
    full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
    return inputs_embeds, full_attention_mask
