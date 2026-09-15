"""Training objective (spec section 4): self-distillation KL loss plus the
optional embedding-scale auxiliary loss.

Both passes are computed via teacher forcing over the same input sequence
(causal attention makes each position's logits equivalent to the
"next-token given tokens so far" distribution a free-running generation
step would produce, without needing to actually decode step by step) --
this is the standard, computationally tractable way to implement
"KL divergence ... averaged over generation steps" for this pipeline.
"""
import torch
import torch.nn.functional as F


def kl_distillation_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    n_soft_tokens: int,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """KL(teacher || student), averaged over valid (non-padding) token
    positions.

    student_logits has n_soft_tokens extra leading positions (the prefix)
    that have no teacher counterpart -- those are sliced off before
    comparing. Position N+i in the student sequence corresponds to
    position i in the teacher sequence, since the soft prefix only adds
    positions in front and never touches the real prompt tokens.
    """
    student_aligned = student_logits[:, n_soft_tokens:, :]
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = F.log_softmax(student_aligned.float(), dim=-1)

    # F.kl_div(input=log_student, target=log_teacher, log_target=True)
    # computes sum(teacher_p * (log_teacher_p - log_student_p)) = KL(teacher || student).
    kl = F.kl_div(student_logp, teacher_logp, log_target=True, reduction="none").sum(dim=-1)

    if attention_mask is not None:
        mask = attention_mask.float()
        return (kl * mask).sum() / mask.sum().clamp_min(1.0)
    return kl.mean()


def embedding_scale_aux_loss(soft_vectors: torch.Tensor, target_mean_norm: torch.Tensor) -> torch.Tensor:
    """Norm-matching variant of the optional auxiliary loss (spec section
    4, item 5): penalizes the generated vectors' norm for drifting away
    from the frozen model's real token-embedding norm scale."""
    norms = soft_vectors.norm(dim=-1)
    target = target_mean_norm.to(device=norms.device, dtype=norms.dtype).expand_as(norms)
    return F.mse_loss(norms, target)
