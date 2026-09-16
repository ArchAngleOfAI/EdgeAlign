"""Training objective (spec section 4): self-distillation KL loss plus the
optional embedding-scale auxiliary loss.

Both passes are computed via teacher forcing over the same input sequence
(causal attention makes each position's logits equivalent to the
"next-token given tokens so far" distribution a free-running generation
step would produce, without needing to actually decode step by step) --
this is the standard, computationally tractable way to implement
"KL divergence ... averaged over generation steps" for this pipeline.
"""
from typing import Optional

import torch
import torch.nn.functional as F

# Liger Kernel's LigerKLDIVLoss is a fused-Triton-kernel drop-in
# replacement for nn.KLDivLoss/F.kl_div (~15% memory reduction on the
# reduction step at large vocab sizes -- see MEMORY.md for why this
# matters: at real vocab_size=151936 and long sequences, the KL
# computation itself was OOMing). Triton requires CUDA, so this is only
# ever used on GPU; CPU (all local dev/smoke-test runs) always falls
# back to the plain F.kl_div path below, unchanged.
try:
    from liger_kernel.transformers import LigerKLDIVLoss

    _liger_kl_module = LigerKLDIVLoss(reduction="none", log_target=True)
except ImportError:
    _liger_kl_module = None


def _kl_over_logits(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) per position, given already-aligned,
    already-restricted-to-whatever-vocabulary-subset logits of matching
    shape (..., V). Shared by both the chunked and unchunked paths."""
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)

    # input=log_student, target=log_teacher, log_target=True computes
    # sum(teacher_p * (log_teacher_p - log_student_p)) = KL(teacher || student),
    # via Liger's fused kernel on GPU, F.kl_div (identical math) on CPU.
    if _liger_kl_module is not None and student_logp.is_cuda:
        # Liger's kernel expects a flattened (N, vocab) input, not
        # (..., vocab) -- confirmed by direct testing (numerically
        # identical to F.kl_div once reshaped, max abs diff ~1e-7).
        *lead_dims, vocab = student_logp.shape
        kl_flat = _liger_kl_module(student_logp.reshape(-1, vocab), teacher_logp.reshape(-1, vocab))
        return kl_flat.sum(dim=-1).view(*lead_dims)
    return F.kl_div(student_logp, teacher_logp, log_target=True, reduction="none").sum(dim=-1)


def kl_distillation_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    n_soft_tokens: int,
    attention_mask: torch.Tensor = None,
    top_k: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """KL(teacher || student), averaged over valid (non-padding) token
    positions.

    student_logits has n_soft_tokens extra leading positions (the prefix)
    that have no teacher counterpart -- those are sliced off before
    comparing. Position N+i in the student sequence corresponds to
    position i in the teacher sequence, since the soft prefix only adds
    positions in front and never touches the real prompt tokens.

    top_k/chunk_size (both None by default -- exact, unchanged behavior,
    what every existing smoke test still exercises): at the real
    vocab_size=151936, casting the full (batch, seq, vocab) logits to
    float32 for log_softmax is itself a large allocation (~10GB at
    batch=4, seq=4096) -- this happens BEFORE Liger's kernel is ever
    involved, so LigerKLDIVLoss alone doesn't fix it (confirmed by
    hitting this exact OOM with Liger already wired in -- see MEMORY.md).
    - top_k: restrict the KL sum to the teacher's top-K most probable
      tokens per position (an approximation -- the dropped tail's
      contribution is assumed negligible, which holds well for a
      reasonably peaked distribution). Student's logits are gathered at
      the SAME vocab indices the teacher's top-K selected, since KL(teacher
      || student) sums over teacher's support. Both are then
      renormalized via log_softmax over just those K entries -- this is
      the standard top-k renormalization (matching top-k sampling), not
      an attempt to reconstruct the true full-vocab probabilities.
    - chunk_size: process chunk_size sequence positions at a time instead
      of the whole sequence at once, discarding each chunk's intermediates
      before the next. Composes with top_k (the gather/top-k step also
      runs per chunk, bounding its overhead too), and is still exact on
      its own (no top_k) -- just trades a python-level loop for lower
      peak memory.
    """
    student_aligned = student_logits[:, n_soft_tokens:, :]
    batch, seq_len, vocab = teacher_logits.shape
    step = chunk_size if chunk_size is not None else seq_len

    kl_chunks = []
    for start in range(0, seq_len, step):
        end = min(start + step, seq_len)
        teacher_chunk = teacher_logits[:, start:end, :]
        student_chunk = student_aligned[:, start:end, :]

        if top_k is not None:
            # Top-K by the TEACHER's logits (KL(teacher || student) sums
            # over teacher's support) -- gather student's logits at the
            # SAME vocab indices, not student's own top-K.
            teacher_topk, topk_idx = teacher_chunk.topk(top_k, dim=-1)
            student_topk = torch.gather(student_chunk, dim=-1, index=topk_idx)
            kl_chunks.append(_kl_over_logits(teacher_topk, student_topk))
        else:
            kl_chunks.append(_kl_over_logits(teacher_chunk, student_chunk))

    kl = torch.cat(kl_chunks, dim=1) if len(kl_chunks) > 1 else kl_chunks[0]

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
