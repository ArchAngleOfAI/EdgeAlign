"""Initialization scheme (spec section 5): calibrate the generator
front-end's output layer against the frozen model's real token-embedding
scale, so early-training soft prefixes aren't wildly out of distribution.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn


@torch.no_grad()
def compute_embedding_scale_stats(
    frozen_llm, sample_size: int = 10000, generator: Optional[torch.Generator] = None
) -> Dict[str, torch.Tensor]:
    """Sample real vocabulary embeddings from the frozen model and return
    their empirical mean, std, and mean norm."""
    weight = frozen_llm.model.get_input_embeddings().weight.detach().float()
    vocab_size = weight.shape[0]
    k = min(sample_size, vocab_size)
    idx = torch.randperm(vocab_size, generator=generator)[:k]
    sample = weight[idx]
    return {
        "mean": sample.mean(dim=0),
        "std": sample.std(dim=0),
        "mean_norm": sample.norm(dim=-1).mean(),
    }


def _match_out_features(target: torch.Tensor, out_features: int) -> torch.Tensor:
    """The output layer often produces N concatenated soft-prompt vectors
    (N * embedding_dim units) rather than a single embedding_dim vector.
    Tile the per-embedding-dim target stat across N repeats to match."""
    if target.numel() == out_features:
        return target
    if out_features % target.numel() == 0:
        return target.repeat(out_features // target.numel())
    raise ValueError(
        f"target stat of size {target.numel()} does not evenly tile into "
        f"out_features={out_features}"
    )


@torch.no_grad()
def rescale_output_layer_to_target_stats(
    linear: nn.Linear, target_mean: torch.Tensor, target_std: torch.Tensor, probe_batch: int = 256
) -> None:
    """Rescale linear's weight/bias in place so that, given typical-scale
    input, its output's mean and std roughly match the frozen model's real
    token embedding statistics.

    Method: probe the layer's current (freshly initialized) output
    distribution with random standard-normal input, measure its actual
    std, then scale the weight to hit the target std and set the bias to
    hit the target mean directly (the weight contribution to the mean
    washes out under zero-mean input).
    """
    device = linear.weight.device
    dtype = linear.weight.dtype

    probe_input = torch.randn(probe_batch, linear.in_features, device=device, dtype=dtype)
    probe_output = linear(probe_input)
    current_std = probe_output.std().clamp_min(1e-6)

    target_std_scalar = target_std.to(device=device, dtype=dtype).mean().clamp_min(1e-6)
    scale = target_std_scalar / current_std
    linear.weight.mul_(scale)

    if linear.bias is not None:
        target_mean_matched = _match_out_features(target_mean, linear.out_features).to(device=device, dtype=dtype)
        linear.bias.zero_()
        linear.bias.add_(target_mean_matched)
