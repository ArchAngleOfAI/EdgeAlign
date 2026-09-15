"""Generator front-end interface (spec section 3).

This module boundary is the ONLY thing that differs between v1 (separate
BERT encoder), v2 (self-embedding, taps the frozen LLM's own hidden
states), and v3 (linear attention / Mamba encoder) -- see PROMPTS/ for
each variant's spec. No variant has been chosen for implementation yet.
The training loop, injection mechanism, and evaluation code in this
package depend only on this interface, never on a specific variant's
internals.
"""
from typing import List, Optional

import torch
import torch.nn as nn


class GeneratorFrontend(nn.Module):
    """Takes a batch of prompts (plus optional context strings) and
    returns exactly n_soft_tokens vectors of embedding_dim per example.

    Concrete subclasses set self.n_soft_tokens and self.embedding_dim in
    __init__, and are responsible for their own tokenization/encoding --
    the interface intentionally does not assume a shared tokenizer with
    the frozen main LLM, since v1's BERT encoder and v2's "reuse the main
    LLM's hidden states" approach have fundamentally different input
    requirements.
    """

    n_soft_tokens: int
    embedding_dim: int

    def forward(self, prompts: List[str], contexts: Optional[List[str]] = None) -> torch.Tensor:
        """Returns a (batch, n_soft_tokens, embedding_dim) tensor."""
        raise NotImplementedError

    def output_layer_for_init(self) -> nn.Linear:
        """Returns the final nn.Linear layer whose output scale should be
        calibrated against the frozen model's real token-embedding
        statistics (spec section 5). Every concrete variant must expose
        one."""
        raise NotImplementedError
