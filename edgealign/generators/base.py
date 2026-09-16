"""Generator front-end interface (spec section 3).

This module boundary is the ONLY thing that differs between v1 (separate
BERT encoder), v2 (self-embedding, taps the frozen LLM's own hidden
states), and v3 (linear attention / Mamba encoder) -- see PROMPTS/ for
each variant's spec. The training loop, injection mechanism, and
evaluation code in this package depend only on this interface, never on
a specific variant's internals.

v1/v3-style variants are fully self-contained: they take raw text and do
their own tokenization/encoding, with no dependency on the frozen main
LLM. v2 (self-embedding) is different by design -- it has no separate
encoder and instead needs the frozen LLM's own hidden states as its
input, which only exist after a forward pass through that LLM. The
`needs_frozen_hidden_states` flag, `required_hidden_state_layers`, and
the extra `hidden_states`/`attention_mask` forward() arguments below
exist ONLY to support that case; a front-end that doesn't set the flag
never receives them and can ignore all of it.
"""
from typing import Dict, List, Optional

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

    # Set True by a variant (only v2 today) that needs the frozen main
    # LLM's own hidden states as input. When True, the training loop
    # captures ONLY the layers named in required_hidden_state_layers
    # (via FrozenLLMHarness.teacher_pass_selected_layers -- forward
    # hooks, not output_hidden_states=True) and passes the result
    # through to forward() below instead of relying on prompts alone.
    # A generator that sets needs_frozen_hidden_states=True MUST also
    # set required_hidden_state_layers to a concrete list of indices
    # (the output_hidden_states-tuple convention: index i, 1 <=
    # i <= num_hidden_layers, is decoder layer i's output; index 0, the
    # embedding output, is not capturable this way).
    #
    # Why this matters, not just an optimization: output_hidden_states=
    # True forces the frozen model to retain every layer's output
    # simultaneously, which at long sequence lengths is a real, large
    # memory cost regardless of how many layers are actually used
    # downstream (e.g. ~20GB for a 37-layer 8B model at seq_len=16384,
    # batch=4, to read only 3 layers) -- confirmed by an actual OOM
    # during real-config integration testing, not just theory.
    needs_frozen_hidden_states: bool = False
    required_hidden_state_layers: Optional[List[int]] = None

    def forward(
        self,
        prompts: List[str],
        contexts: Optional[List[str]] = None,
        hidden_states: Optional[Dict[int, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns a (batch, n_soft_tokens, embedding_dim) tensor.

        hidden_states/attention_mask are only ever populated when
        needs_frozen_hidden_states is True (see above) -- variants that
        don't set that flag can ignore both parameters entirely.
        hidden_states is keyed by the same indices listed in
        required_hidden_state_layers, not a plain 0-indexed sequence.
        """
        raise NotImplementedError

    def output_layer_for_init(self) -> nn.Linear:
        """Returns the final nn.Linear layer whose output scale should be
        calibrated against the frozen model's real token-embedding
        statistics (spec section 5). Every concrete variant must expose
        one."""
        raise NotImplementedError
