"""v2 -- self-embedding generator front-end.

PROMPTS/soft_prompt_generator_spec_v2_self_embedding.md: no separate
encoder. Reuses the frozen main LLM's own hidden states from the last
three transformer layers before the final layer (L-3, L-2, L-1),
concatenated per token position (not averaged -- concatenation preserves
each layer's distinct signal) and mean-pooled across token positions,
then fed through a 3-hidden-layer MLP (256/256/128, GELU) to produce the
N soft prompt vectors.

Only this MLP is trained. There is no separate encoder to train, and the
frozen LLM is never updated -- this module only ever *reads* hidden
states the frozen LLM already computed during the teacher pass; it never
runs the frozen LLM itself. That's why it declares
needs_frozen_hidden_states = True: the training loop is responsible for
running the teacher pass with output_hidden_states=True and handing the
result to forward() (see edgealign/train.py training_step and
edgealign/evaluate.py evaluate).

Cost tradeoff, per spec: this requires two forward passes through the
full frozen LLM per example (the teacher/hidden-state-extraction pass,
plus the soft-prefix student pass) rather than one pass through a small
separate encoder -- a real compute cost, traded for richer, more
integrated representations than an independent encoder would build.
"""
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .base import GeneratorFrontend

# Minimum number of transformer layers the frozen LLM must have for
# "the last three layers before the final layer" to make sense at all
# (L-3 must be >= 1, i.e. a real post-transformer-layer hidden state,
# never index 0, which is the pre-transformer embedding output).
_MIN_HIDDEN_LAYERS = 4


class SelfEmbeddingGenerator(GeneratorFrontend):
    needs_frozen_hidden_states = True

    def __init__(
        self,
        n_soft_tokens: int,
        embedding_dim: int,
        hidden_size: int,
        num_hidden_layers: int,
        mlp_hidden_1: int = 256,
        mlp_hidden_2: int = 256,
        mlp_hidden_3: int = 128,
    ):
        super().__init__()
        if num_hidden_layers < _MIN_HIDDEN_LAYERS:
            raise ValueError(
                f"self-embedding generator needs the frozen LLM to have at least "
                f"{_MIN_HIDDEN_LAYERS} transformer layers to sample layers L-3..L-1 "
                f"without hitting the embedding layer (index 0); got num_hidden_layers="
                f"{num_hidden_layers}"
            )

        self.n_soft_tokens = n_soft_tokens
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        # hidden_states tuple indices for layers L-3, L-2, L-1 (L = num_hidden_layers),
        # excluding the final layer L and the pre-transformer embedding output at index 0.
        self._layer_indices = [num_hidden_layers - 3, num_hidden_layers - 2, num_hidden_layers - 1]

        concat_dim = 3 * hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, mlp_hidden_1),
            nn.GELU(),
            nn.Linear(mlp_hidden_1, mlp_hidden_2),
            nn.GELU(),
            nn.Linear(mlp_hidden_2, mlp_hidden_3),
            nn.GELU(),
        )
        self.out = nn.Linear(mlp_hidden_3, n_soft_tokens * embedding_dim)

    def forward(
        self,
        prompts: List[str],
        contexts: Optional[List[str]] = None,
        hidden_states: Optional[Sequence[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states is None or attention_mask is None:
            raise ValueError(
                "SelfEmbeddingGenerator requires hidden_states and attention_mask from "
                "the frozen LLM's teacher pass (needs_frozen_hidden_states=True) -- the "
                "training loop must call teacher_pass(..., output_hidden_states=True) "
                "and pass the result through."
            )

        device = self.out.weight.device
        selected = [hidden_states[i].to(device) for i in self._layer_indices]
        concatenated = torch.cat(selected, dim=-1).float()  # (B, T, 3*hidden_size)

        mask = attention_mask.to(device).unsqueeze(-1).float()  # (B, T, 1)
        pooled = (concatenated * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # (B, 3*hidden_size)
        pooled = pooled.to(self.mlp[0].weight.dtype)

        hidden = self.mlp(pooled)
        out = self.out(hidden)
        return out.view(len(prompts), self.n_soft_tokens, self.embedding_dim)

    def output_layer_for_init(self) -> nn.Linear:
        return self.out
