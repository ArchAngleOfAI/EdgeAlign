"""Placeholder generator front-end -- NOT one of the v1/v2/v3 variants.

No architecture decision has been made yet on which of the three specced
variants (separate BERT encoder / self-embedding / linear-attention
encoder) to actually implement. This stub exists solely so the shared
infrastructure (injection, KL-distillation loop, init scheme) can be
exercised end-to-end in tests before that decision is made. Do not use it
for a real training run, and do not treat it as a fourth variant -- it is
deliberately as simple as possible (hash-based tokenization, mean
pooling, tiny MLP) since its only job is to produce gradient-carrying,
correctly-shaped output.
"""
from typing import List, Optional

import torch
import torch.nn as nn

from .base import GeneratorFrontend


class MeanPoolStubGenerator(GeneratorFrontend):
    def __init__(
        self,
        n_soft_tokens: int,
        embedding_dim: int,
        vocab_size: int = 8192,
        embed_dim: int = 32,
        hidden: int = 64,
    ):
        super().__init__()
        self.n_soft_tokens = n_soft_tokens
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, n_soft_tokens * embedding_dim)

    def _hash_tokenize(self, text: str) -> torch.Tensor:
        ids = [(hash(tok) % (self.vocab_size - 1)) + 1 for tok in text.split()]
        if not ids:
            ids = [0]
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, prompts: List[str], contexts: Optional[List[str]] = None) -> torch.Tensor:
        device = self.out.weight.device
        pooled = []
        for i, prompt in enumerate(prompts):
            text = f"{contexts[i]} {prompt}" if contexts else prompt
            ids = self._hash_tokenize(text).to(device)
            pooled.append(self.token_embedding(ids).mean(dim=0))
        pooled_batch = torch.stack(pooled, dim=0)

        hidden = self.mlp(pooled_batch)
        out = self.out(hidden)
        return out.view(len(prompts), self.n_soft_tokens, self.embedding_dim)

    def output_layer_for_init(self) -> nn.Linear:
        return self.out
