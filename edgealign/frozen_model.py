"""Frozen base model harness (spec section 1).

Wraps a frozen causal LM (Qwen 3 32B for the prototype phase) and exposes
exactly the two forward passes the training loop needs: a teacher pass
(plain prompt, no prefix) and a student pass (soft prefix injected).
Qwen 3's weights are never updated -- every parameter has requires_grad
forced to False on construction. Gradients still flow *through* the
frozen model's activations into the soft-prefix embeddings during the
student pass, since that pass is never wrapped in torch.no_grad().
"""
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig
from .injection import inject_soft_prefix

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class FrozenLLMHarness:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, model_config: ModelConfig) -> "FrozenLLMHarness":
        """Real construction path: downloads/loads the actual checkpoint.
        Used for real training runs (prototype phase: Qwen 3 32B), not for
        local smoke testing -- see scripts/run_smoke_test.py for the
        tiny-random-model path used there instead."""
        dtype = _DTYPE_MAP[model_config.dtype]
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.name_or_path, trust_remote_code=model_config.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_config.name_or_path,
            torch_dtype=dtype,
            device_map=model_config.device_map,
            trust_remote_code=model_config.trust_remote_code,
        )
        return cls(model, tokenizer)

    @property
    def embedding_dim(self) -> int:
        return self.model.get_input_embeddings().embedding_dim

    @property
    def input_embedding_device(self):
        return self.model.get_input_embeddings().weight.device

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def teacher_pass(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Plain prompt, no soft prefix. No gradient -- this branch never
        needs one, teacher logits are a fixed target."""
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits

    def student_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        soft_prefix: torch.Tensor,
    ) -> torch.Tensor:
        """Soft prefix injected ahead of the real prompt's token
        embeddings. Deliberately no torch.no_grad() here: gradients must
        flow from the loss, through the frozen model's forward graph,
        back into soft_prefix (and from there into the generator's
        parameters) -- the frozen model's own parameters simply don't
        accumulate gradients because requires_grad is False on all of
        them."""
        token_embeds = self.embed_tokens(input_ids)
        inputs_embeds, full_attention_mask = inject_soft_prefix(token_embeds, attention_mask, soft_prefix)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask)
        return out.logits
