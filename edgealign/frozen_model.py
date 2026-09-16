"""Frozen base model harness (spec section 1).

Wraps a frozen causal LM (Qwen 3 32B for the prototype phase) and exposes
exactly the two forward passes the training loop needs: a teacher pass
(plain prompt, no prefix) and a student pass (soft prefix injected).
Qwen 3's weights are never updated -- every parameter has requires_grad
forced to False on construction. Gradients still flow *through* the
frozen model's activations into the soft-prefix embeddings during the
student pass, since that pass is never wrapped in torch.no_grad().
"""
from typing import Dict, List, Optional

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

        max_memory = None
        if model_config.max_memory_gib is not None and torch.cuda.is_available():
            cap = f"{model_config.max_memory_gib}GiB"
            max_memory = {i: cap for i in range(torch.cuda.device_count())}

        model = AutoModelForCausalLM.from_pretrained(
            model_config.name_or_path,
            torch_dtype=dtype,
            device_map=model_config.device_map,
            max_memory=max_memory,
            trust_remote_code=model_config.trust_remote_code,
        )

        devices = {str(p.device) for p in model.parameters()}
        if "meta" in devices:
            raise RuntimeError(
                f"model partially loaded onto the meta device -- devices seen: {devices}. "
                f"device_map=\"auto\" couldn't find enough memory across the visible GPUs at "
                f"load time. On a shared cluster this can be genuine insufficient capacity, or "
                f"transient contention from another process at the exact moment of loading "
                f"(confirmed to happen -- see MEMORY.md); check GPU memory and retry before "
                f"assuming more/larger GPUs are needed."
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

    def teacher_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        """Plain prompt, no soft prefix. No gradient -- this branch never
        needs one, teacher logits are a fixed target.

        Returns just the logits by default. When output_hidden_states is
        True (needed by hidden-state-based generator front-ends such as
        v2), returns (logits, hidden_states) instead, where hidden_states
        is the tuple HF returns: index 0 is the embedding layer's output,
        index i (1 <= i <= num_hidden_layers) is transformer layer i's
        output.
        """
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )
        if output_hidden_states:
            return out.logits, out.hidden_states
        return out.logits

    def teacher_pass_selected_layers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_indices: List[int],
    ):
        """Like teacher_pass(output_hidden_states=True), but captures
        ONLY the requested layers via forward hooks instead of asking
        the model to materialize every layer's output.

        output_hidden_states=True forces HF to retain all
        num_hidden_layers+1 intermediate tensors regardless of how many
        are actually used downstream -- at long sequence lengths this is
        a real, large memory cost (e.g. ~20GB for a 37-layer model at
        seq_len=16384, batch=4, even though a generator like
        self-embedding only reads 3 of them). This method only ever
        holds the requested layers in memory.

        layer_indices use the same convention as the output_hidden_states
        tuple: index 0 is the embedding output (not reachable here -- use
        output_hidden_states=True if index 0 is ever needed), index i
        (1 <= i <= num_hidden_layers) is decoder layer i's output, i.e.
        `self.model.model.layers[i - 1]`'s output.

        Returns (logits, {index: tensor}).
        """
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(idx: int):
            def hook(module, inputs, output):
                captured[idx] = output[0] if isinstance(output, tuple) else output

            return hook

        decoder_layers = self.model.model.layers
        for idx in layer_indices:
            if idx < 1:
                raise ValueError(
                    f"layer index {idx} is out of range for hook-based capture -- index 0 "
                    f"(the embedding output) isn't a decoder layer; use output_hidden_states=True instead"
                )
            handles.append(decoder_layers[idx - 1].register_forward_hook(make_hook(idx)))

        try:
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for handle in handles:
                handle.remove()

        return out.logits, captured

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
