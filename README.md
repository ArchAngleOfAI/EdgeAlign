<p align="center">
  <img src="edgealign_logo_v4.png" alt="EdgeAlign logo" width="220">
</p>

# EdgeAlign

> **Branch: `self-embedding`.** This branch implements the v2
> self-embedding generator variant
> (`PROMPTS/soft_prompt_generator_spec_v2_self_embedding.md`) as the
> concrete generator front-end, on top of the shared infrastructure. See
> "v2 self-embedding implementation" below for what's implemented,
> verified, and still open. `main` has the shared infrastructure with no
> variant chosen yet.

EdgeAlign is a research project for a **soft prompt generator**: a small
trainable network that compresses a prompt (and optional context) into a
short sequence of "soft prompt" prefix vectors, prepended to the token
embeddings fed into a **frozen** LLM. The frozen LLM's weights are never
updated at any stage — only the generator learns.

## How it's trained

1. **Self-distillation warm start** — the generator is taught to produce
   soft prefixes that leave the frozen model's output distribution
   effectively unchanged versus running the plain prompt with no prefix at
   all. Loss is KL divergence between the teacher (no prefix) and student
   (soft prefix) output distributions; only the generator gets gradients.
2. **RL fine-tuning** (not yet specced in detail) — starting from the
   warm-start checkpoint, real agent tool-use task success/failure is used
   as a reward signal to further fine-tune the generator only.

The end goal: an on-device coding/tool-use agent where the soft prompt
generator adapts a frozen small LLM's behavior cheaply, without ever
fine-tuning the LLM itself.

## Deployment phases

- **Phase 1 — Prototype**: Qwen 3 32B dense, frozen, on server-class
  hardware (A100s). Priority: fast iteration, validating that the method
  works at all.
- **Phase 2 — Deploy**: Qwen 3 8B dense, frozen, quantized, on a real
  customer laptop (e.g. MacBook Air class, 16GB+ unified memory). Priority:
  memory footprint and inference speed. The generator is retrained from
  scratch against the 8B model, not ported from Phase 1.

## Architecture variants (three candidates, spec'd in `PROMPTS/`)

- **v1 — Baseline (separate encoder)** — a frozen BERT-family encoder reads
  the prompt/context, feeding a 3-layer MLP (128 units, GELU) that outputs
  the soft prompt vectors.
- **v2 — Self-embedding generator** *(implemented on this branch — see
  below)* — no separate encoder; reuses the frozen LLM's own hidden
  states (last three transformer layers before the final layer,
  concatenated and mean-pooled) as the generator's input. Costs two
  forward passes through the frozen LLM per example, in exchange for
  richer representations.
- **v3 — Linear attention / SSM encoder** — replaces the BERT-family
  encoder with a frozen linear-attention or state-space model encoder (e.g.
  Mamba), to remove the short fixed-context-length ceiling that
  full-attention encoders impose.

v1 and v3 are not implemented anywhere yet — see `PROMPTS/` for the full
specs, and `MEMORY.md` for a detailed summary and current project state.

## v2 self-embedding implementation (this branch)

`edgealign/generators/self_embedding.py`'s `SelfEmbeddingGenerator`
implements the spec directly:

- No separate encoder module. It reads the frozen LLM's own hidden
  states from layers `L-3, L-2, L-1` (`L` = `num_hidden_layers`),
  concatenates them per token position (not averaged), mean-pools across
  positions (respecting the attention mask), then runs a 3-hidden-layer
  MLP (256/256/128, GELU) into the `N * embedding_dim` output.
- `hidden_size`/`num_hidden_layers` are never hand-configured — they're
  pulled automatically from whichever frozen model is actually loaded
  (`edgealign/train.py`'s `build_generator`), so they can't silently
  drift out of sync with the real checkpoint.
- This variant needs the frozen LLM's hidden states as input, which only
  exist after a forward pass through it — unlike v1/v3, which are
  self-contained. To support that without baking v2's specifics into the
  shared training loop, `GeneratorFrontend` (in
  `edgealign/generators/base.py`) gained a `needs_frozen_hidden_states`
  flag and two new optional `forward()` arguments
  (`hidden_states`, `attention_mask`); `FrozenLLMHarness.teacher_pass`
  gained an `output_hidden_states` flag. Both are no-ops for any
  front-end that doesn't opt in (the existing stub is unaffected). This
  was anticipated by the infrastructure spec's own section 3, not scope
  creep.
- Guards against a real footgun: sampling "layers `L-3..L-1`" only makes
  sense when `L >= 4` (otherwise index `L-3` would land on index 0 — the
  pre-transformer embedding output, not a transformer layer). The
  constructor raises immediately if a too-shallow model is passed in.

**Verified locally** (no GPU, no downloads): `configs/prototype_32b.yaml`
now sets `generator.type: "self_embedding"` and parses correctly.
`scripts/run_smoke_test_self_embedding.py` runs the real training loop
against a tiny 4-layer random `Qwen3ForCausalLM` and synthetic prompts —
passes, confirms the frozen model's parameters stay unchanged while the
generator's do, and confirms the hidden-state layer indices selected
(`[L-3, L-2, L-1]`) exclude both the embedding output and the final
layer as intended. The pre-existing stub smoke test (`run_smoke_test.py`)
still passes unmodified, confirming the interface extension didn't
regress the non-hidden-state path.

**Not yet done**: never run against the real Qwen 3 32B or real data —
that's the cluster's job, not this dev machine's.

## Training data (pretraining stage)

Weighted toward English code and tool-use, not general web text:
[Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) (code
coverage), **OpenCodeInstruct** (paired instruction-and-code prompts), and
**xLAM** (tool-calling prompt shapes).

## Repo layout

- `PROMPTS/` — the source-of-truth spec documents for the generator
  architectures and training pipeline.
- `edgealign/` — the shared infrastructure implementation (frozen-model
  harness, embedding injection, KL-distillation training loop, init
  scheme, dataset loaders) per
  `PROMPTS/soft_prompt_generator_infrastructure_spec.md`. The generator
  front-end (v1/v2/v3) is a pluggable interface
  (`edgealign/generators/base.py`); `edgealign/generators/self_embedding.py`
  is the real v2 implementation (this branch);
  `edgealign/generators/stub.py` is a smoke-test-only placeholder, not a
  real variant.
- `configs/prototype_32b.yaml` — the real config for the training
  cluster (Qwen 3 32B, real datasets, `generator.type: "self_embedding"`
  on this branch) — do not run it on a laptop.
- `scripts/run_smoke_test.py` / `scripts/run_smoke_test_self_embedding.py`
  — run the real training loop against a tiny random Qwen3-architecture
  model and synthetic prompts (the latter with 4 layers, needed for v2's
  layer indexing), to verify the plumbing without any GPU, download, or
  dataset access. Both build their own tiny, config-free setup.
- `scripts/prepare_stack_edu.py` — offline dataset prep for Stack-Edu
  (resolves file content from the Software Heritage S3 mirror). Run this
  once **on the cluster**, never locally — see spec section 6.1.
- `MEMORY.md` / `SHORT_MEMORY.md` / `AGENT.md` — the project's persistent
  memory and agent operating instructions (see those files for details).

## Running this

Local (no GPU, no downloads, verifies plumbing only):

```bash
conda create -n edgealign python=3.11
conda activate edgealign
pip install -r requirements.txt
python scripts/run_smoke_test.py                    # stub front-end
python scripts/run_smoke_test_self_embedding.py      # v2 front-end
```

On the training cluster (real Qwen 3 32B, real data, A100s):

```bash
# once, before training:
python scripts/prepare_stack_edu.py --output-dir /data/stack_edu_prepared

# training:
python -m edgealign.train --config configs/prototype_32b.yaml
```

`device_map: "auto"` in `configs/prototype_32b.yaml` shards the 32B
frozen model across all GPUs visible on the node (via `accelerate`) — no
manual multi-GPU/model-parallel code is needed for this "frozen giant
model + tiny trainable generator" shape.
