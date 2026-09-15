# MEMORY.md — EdgeAlign Long-Term Project Memory

Durable knowledge about the project. Update by editing in place when facts
change; don't just append. See AGENT.md for how the agent should behave,
and SHORT_MEMORY.md for current in-flight work.

## Project goal

EdgeAlign is building a **soft prompt generator**: a small trainable network
that compresses a prompt (and optional context) into a short sequence of
"soft prompt" prefix vectors, which are prepended to the token embeddings
fed into a **frozen** LLM. The frozen LLM's own weights are never updated at
any stage.

Training happens in two stages:

1. **Self-distillation warm start (pretraining)** — teach the generator to
   produce soft prefixes that leave the frozen model's output distribution
   effectively unchanged vs. running the plain prompt with no prefix at all.
   Loss: KL divergence between teacher (no prefix) and student (soft
   prefix) output distributions. Only the generator's parameters get
   gradients; the frozen LLM (and any frozen encoder) never do.
2. **RL fine-tuning (not yet specced in detail)** — starting from the
   warm-start checkpoint, use real agent tool-use task success/failure as a
   reward signal to further fine-tune the generator only.

The end goal is an on-device (laptop-class) coding/tool-use agent where the
soft prompt generator adapts a frozen small LLM's behavior cheaply, without
ever fine-tuning the LLM itself.

## Two deployment phases

- **Phase 1 — Prototype**: Qwen 3 32B dense, frozen, run on server-class
  hardware (A100s). Priority: fast iteration, validate that the method
  works at all (does self-distillation converge, does the later RL loop
  improve task success). Not the deployment target.
- **Phase 2 — Deploy**: Qwen 3 8B dense, frozen, quantized, run on a real
  customer laptop (e.g. MacBook Air class, 16GB+ unified memory). Priority:
  memory footprint and inference speed, in addition to correctness.
  Checkpoints/weights from Phase 1 are **not** assumed to transfer — the
  generator must be retrained from scratch against the 8B model's different
  embedding space, using the *validated method* as a guide only.

## Architecture variants (three specs on file, not yet chosen between)

All variants share: N soft prompt vectors (start N=16–32) prepended to the
real prompt's token embeddings; output layer initialized so its scale
roughly matches empirically-sampled real Qwen 3 token embedding
mean/std, to avoid destabilizing the frozen model early in training;
optional auxiliary cosine-distance/scale-matching loss.

- **v1 — Baseline (separate encoder)**
  `PROMPTS/soft_prompt_generator_spec.md` (+ prototype/deploy variants)
  - Frozen BERT-family encoder (e.g. bert-base or distilled) reads raw
    prompt/context text → fixed-size embedding.
  - 3-layer MLP, 128 units/layer, GELU (or ReLU for simplicity) → N soft
    prompt vectors.
  - Simplest design; pays the cost of a second model (the encoder)
    building its own independent representation of the prompt.

- **v2 — Self-embedding generator**
  `PROMPTS/soft_prompt_generator_spec_v2_self_embedding.md`
  - No separate encoder. Reuses the frozen LLM's own hidden states.
  - First forward pass (no prefix) captures hidden states from the last
    three transformer layers before the final layer (L-3, L-2, L-1) —
    avoids both under-integrated early layers and the over-specialized
    final layer.
  - These three layers are **concatenated** (not averaged) per token
    position, then mean-pooled across positions into one fixed vector.
  - MLP is wider to match: 256 / 256 / 128 units, GELU.
  - Costs **two** forward passes through the full frozen LLM per example
    (one to extract hidden states, one with the prefix attached) — real
    compute cost, traded against expected representation quality.
  - Known open risk: concatenated input is large (~15,360-dim for 32B,
    ~12,288-dim for 8B, 3 layers × hidden size) compressed into a 256-unit
    first layer in one step — a possible bottleneck. If KL-to-teacher loss
    fails to converge, this is the first thing to check before concluding
    the approach doesn't work.

- **v3 — Linear attention / SSM encoder**
  `PROMPTS/soft_prompt_generator_spec_v3_linear_attention_encoder.md`
  - Replaces the BERT-family encoder with a frozen linear-attention or
    state-space model encoder (e.g. Mamba), specifically to remove the
    short fixed-context-length ceiling that full-attention encoders like
    BERT impose.
  - Candidate: Mamba checkpoints at 130M/370M/790M/1.4B/2.8B params;
    starting point is 130M or 370M to stay lightweight.
  - Also considering **truncated** Mamba checkpoints (cutting later
    layers off a pretrained checkpoint) as a cheaper-than-any-official-size
    option — to be evaluated empirically against standard sizes.
  - Known tradeoff: linear-attention/SSM fixed-size running state is lossy
    over very long inputs ("lossy memory problem") — exact early details
    can get diluted. Weigh against the benefit of no hard length ceiling.
  - MLP sizing not finalized — depends on chosen Mamba checkpoint's output
    dimension; same principle as v2 (avoid an aggressive bottleneck).
  - Pretraining/eval data for this variant should deliberately include
    long-context examples, since removing the length ceiling is the whole
    point.

No decision has been made yet on which variant to actually implement first.

## Training data (pretraining / self-distillation stage)

No task reward at this stage — just teaching the generator to preserve the
frozen model's behavior. Scope: target agent is **English-only**, focused
on **coding and tool-use**, so data is weighted toward code/tool-use rather
than general multilingual web text.

- **Stack-Edu** (`HuggingFaceTB/stack-edu`) — 125B token, English,
  educational-quality code, filtered from Stack v2 (same corpus used for
  StarCoder 2). Primary source of code coverage.
- **OpenCodeInstruct** — instruction+Python-code pairs derived from Stack
  v2 via OSS-Instruct, with LLM-generated unit tests for quality filtering.
  Covers paired instruction-and-code prompt shapes.
- **xLAM** function calling dataset — purpose-built for function
  calling/tool use fine-tuning, has a documented HF workflow. Covers
  tool-calling prompt shapes.

Mixing ratio across the three is an open, tunable hyperparameter — not
fixed by any spec.

## Success criteria

- **Pretraining stage**: student (soft-prefix) output distribution closely
  matches teacher (plain prompt) output distribution on held-out data,
  measured by average KL divergence. This is the warm-start checkpoint the
  RL stage starts from — RL never starts from random init.
- **RL stage** (not yet specced in detail): task success rate on real
  agent tool-use tasks should improve over the pretrained warm start.

## Repo structure

- `PROMPTS/` — the raw spec markdown files (source of truth for the specs
  summarized above):
  - `soft_prompt_generator_spec.md` — v1 baseline, generic version
  - `soft_prompt_generator_spec_prototype.md` — v1, Phase 1 (32B) version
  - `soft_prompt_generator_spec_deploy.md` — v1, Phase 2 (8B edge) version
  - `soft_prompt_generator_spec_v2_self_embedding.md` — v2
  - `soft_prompt_generator_spec_v3_linear_attention_encoder.md` — v3
  - `soft_prompt_generator_infrastructure_spec.md` — instructional spec for
    the infrastructure shared by v1/v2/v3 (prototype phase only): frozen
    Qwen3 32B harness, injection mechanism, KL-distillation training loop,
    init scheme, and dataset download/prep (Stack-Edu, OpenCodeInstruct,
    xLAM), with the generator front-end as a pluggable interface
- `files.zip` — a zip archive containing the same 5 original files as
  `PROMPTS/` (predates the infrastructure spec). Appears to be a
  duplicate/backup of the PROMPTS folder rather than distinct content.
  Committed as-is; ask the user before removing it.
- `edgealign_logo_v4.png` — project logo/branding asset, embedded in
  README.md.
- `edgealign/` — the shared infrastructure implementation (see
  "Implementation status" below).
- `configs/prototype_32b.yaml` — real cluster training config.
- `scripts/run_smoke_test.py` — local CPU smoke test, no downloads.
- `scripts/prepare_stack_edu.py` — offline Stack-Edu content resolution,
  meant to run once on the cluster.
- `requirements.txt`, `.gitignore` — Python deps and ignore rules
  (`.venv/`, `checkpoints/`, `__pycache__/`).
- `.claude/skills/push-changes/SKILL.md` — user-invoked (`/push-changes`)
  skill for committing/pushing in logically-grouped commits; never sets
  up CI/CD.

## Implementation status (as of 2026-09-14)

The shared infrastructure from
`soft_prompt_generator_infrastructure_spec.md` is implemented in
`edgealign/`:

- `frozen_model.py` — `FrozenLLMHarness`: loads a frozen causal LM
  (`.from_pretrained` for real runs), exposes `teacher_pass` (no-grad,
  plain prompt) and `student_pass` (soft prefix injected, gradients flow
  through the frozen model's activations into the prefix even though its
  own params never update).
- `injection.py` — `inject_soft_prefix`: the embedding-space prepend
  operation (spec section 2), used by `student_pass`.
- `generators/base.py` — `GeneratorFrontend` interface (prompts+contexts
  in, `(B, N, D)` tensor out). `generators/stub.py`'s
  `MeanPoolStubGenerator` is a smoke-test-only placeholder (hash
  tokenization + mean pool + tiny MLP) — **not** a v1/v2/v3 variant; no
  variant has been chosen for implementation yet.
- `losses.py` — `kl_distillation_loss` (KL(teacher‖student), teacher
  forcing over one shared sequence stands in for "per generation step",
  which is the standard tractable way to implement that) and
  `embedding_scale_aux_loss` (norm-matching variant of the optional
  auxiliary loss).
- `init_utils.py` — samples real embedding mean/std/mean-norm from the
  frozen model and rescales the generator's output layer to match (spec
  section 5); handles the case where the output layer produces N tiled
  copies of the embedding-dim stat.
- `data/` — real loaders for all three sources exactly as documented in
  the infra spec (`stack_edu.py` incl. S3 content resolution,
  `opencodeinstruct.py`, `xlam.py`), `pipeline.py` mixing them via
  `datasets.interleave_datasets`, and `synthetic.py` (a tiny hand-written
  in-code corpus, local-smoke-test only, never a substitute for the real
  mix).
- `train.py` / `evaluate.py` — the actual training loop and held-out KL
  tracking, config-driven (`config.py`, YAML).

**Verified locally** (conda env `edgealign`, CPU-only, no GPU on this
dev machine): `scripts/run_smoke_test.py` runs the real `run_training`
loop against a tiny randomly-initialized `Qwen3ForCausalLM` (2 layers,
hidden 32) and synthetic prompts — no network access, no dataset
download, no real Qwen 3 weights. It asserts the loss stays finite and
non-negative, the frozen tiny model's parameters are byte-for-byte
unchanged after training, and the generator's parameters do change.
Passed on first run (loss ~0.005–0.007 over 5 steps). All real-path
modules (`data/pipeline.py`, `data/stack_edu.py`, etc.) were also
confirmed to import cleanly and `configs/prototype_32b.yaml` parses
correctly — but the real S3/HF downloads and the real 32B model were
never invoked here, per instruction.

**Not yet done / explicit next steps for the cluster run**:
- No v1/v2/v3 variant is implemented — `generator.type` in
  `prototype_32b.yaml` is still `dummy_stub` and must be swapped for a
  real registered variant before a real training run.
- `scripts/prepare_stack_edu.py` has never actually been run (needs AWS
  credentials + real network access on the cluster).
- No held-out validation split is wired into `configs/prototype_32b.yaml`
  yet (eval_every is set, but `train.main()` doesn't currently pass a
  separate `eval_data_iter` — would need a small addition before relying
  on it for the real run).

## Git structure

- Repo did not exist before 2026-09-14; initialized fresh by the agent that
  day (`git init`), default branch renamed `master` → `main`.
- Remote `origin` → `git@github.com:ArchAngleOfAI/EdgeAlign.git` (SSH).
  Local `main` tracks `origin/main`.
- Commit identity (local to this repo, not global git config):
  `ArchAngleOfAI <ArchAngleOfAI@users.noreply.github.com>` — the user's own
  email is deliberately not used for commits.
- SSH auth for this machine: key at `~/.ssh/id_ed25519_github`, pinned to
  `github.com` via `~/.ssh/config`, public key added to the user's GitHub
  account by the user.
- Initial commit `8349752` pushed 2026-09-14, containing all files above.
- Per AGENT.md, the agent does not commit or push automatically — it
  stages/edits and notifies the user, and confirms before pushing.
