# SHORT_MEMORY.md — Current Working State

Purpose: if this agent session goes offline and needs to restart, read this
file first to pick up exactly where things left off. Keep it current — prune
finished items, don't let it grow stale. Long-term facts belong in
MEMORY.md, not here.

## Last updated

2026-09-15 (switched working target to Qwen 3 8B)

## Status

- `main`: shared infrastructure, committed/pushed (`0a64ba5`).
- `self-embedding` branch, all pushed as of last note:
  `2d9cfc5` (v2 generator), `74506dc` (Wikipedia dataloader),
  `86dcc44` (real Qwen3-32B smoke test + bf16 bug found). **This
  session's Qwen3-8B switch (configs/qwen3_8b.yaml, new smoke test
  scripts, doc updates below) is NOT committed/pushed yet.**
- **Decided: prototyping directly on Qwen 3 8B for now, not the
  originally-intended 32B.** 32B has an unresolved real bf16 bug in its
  own frozen forward pass (tried and disproved: transformers version
  pin 4.51.0 — didn't fix it, made it worse). 8B is fully verified:
  forward path clean, and a full real-weights training-loop smoke test
  passes with a sane loss trajectory. Along the way, also found and
  fixed a *second*, unrelated issue: the smoke tests' hardcoded
  `learning_rate=0.01` (borrowed from tiny-toy-model tests) caused a
  finite-but-wildly-unstable loss on real 8B activations; `0.0001`
  (matching the real training config) is stable. Fixed in both real
  smoke test scripts, plus added a sanity-bound assertion
  (`max(history) < 20`) so this class of "technically finite but
  actually broken" result can't silently read as PASSED again.
- **This status is now reflected everywhere**, per explicit request:
  `MEMORY.md` (new banner at the top, updated Phase 1 description,
  full "Working target switched" section), `README.md` (branch banner,
  repo layout, running instructions), `PROMPTS/soft_prompt_generator_infrastructure_spec.md`
  and `PROMPTS/soft_prompt_generator_spec_v2_self_embedding.md`
  (implementation notes near their respective "target the 32B" lines).
  `configs/qwen3_8b.yaml` is new (the config actually in use);
  `configs/prototype_32b.yaml` kept as-is with a warning, ready for
  when the bug is fixed.
- GPU etiquette unchanged: check `nvidia-smi` before each run. No
  orphaned processes/GPU memory left behind after any run this session
  — verified each time.

## Pending / needs user decision

- **Commit + push this session's work** — the 8B switch (config,
  scripts, LR fix) and all the doc updates above are uncommitted.
- Root cause of the 32B bf16 bug is still technically open (depth vs.
  multi-GPU sharding, undisambiguated) but is no longer blocking
  progress now that 8B is the working target — revisit only if/when
  32B is needed again.
- Whether/when to clean up `.venv_tf451` on the cluster (transformers
  4.51.0 test env, no longer needed now that the version-pin hypothesis
  is closed) — not urgent, ~55GB still free there.
- Whether to open a PR for `self-embedding` → `main`.
- `files.zip` duplicate, v1/v3 unimplemented, `/data` cluster access,
  AWS creds for Stack-Edu — all long-standing, unchanged.

## Immediate next steps

- Commit + push.
- Real (non-smoke) training run on Qwen 3 8B + Wikipedia data
  (`configs/qwen3_8b.yaml`), to see if KL distillation actually
  converges over a real run, not just a few smoke-test steps.
- No RL-stage spec exists yet in PROMPTS/.
