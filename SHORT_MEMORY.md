# SHORT_MEMORY.md — Current Working State

Purpose: if this agent session goes offline and needs to restart, read this
file first to pick up exactly where things left off. Keep it current — prune
finished items, don't let it grow stale. Long-term facts belong in
MEMORY.md, not here.

## Last updated

2026-09-14 (implementation session)

## Status

- Git repo initialized, connected, and pushed. Remote: `origin` →
  `git@github.com:ArchAngleOfAI/EdgeAlign.git` (SSH). Local `main` tracks
  `origin/main`, both at commit `8349752`, working tree clean.
- Commit identity for this repo (local, not global git config):
  `ArchAngleOfAI <ArchAngleOfAI@users.noreply.github.com>`.
- SSH auth set up on this machine: key at `~/.ssh/id_ed25519_github`,
  `~/.ssh/config` has a `Host github.com` entry pinning that key,
  `~/.ssh/known_hosts` has GitHub's host key. `ssh -T git@github.com`
  confirms auth as `ArchAngleOfAI`.
- Created the standing files: AGENT.md, MEMORY.md, SHORT_MEMORY.md,
  README.md (with logo embedded).
- Wrote `PROMPTS/soft_prompt_generator_infrastructure_spec.md` — an
  instructional spec extracting the infrastructure shared by all three
  generator variants (v1/v2/v3): frozen Qwen3 32B harness, injection
  mechanism, KL-distillation training loop, init scheme, and verified
  download/prep instructions for the three training datasets (Stack-Edu,
  OpenCodeInstruct, xLAM), with the generator front-end left as a
  pluggable interface. Scoped to prototype phase only (32B, server
  hardware) — deploy phase (8B edge) intentionally deferred.
- Also created `.claude/skills/push-changes/SKILL.md` — user-invoked
  `/push-changes` skill for logically-grouped commit+push, handles
  pre-commit hook failures by fixing root causes, never sets up CI/CD.
- **Implemented the shared infrastructure** in `edgealign/` per
  `soft_prompt_generator_infrastructure_spec.md` (harness, injection,
  KL-distillation loop, init scheme, all three real dataset loaders,
  mixing, held-out eval wiring). Full detail in MEMORY.md's
  "Implementation status" section — don't duplicate it here, it's
  durable now, not in-flight.
- Verified locally via `scripts/run_smoke_test.py` in a conda env named
  `edgealign` (this dev machine has no GPU: CPU-only, torch 2.14+cpu,
  transformers 5.17, python 3.11) — tiny random `Qwen3ForCausalLM`, no
  downloads, no dataset access. Passed. Real-path modules (dataset
  loaders, real config) verified to import/parse cleanly but never
  executed (no S3/HF downloads triggered), per instruction.
- **None of this implementation work is committed or pushed yet** —
  working tree has it staged-ready but untracked/modified. Waiting for
  the user to say go (or invoke `/push-changes`).

## Pending / needs user decision

- No decision yet on which architecture variant (v1/v2/v3) to implement
  as the real generator front-end — `edgealign/train.py`'s
  `GENERATOR_REGISTRY` only has the smoke-test-only stub registered.
  `configs/prototype_32b.yaml`'s `generator.type` must be swapped before
  any real cluster run.
- `files.zip` duplicates `PROMPTS/` contents exactly — committed as-is,
  flagged in MEMORY.md, no action taken pending user input.
- Implementation changes above are uncommitted — confirm before pushing.

## Immediate next steps (once user gives direction)

- Pick and implement one of v1/v2/v3 as a real `GeneratorFrontend`
  registered in `edgealign/train.py`.
- Actually run `scripts/prepare_stack_edu.py` and a real training run —
  both need to happen on the user's A100 cluster, not this dev machine.
- No RL-stage spec exists yet in PROMPTS/ — only referenced as "next stage,
  not covered" in every spec. May need to be written before that stage.
- Remember AGENT.md's git rule (notify before committing/pushing) and the
  new reporting rule (explain major implementation work clearly, which
  this entry + the chat response already did for this milestone).
