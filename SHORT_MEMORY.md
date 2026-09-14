# SHORT_MEMORY.md — Current Working State

Purpose: if this agent session goes offline and needs to restart, read this
file first to pick up exactly where things left off. Keep it current — prune
finished items, don't let it grow stale. Long-term facts belong in
MEMORY.md, not here.

## Last updated

2026-09-14

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
- No implementation code exists yet. Still at the spec stage. No variant
  (v1/v2/v3) has been chosen for implementation yet.

## Pending / needs user decision

- No decision yet on which architecture variant (v1/v2/v3) to implement
  first as the pluggable generator front-end, or whether to implement more
  than one for comparison.
- `files.zip` duplicates `PROMPTS/` contents exactly — committed as-is,
  flagged in MEMORY.md, no action taken pending user input.

## Immediate next steps (once user gives direction)

- Likely next task: start implementing the shared infrastructure
  (`soft_prompt_generator_infrastructure_spec.md`) plus whichever variant
  spec the user picks first as the generator front-end.
- No RL-stage spec exists yet in PROMPTS/ — only referenced as "next stage,
  not covered" in every spec. May need to be written before that stage.
- Since we now push to GitHub: remember AGENT.md's git rule (notify before
  committing) — pushing is an additional, even more visible step; confirm
  with the user before pushing future commits, same as before.
