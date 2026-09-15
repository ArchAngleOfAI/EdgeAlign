# SHORT_MEMORY.md — Current Working State

Purpose: if this agent session goes offline and needs to restart, read this
file first to pick up exactly where things left off. Keep it current — prune
finished items, don't let it grow stale. Long-term facts belong in
MEMORY.md, not here.

## Last updated

2026-09-15 (self-embedding branch session)

## Status

- `main` is fully committed and pushed: shared infrastructure
  (`edgealign/`), the `/push-changes` skill, docs — all landed at commit
  `0a64ba5` (pushed). See MEMORY.md's "Implementation status" section
  for what's actually in `main`; not repeating it here.
- **Currently on branch `self-embedding`**, created from `main` at
  `0a64ba5`, specifically to implement v2
  (`soft_prompt_generator_spec_v2_self_embedding.md`) as a real,
  registered generator front-end. Full detail of what was built and
  verified is in MEMORY.md's "`self-embedding` branch" subsection — not
  duplicating it here.
- Local dev-machine setup (unchanged, still true): conda env
  `edgealign`, CPU-only (no GPU on this machine), torch 2.14+cpu,
  transformers 5.17, python 3.11. GitHub SSH auth also unchanged — key
  at `~/.ssh/id_ed25519_github`, remote `git@github.com:ArchAngleOfAI/EdgeAlign.git`.
- **This branch's changes are NOT committed or pushed yet** — working
  tree on `self-embedding` has the v2 implementation, its two smoke
  tests, the interface extension, and README/MEMORY updates, all
  uncommitted as of this note. Waiting for the user to say go.

## Pending / needs user decision

- Whether/when to commit+push this branch, and whether to open a PR
  against `main` or keep iterating on the branch first.
- `files.zip` duplicates `PROMPTS/` contents exactly — long-standing,
  still unaddressed, still fine to ignore unless the user raises it.
- v1 and v3 remain unimplemented — no decision yet on whether either
  gets its own branch next, or whether self-embedding proceeds straight
  to a real cluster run first.

## Immediate next steps (once user gives direction)

- Commit + push this branch (likely via `/push-changes`, though note:
  that skill didn't register via the Skill tool this session — had to
  follow its procedure manually. Worth checking again next session
  whether it registers normally.)
- Real next milestone for this branch: an actual run against Qwen 3 32B
  and real data on the user's A100 cluster — nothing here has touched
  real weights or real datasets yet, by design.
- No RL-stage spec exists yet in PROMPTS/ — only referenced as "next
  stage, not covered" in every spec. Still may need to be written
  eventually.
- Remember AGENT.md's git rule (notify before committing/pushing) and
  reporting rule (explain major implementation work clearly) — both
  already applied this session in the chat response.
