# SHORT_MEMORY.md — Current Working State

Purpose: if this agent session goes offline and needs to restart, read this
file first to pick up exactly where things left off. Keep it current — prune
finished items, don't let it grow stale. Long-term facts belong in
MEMORY.md, not here.

## Last updated

2026-09-15 (real Qwen3-32B smoke test / bf16 bug session)

## Status

- `main`: shared infrastructure, committed/pushed (`0a64ba5`).
- `self-embedding` branch: v2 generator (`2d9cfc5`), pushed. Not merged,
  no PR opened.
- Wikipedia dataloader + packing work from earlier this session is
  about to be committed (see "Pending" below) — full detail in
  MEMORY.md's "Pretraining data actually used" section.
- **Active right now: chasing a real bf16 numerical bug on the real
  Qwen 3 32B checkpoint.** Full diagnostic trail and the finding itself
  are in MEMORY.md's "Real-weights smoke test — Qwen 3 32B, and a real
  bf16 bug found" section — don't duplicate here. Short version: the
  frozen model's own plain forward pass (no soft prefix, nothing to do
  with our code) produces exact-zero blocks and eventual NaN in bf16,
  regardless of attention backend (default or eager), but is completely
  clean in fp32. Leading hypothesis: transformers version mismatch — the
  checkpoint's config says it was saved with `4.51.0`, cluster has
  `5.17.0` installed.
- **User's current instruction**: pin transformers to match the
  checkpoint's stated `4.51.0`, verify the bf16 bug actually goes away,
  and confirm nothing else in the codebase breaks under that older
  version. In progress: creating an isolated second venv for this
  (keeping the working `transformers==5.17.0` baseline untouched, so a
  failed experiment doesn't cost us the already-verified environment).
- Two model paths on the cluster were investigated this session — full
  detail in MEMORY.md, not repeating here — bottom line:
  `/data/models/huggingface/qwen3-32b` is the real target (plain
  Qwen3ForCausalLM, matches our code's assumptions exactly, openly
  readable); `/data/a00481223/models/Qwen3.6-27B` was a red herring (a
  multimodal hybrid-attention model) and is dropped.
- GPU etiquette this session: this cluster is shared, other users run
  real jobs. Checked `nvidia-smi` before each run; currently restricting
  to `CUDA_VISIBLE_DEVICES=4,5,6` (7 has ~27GB used by another user's
  long-running server) per explicit user instruction.

## Pending / needs user decision

- **Commit + push in progress** — Wikipedia loader/packing files plus
  the new Qwen3.6-27B/Qwen3-32B investigation scripts and MEMORY.md
  updates, all still uncommitted as of this note (mid-task when the
  transformers-downgrade request came in; finishing the commit/push
  first, per explicit instruction, before starting the downgrade test).
- Once the transformers pin is tested: does it actually fix the bf16
  instability? If yes, need to decide whether to standardize the
  cluster's real `.venv` on that older version (affects everything, not
  just this one checkpoint) or keep both environments around.
- Whether to open a PR for `self-embedding` → `main`.
- `files.zip` duplicate, v1/v3 unimplemented — long-standing, unchanged.
- `/data` cluster access, AWS creds for Stack-Edu — both still
  unresolved, unchanged from before.

## Immediate next steps

- Finish commit + push of pending work.
- Create isolated venv (e.g. `.venv_transformers451`), install
  `transformers==4.51.0` (pip resolves compatible tokenizers/hf_hub
  automatically), re-run all local-style smoke tests (stub,
  self-embedding, wikipedia-packing) there to confirm nothing else
  broke, then re-run the real Qwen3-32B smoke test to check whether the
  bf16 bug is actually gone.
- If the pin fixes it: decide with the user whether/how to make this
  the standard environment. If it doesn't: report back, next hypothesis
  needed (fp16 instead of bf16, or something else).
