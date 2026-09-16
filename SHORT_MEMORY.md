# SHORT_MEMORY.md — Current Working State

Purpose: if this agent session goes offline and needs to restart, read this
file first to pick up exactly where things left off. Keep it current — prune
finished items, don't let it grow stale. Long-term facts belong in
MEMORY.md, not here.

## Last updated

2026-09-16 (real-scale training infra + GPU fault investigation, all
committed/pushed)

## Status

- `main`: shared infrastructure, committed/pushed (`0a64ba5`).
- `self-embedding` branch: everything through this session is now
  committed and pushed. See MEMORY.md for the full technical detail;
  this is the short version.
- **Training config system is real now**: `edgealign/config.py`'s
  `TrainConfig` supports `grad_accum_steps`, `optimizer`/`optimizer_kwargs`,
  `lr_scheduler`/`warmup_steps`/`lr_scheduler_kwargs`, `num_gpus`/
  `available_gpus` (sets `CUDA_VISIBLE_DEVICES` before any CUDA call),
  and `checkpoint_every` (periodic save + auto-resume in
  `run_training`).
- **Two real bugs found and fixed** while scaling toward the user's
  target config (`seq_len=16384, batch=4, grad_accum=4, 4 GPUs`):
  hook-based selective hidden-state capture (was retaining all 37
  layers via `output_hidden_states=True`), and the KL loss's full-vocab
  float32 materialization (fixed via Liger Kernel + top-k + sequence
  chunking, all verified numerically exact/near-exact against the
  original). Both are in `edgealign/losses.py`/`edgealign/frozen_model.py`.
- **A long GPU-fault investigation concluded**: what looked like "GPU 4
  is faulty" (an earlier, honest-but-wrong finding) is actually a rare,
  non-reproducible transient CUDA fault that occurs GPU-independently —
  proven by running the same real step simultaneously on all 6 idle
  GPUs and having all 6 pass right after GPU 7 alone had just failed
  twice. Along the way, found and fixed two more real things: a genuine
  OOM at `seq_len=4096` (confirmed via `torch.autograd.set_detect_anomaly`,
  not assumed), and a real device-mismatch bug in
  `edgealign/init_utils.py` (`torch.randperm` defaulting to CPU while
  indexing a CUDA tensor). Also found (separately, still open): multi-GPU
  `device_map="auto"` sharding silently corrupts Qwen3-8B's numerics —
  confirmed via a clean single-GPU-vs-2-GPU logit comparison. Practical
  fix applied: single-GPU only for now (see `configs/qwen3_8b.yaml`).
- **Built a checkpoint + restart recovery system** for the (confirmed
  sticky, no safe in-process recovery) CUDA fault:
  `TrainConfig.checkpoint_every` + auto-resume in `run_training`,
  `main()` exits with a sentinel code on that specific fault class, and
  `scripts/train_with_restart.sh` restarts a fresh process on that
  code only. Verified locally (resume test) and on the cluster
  (10/10 correct restarts during a sporadic-fault run).
- GPU 1 has been independently reported (by the cluster's other users)
  to currently be out of order — avoid it. This is unrelated to the
  now-disproven "GPU 4" finding.
- GPU etiquette unchanged: check `nvidia-smi` before each run.

## Pending / needs user decision

- No real (non-smoke, to-convergence) training run has been started
  yet — this session was entirely verification/infrastructure/debugging.
  Next real attempt should use `configs/qwen3_8b.yaml` via
  `scripts/train_with_restart.sh`, single GPU, `seq_len=2048`.
- The multi-GPU `device_map="auto"` sharding-corruption bug is
  unresolved (root cause not isolated) — blocks getting back to the
  originally-requested `seq_len=16384`/multi-GPU config; a single
  40GB GPU is the real current memory ceiling.
- The sporadic CUDA fault itself is still unexplained at the root
  (best guess: rare transient condition on this heavily shared node);
  the checkpoint+restart system mitigates it but doesn't fix it, and
  does NOT help if it starts occurring on every attempt in a row
  (observed once, under unusually heavy concurrent node load).
- Whether/when to clean up `.venv_tf451` and the newer
  `.venv_torch_stable` on the cluster (both were disproof-of-hypothesis
  throwaway envs, no longer needed) — not urgent.
- Whether to open a PR for `self-embedding` → `main`.
- `files.zip` duplicate, v1/v3 unimplemented, AWS creds for Stack-Edu —
  all long-standing, unchanged.

## Immediate next steps

- Kick off a real (longer, to-convergence) training run via
  `scripts/train_with_restart.sh configs/qwen3_8b.yaml`.
- Revisit the multi-GPU sharding bug if/when more memory budget is
  needed (currently blocked at single-GPU, `seq_len=2048`).
- No RL-stage spec exists yet in PROMPTS/.
