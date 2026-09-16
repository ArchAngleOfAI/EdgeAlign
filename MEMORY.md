# MEMORY.md — EdgeAlign Long-Term Project Memory

Durable knowledge about the project. Update by editing in place when facts
change; don't just append. See AGENT.md for how the agent should behave,
and SHORT_MEMORY.md for current in-flight work.

> **Current status (2026-09-15): prototyping is happening directly on
> Qwen 3 8B, not the originally-intended 32B.** The 32B checkpoint has
> an unresolved real bf16 numerical bug (frozen model's own forward pass
> produces NaN/zero degeneration — see "Real-weights smoke test — Qwen 3
> 32B" below); 8B's forward path and a full training-loop smoke test
> both check out clean. This is a practical substitution within Phase 1
> (prototype), not a move to Phase 2 (which means 8B *quantized* on
> laptop hardware — this is plain 8B on the same A100 cluster). See
> "Working target switched to Qwen 3 8B for now" below for full detail.
> Everywhere below that says "32B" is prototype-phase, historical
> context from before this switch, or refers to `configs/prototype_32b.yaml`
> kept ready for when the bug is fixed — read current instructions
> (`configs/qwen3_8b.yaml`, `scripts/run_smoke_test_qwen3_8b_real.py`)
> as the actual working target for now.

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
  improve task success). Not the deployment target. **As of 2026-09-15,
  actually prototyping directly on Qwen 3 8B instead** (32B blocked on
  an unresolved bf16 bug — see the banner at the top of this file and
  "Working target switched to Qwen 3 8B for now" below). Still Phase 1
  in spirit (fast iteration, method validation), just a smaller model
  than originally specced, and still unquantized/server-hardware unlike
  Phase 2 below.
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
- `scripts/prepare_stack_edu.py` has never actually been run (needs AWS
  credentials + real network access on the cluster).
- (Fixed same day, no longer open: held-out eval is now wired into
  `train.main()` via `build_data_iterators`/`eval_fraction`.)

## Implementation status — `self-embedding` branch (2026-09-15)

Branched from `main` at commit `0a64ba5` specifically to implement v2
(`PROMPTS/soft_prompt_generator_spec_v2_self_embedding.md`) as a real,
registered generator front-end. `main` itself still has no variant
chosen — this work exists only on this branch until/unless merged.

- `edgealign/generators/self_embedding.py`'s `SelfEmbeddingGenerator`:
  reads the frozen LLM's own hidden states at layers `L-3, L-2, L-1`
  (`L` = `num_hidden_layers`, from the loaded model's real config, never
  hand-specified), concatenates them per token position, mean-pools
  across positions (attention-mask aware), 3-hidden-layer MLP
  (256/256/128, GELU) → `N * embedding_dim` output. Raises immediately
  if the frozen model has fewer than 4 transformer layers (below that,
  index `L-3` would hit the pre-transformer embedding output, not a real
  transformer layer — the spec's indexing silently stops making sense).
- **Shared-infrastructure extension required and made**: v2 needs the
  frozen LLM's hidden states as input, which only exist after a forward
  pass through it — unlike v1/v3, which take raw text and are fully
  self-contained. This was anticipated by
  `soft_prompt_generator_infrastructure_spec.md` section 3 but not fully
  specified. Added: `GeneratorFrontend.needs_frozen_hidden_states` flag
  and two new optional `forward()` args (`hidden_states`,
  `attention_mask`) in `generators/base.py`;
  `FrozenLLMHarness.teacher_pass` gained an `output_hidden_states` flag
  (returns `(logits, hidden_states)` when True, else just `logits` as
  before); `train.py`'s `training_step`/`build_generator` and
  `evaluate.py`'s `evaluate` branch on that flag. All of this is a no-op
  for generators that don't set the flag — confirmed by re-running the
  pre-existing stub smoke test unmodified after the change (still
  passes).
- `train.py`'s `build_generator` now also accepts the frozen model's HF
  config and auto-fills `hidden_size`/`num_hidden_layers` for any
  generator that needs them, rather than trusting a YAML-specified value
  that could drift out of sync with whichever checkpoint is actually
  loaded.
- `configs/prototype_32b.yaml` on this branch: `generator.type:
  "self_embedding"`.

**Verified locally** (same conda env `edgealign`, CPU-only, no GPU):
`scripts/run_smoke_test_self_embedding.py` — a tiny 4-layer random
`Qwen3ForCausalLM` (deeper than the baseline smoke test's 2 layers,
specifically to exercise real `L-3..L-1` indexing), synthetic prompts,
real `run_training`/`training_step`. Passed: finite non-negative loss,
frozen params unchanged, generator params updated, selected layer
indices confirmed `[1, 2, 3]` for `L=4` (excludes index 0 embedding
output and index 4 final layer, as intended). Also unit-verified
`build_generator`'s auto-fill logic directly against a mock 32B-shaped
config (`hidden_size=5120, num_hidden_layers=64` → indices `[61,62,63]`)
and the shallow-model guard (raises for `num_hidden_layers=3`).

**Not yet done**: never run against the real Qwen 3 32B or real data —
cluster-only, not this dev machine. v1/v3 remain unimplemented.

## GPU training cluster (connected 2026-09-15)

- Access: `ssh -p 2030 a84460786@184.150.234.220`. From this dev machine,
  use the alias `edgealign-cluster` (configured in `~/.ssh/config`,
  pointing at a dedicated key `~/.ssh/id_ed25519_cluster` — separate
  from the GitHub key). Auth is key-only; the user's password was never
  shared with the agent (by design — see AGENT.md-level judgment: never
  accept passwords through chat, always set up key auth instead).
- Hardware: 8× NVIDIA A100-PCIE-40GB, driver 595.58.03, CUDA 13.2. Shared
  multi-tenant box — other users' jobs run concurrently (saw one at 100%
  GPU util on first login). Host: `ptlabb03`.
- **Disk is the real constraint here**: `/` (where `$HOME` lives) is a
  557.9GB volume and was at 99% full (as low as ~2.9GB free, briefly 0
  free) before the user manually freed space; sits around 60GB free as
  of this session's end. `/data` is a separate 4.4TB LVM volume (6×
  NVMe drives) with 2.4TB free, clearly the intended place for large
  installs/models/datasets on this box (other users have subdirectories
  there) — **this account has no access to it** (`mkdir` at top level →
  permission denied; not in any group that grants access). `/tmp` and
  `/var/tmp` are writable but are NOT separate mounts — same root disk,
  no extra capacity. No other block devices exist per `lsblk`/`fstab`.
- **Incident**: installing GPU-enabled torch via pip in `$HOME` (the
  only writable large-ish space available) drove the shared root disk
  to 100% full mid-install (torch's CUDA-bundled wheel alone is
  ~4.1GB). Cleaned up immediately (removed the partial venv + pip/
  virtualenv caches), disk returned to its prior ~7GB-free baseline, no
  lasting damage. User then freed ~50GB manually, after which the full
  install succeeded normally.
- No `conda` on this box, and the system Python (3.12.3) has no
  `python3-venv` OS package installed (and no sudo available to this
  account) — stdlib `venv` fails with a `ensurepip` error. Workaround:
  `pip install --user --break-system-packages virtualenv` (bypasses the
  PEP 668 externally-managed-environment guard safely, since `--user`
  never touches OS-managed packages), then
  `~/.local/bin/virtualenv .venv` to create environments normally.
- Working environment: `~/EdgeAlign/.venv` (via the virtualenv
  workaround above), `pip install -r requirements.txt` →
  torch 2.14.0+cu130 (CUDA available, all 8 GPUs visible), transformers
  5.17.0, datasets 5.0.1. Confirmed via
  `python -c "import torch; torch.cuda.is_available()"` → True.
- Repo cloned at `~/EdgeAlign` (plain HTTPS, no auth needed — the GitHub
  repo is public). Cloned shallow by default; unshallowed
  (`git fetch --unshallow`) and fetched `self-embedding` explicitly
  (`git clone --depth 1`/plain `clone` only grabs the default branch's
  ref by default). Currently checked out on `self-embedding` at
  `2d9cfc5`.
- **Both smoke tests verified passing on this cluster**
  (`run_smoke_test.py`, `run_smoke_test_self_embedding.py`) — confirms
  the GPU-enabled environment actually works end-to-end, not just that
  it imports. Neither the real Qwen 3 32B nor real datasets have been
  touched here yet.
- **Open question before any real run**: `/data` access is still
  unresolved. A real Qwen 3 32B training run's checkpoints/logs, plus
  the Stack-Edu content resolution step, will likely need more headroom
  and a stabler location than the shared, historically-99%-full root
  disk. Revisit before committing to a long real run.

## Pretraining data actually used — Wikipedia corpus (2026-09-15)

Surveyed the cluster for existing data we could reuse, regardless of
whether it matched the code/tool-use scope in the specs. Findings, with
one correction after further checking:

- **WebShop trajectories** (`/data/a00481223/WebShop/all_trajs/*.jsonl`,
  99 files) — initially reported as ~16GB of readable agent trajectory
  data. **That was wrong.** Only the directory listing (filenames) was
  ever actually checked; every one of those files is `-rw-------`
  (owner-only) and none are actually readable by this account. Corrected
  and dropped from consideration. (The only genuinely readable jsonl
  files in that repo are `search_engine/resources*/documents.jsonl` —
  e-commerce product-catalog listings, not trajectories — not used.)
- **`wiki-18.jsonl`** (`/data/r50058044/reskill_search/retriever/`,
  reported size 14.4GB) — also not what its name suggests: it's
  actually a **tar archive**, containing one real file,
  `data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl`
  (~13.4GB). This one IS genuinely readable and usable. Schema per line:
  `{"id": "<n>", "contents": "\"<Article Title>\"\n<~100-word passage text>"}`
  — the well-known FlashRAG-style `wiki_dpr_100w` DPR passage split.

**Decision**: pretrain on this Wikipedia passage corpus alone (not a
mix — WebShop is unusable). This is a deliberate, acknowledged
departure from the code/tool-use data scope stated in every phase spec
— see the note added to
`PROMPTS/soft_prompt_generator_spec_v2_self_embedding.md`. Treat any
resulting checkpoint as validating the training *method*, not as a
generator ready to generalize to the real coding/tool-use deployment
target.

**Pack vs. pad vs. truncate**: passages are short (~100 words) and
fairly uniform — padding each one individually to `max_seq_len` (512)
would waste most of every sequence on padding. Decision: **pack**
several passages per training example (concatenated, separated by the
tokenizer's real EOS token — matching how the frozen model itself was
almost certainly pretrained on packed documents), up to an
approximate character-budget estimate of `max_seq_len`, and let the
existing per-batch tokenizer call's `truncation=True` silently correct
any slight overshoot. This needed zero changes to the training loop
(`training_step`/`evaluate` still just receive a `List[str]` of
"prompts" — a packed string is indistinguishable from any other prompt
string to that code). Truncation as a hard *ceiling* is still in effect
via the existing tokenizer call.

**Calibrated against the real Qwen 3 32B tokenizer** (downloaded just
the tokenizer files, not the model — a few MB): on 500 real passages,
measured 4.212 real chars/token (vs. the initial 4.0 guess — close, not
the main source of error). The real source of overshoot was greedy
packing's boundary effect: each atomic passage is a large chunk
(~150 tokens) relative to the 512-token budget, so crossing the
threshold can overshoot by nearly a full passage. Tuned
`target_fill` from 0.95 → 0.85 to leave headroom for that; re-measured
with real tokenizer counts (not the character estimate) on the same
500 passages: average packed length 465 real tokens (target 512, now
erring slightly under rather than over), only 22/163 packed strings
(~13%) exceed 512 and would get truncated at all, down from the
untuned version's systematic overshoot on nearly every packed string.

**Implementation**:
- `edgealign/data/wikipedia.py` — `iter_passages(tar_path, ...)`:
  streams the jsonl member directly out of the tar via Python's
  `tarfile` (never extracts the ~13.4GB archive to disk — deliberately,
  given the disk-space incident above), with an approximate
  reservoir-buffer shuffle (the corpus is far too large to shuffle in
  memory outright).
- `edgealign/data/packing.py` — `pack_texts(texts, max_seq_len, ...)`:
  the character-budget packing described above, `target_fill=0.85`.
- `edgealign/train.py`'s `build_data_iterators` gained a `"wikipedia"`
  data mode wiring these two together into the same batch-of-strings
  shape every other mode already produces.

**Verified**: locally, a synthetic tiny tar file (same structure: tar
containing one `.jsonl` member, a handful of short fake passages)
exercises `iter_passages`+`pack_texts`+the real training loop against
the tiny random Qwen3 model used by the other smoke tests — passes
(`scripts/run_smoke_test_wikipedia_packing.py`). Separately, on the
cluster, the loader was pointed at the real `wiki-18.jsonl` tar
(`scripts/check_wikipedia_on_cluster.py`) and confirmed to stream and
pack real passages correctly, and the packing calibration above was
verified against real tokenizer output, not just the character
estimate.

## Real-weights smoke test — Qwen 3 32B, and a real bf16 bug found (2026-09-15)

**Two cached model paths investigated, one dropped**:

- `/data/a00481223/models/Qwen3.6-27B` — the user initially proposed this
  as a stand-in for the intended 32B. Investigated its config and found
  it is **not** a plain Qwen3 decoder: `architectures:
  ["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"` — a
  **multimodal vision-language model** (has `vision_config`, image/video
  token IDs) whose 64-layer text backbone is a **hybrid attention
  stack**: only every 4th layer is standard full attention (16 of 64);
  the other 48 are `linear_attention` (Mamba/GatedDeltaNet-style, per
  `mamba_ssm_dtype`/`linear_conv_kernel_dim` fields). Concretely, layers
  `L-3, L-2, L-1` (61, 62, 63) — exactly what the self-embedding
  generator reads — would all three be linear-attention layers, never
  full-attention, since layer 64 (excluded) is the nearest full-attention
  layer. Also note: the safetensors weight files here turned out to be
  world-readable (`-rw-rw-r--`) despite the user's belief that "the
  pretrained weights are not accessible now" — that assumption was
  wrong, confirmed by reading real bytes from a shard directly. **Not
  used** — dropped once the user found the actual intended model at the
  path below. If revisited later, the hybrid-attention layer-selection
  question above needs a real decision, not a silent default.
- `/data/models/huggingface/qwen3-32b` — the actual intended model.
  Config confirms `architectures: ["Qwen3ForCausalLM"]`, `model_type:
  "qwen3"`, `hidden_size=5120`, `num_hidden_layers=64`,
  `num_attention_heads=64`, `num_key_value_heads=8`, `vocab_size=151936`
  — a plain, uniform decoder-only causal LM, exactly matching what the
  self-embedding generator (and the rest of the infra) assumes. No
  hybrid attention, no multimodal wrapper. Permissions `-rwxrwxrwx`
  (openly readable) — confirmed by reading real weight bytes directly.
  **This is the model now in use.**

**Smoke test built**: `scripts/run_smoke_test_qwen3_32b_real.py` — unlike
the other smoke tests (tiny random models, run anywhere), this loads
the REAL 32B weights via `FrozenLLMHarness.from_pretrained` and must run
on the cluster with real GPUs. Uses the v2 self-embedding generator with
`hidden_size`/`num_hidden_layers` auto-derived from the loaded model's
real config. Minimal otherwise (batch size 1, short sequences, 3 steps)
— the point is catching real integration issues, not producing a useful
checkpoint.

**Result: a real bug, not in our code.** First run: `kl_loss=7.9` at
step 0 (already very high), `nan` by step 1. Diagnosed methodically
rather than just lowering the learning rate:
1. Probed real hidden-state magnitudes directly (bypassing our training
   loop) — found the frozen model's own plain forward pass
   (`teacher_pass`, no soft prefix, completely standard call) produces
   **NaN starting around layer 10** with the default attention
   implementation.
2. Forcing `attn_implementation="eager"` fixed the early NaN (layers
   1/2/10/30 all clean) but a **second, different** degeneration
   appeared near the tail: exact-zero blocks scattered through layers
   ~38-63 (confirmed via per-layer magnitude dump — not a clean single
   break at a device-sharding boundary, ruled that out explicitly by
   checking `model.hf_device_map`), cascading to **NaN at the final
   layer (64)**.
3. **Root cause confirmed** by re-running the identical forward pass in
   **float32 on CPU** (this cluster has 755GB RAM — trivial to do for
   one diagnostic forward pass, even though far too slow/memory-heavy
   for real training): every layer came back clean, finite,
   smoothly-growing values, no zeros, no NaN, sane final logits. **This
   is a genuine bf16 numerical instability specific to this checkpoint
   on this transformers version — not a bug in our injection/generator
   code, not a device-sharding artifact, not fixed by switching
   attention backends.**
4. **Hypothesis tested and DISPROVEN**: the checkpoint's own
   `config.json` states `"transformers_version": "4.51.0"`; this
   cluster's main venv has `transformers==5.17.0` installed. Built a
   second, isolated venv (`~/EdgeAlign/.venv_tf451` on the cluster —
   deliberately kept separate from the working main `.venv` so a failed
   experiment couldn't cost the already-verified baseline) with
   `transformers==4.51.0` pinned exactly (pip auto-resolved
   `accelerate==1.15.0`, compatible `tokenizers`/`datasets`). Confirmed
   first that nothing else regressed: all three tiny-model smoke tests
   (stub, self-embedding, wikipedia-packing) still pass unmodified under
   4.51.0. Then re-ran the real Qwen3-32B bf16 check under 4.51.0 with
   the default (SDPA) attention backend — **still broken, and actually
   worse**: NaN starts at layer 6 (vs. layer 10 under 5.17.0), same
   exact-zero/NaN mixed pattern through the rest of the network, same
   NaN at the final layer, same zero logits. The transformers version
   is not the cause.

**Status: root cause still open, but narrowed.** Since bf16 is broken
under both library versions (4.51.0 and 5.17.0) while fp32 is completely
clean under 5.17.0, this isn't a transformers regression — it's
something more fundamental about running this specific checkpoint's
bf16 *compute* (not storage — the checkpoint's weights are natively
bf16 already, so fp32 loading just losslessly upcasts them; the fp32
run's cleanliness shows the input values are fine, only the bf16
*arithmetic* through 64 layers is not) at this depth, on this hardware/
torch combination. fp32 itself isn't viable for real training (2x
memory, ~108GB just for weights, much slower). Untested candidates:
fp16 instead of bf16 (unclear given residual-stream magnitudes already
reaching ~20+ in fp32 — fp16's much narrower range could overflow
outright rather than underflow); a different torch version (2.14.0 is
very new; a CUDA/kernel-level bf16 regression on A100 specific to this
torch release is untested); comparing against a smaller Qwen3 checkpoint
in bf16 to see if this is checkpoint-specific or general to the
architecture at this depth. **Do not resume real training on this
checkpoint in bf16 until this is actually fixed and re-verified** — the
smoke test's job (catching exactly this kind of issue before a real
run) worked as intended.

**Update: tested Qwen 3 8B (`/data/models/huggingface/qwen3-8b`, same
`Qwen3ForCausalLM`/`qwen3` family, same tokenizer/vocab, 36 layers,
hidden_size=4096) in bf16, forward path only (no training, per
instruction) — clean.** Every layer (checked every 3rd, 0 through 36)
produced finite, smoothly-growing values, no NaN, no zero blocks; final
logits were not just finite but genuinely sensible (given `"def add(a,
b):\n    return a + b"`, predicted `"\n\n"` as the next token — exactly
right). Same transformers 5.17.0, same torch/hardware, same `sdpa`
attention backend that failed on the 32B. Ran on a single GPU (8B in
bf16 is ~16GB, fits on one A100; the 32B needed 3-way sharding).

**So this is specific to the 32B checkpoint (or its depth/sharding),
not a general Qwen3-in-bf16 problem.** Two candidate explanations,
not yet disambiguated: (a) **depth** — 64 layers vs. 36, and the 32B's
own fp32 residual-stream norm was already seen growing to ~20+ by layer
63 vs. the 8B's more modest ~6.4 peak (at layer 33) before dropping to
0.84 by layer 36 — a deeper network accumulating more bf16 rounding
error is a plausible, literature-consistent explanation; or (b)
**multi-GPU sharding** — the 32B test required `device_map="auto"`
across 3 GPUs, the 8B test used only 1, and this hasn't been cleanly
separated from the depth explanation since no single GPU here has
enough memory to hold the 32B alone to test depth in isolation from
sharding.

**GPU etiquette note**: this cluster is shared and other users run real
jobs concurrently. Before each run, checked `nvidia-smi` for free GPUs
rather than assuming, and switched to a user-specified safe subset
(`CUDA_VISIBLE_DEVICES=4,5,6` fully free; excluded 7 which had ~27GB
already used by another user's long-running llama-server) after an
earlier attempt using GPU 7 alongside free ones showed a misleading
"parameters offloaded to CPU" warning — later ruled out as the actual
cause of the bf16 issue (the same degeneration reproduced identically
on a clean 3-GPU-only run with no offloading), but avoiding partially-
occupied GPUs remains the safer default on this box regardless.

## Working target switched to Qwen 3 8B for now (2026-09-15)

**Decision**: since the 32B bf16 bug above is a real, unresolved blocker
and Qwen 3 8B's forward path checked out clean, everything (configs,
default scripts) is now pointed at
`/data/models/huggingface/qwen3-8b` until the 32B issue is actually
fixed. This is a pragmatic stopgap, not a move to "Phase 2 — Deploy"
(that phase means an 8B model, *quantized*, on laptop-class hardware —
this is the plain unquantized 8B on the same A100 cluster, still
functionally standing in for the Phase 1 prototype target). Don't
confuse the two when reading old notes above that assume 32B is "the"
prototype model.

- `configs/qwen3_8b.yaml` — new config, model path pointed at the 8B
  checkpoint, otherwise mirrors `configs/prototype_32b.yaml` (same
  generator type, same data sources). Use this one for now.
- `configs/prototype_32b.yaml` is left as-is (still targets the real
  32B) — not deleted, so it's ready to use again once the bf16 bug is
  actually fixed and re-verified.
- `scripts/run_smoke_test_qwen3_8b_real.py` — new script, same pattern
  as the 32B one but defaulting to the 8B path.
  `scripts/run_smoke_test_qwen3_32b_real.py` is kept as-is for
  re-testing 32B later.

**Full end-to-end verification, not just forward-path**: ran the actual
training-loop smoke test (`run_smoke_test_qwen3_8b_real.py`) against the
real 8B weights with the real self-embedding generator. First attempt
"passed" its assertions (finite loss, frozen params unchanged, generator
params updated) but the loss trajectory was actually `[0.635, 268271.75,
2480.09]` — technically finite, but nowhere near a sane KL divergence
(bounded by ~11.9 nats for this vocab size in a healthy regime). The
lenient finite-only assertion missed this. Diagnosed rather than just
reported it as passing:
- Checked initial calibration directly: soft-prefix norm 2.05 vs. real
  embedding mean-norm 1.37 (close, not the problem), initial KL 0.38
  (sane) — so the blowup happened *after* the first optimizer step, not
  from bad initialization.
- The smoke test scripts had hardcoded `learning_rate=0.01`, borrowed
  from the tiny-toy-model smoke tests. Re-ran with `learning_rate=0.0001`
  (matching the real training config) — loss trajectory:
  `[0.734, 1.233, 0.540, 0.586, 0.301]`, completely stable. **Root cause:
  smoke-test learning rate too aggressive for real activations — a
  smoke-test hyperparameter mismatch, not a fundamental architecture or
  precision bug.**
- Fixed both `run_smoke_test_qwen3_8b_real.py` and
  `run_smoke_test_qwen3_32b_real.py` to use `learning_rate=0.0001`, and
  added a sanity-bound assertion (`max(history) < 20`) so a finite-but-
  wildly-unstable trajectory can't silently read as "PASSED" again.
  Re-ran the fixed 8B smoke test end-to-end: clean pass, loss trajectory
  `[0.635, 0.150, 0.699, 0.513, 0.425]`.

**Bottom line: Qwen 3 8B is fully verified working** — frozen forward
path clean, full training-loop smoke test clean with sane loss, real
weights, real self-embedding generator, on the real cluster. This is
the actual current working target for prototyping.

**Important caveat for future debugging — read this before blaming the
generator.** We now have direct, hard-won evidence that the *frozen
model itself* can silently fail in ways that have nothing to do with
the generator or training code (the 32B bf16 bug: exact-zero blocks and
NaN in the frozen model's own plain forward pass, no soft prefix, no
generator involved at all — see above). **If training or evaluation on
Qwen 3 8B fails, produces garbage, or fails to converge at some point in
the future, do not assume it's necessarily the generator's architecture
or training logic that's at fault.** Check the frozen model's own
plain-forward-pass health first (the same kind of per-layer NaN/zero
probe used above) before concluding the self-embedding design doesn't
work. A generator trained against a frozen model that is itself
producing degenerate output cannot possibly learn anything meaningful,
regardless of how correct the generator's own code is.

## Cluster GPU topology, and the peer-access fault (2026-09-16, corrected 2026-09-16)

**Superseded finding, kept for the record: "GPU 4 is the culprit" was
WRONG.** An earlier note here concluded GPU 4 specifically was faulty
because every failing combination happened to include it and `[1, 3]`
(excluding it) got further. That was a real, honest finding at the
time but a false lead — later, more controlled testing directly
disproved it (see below). Do not re-derive "avoid GPU 4" from this
history; there is no confirmed bad GPU on this node.

**What actually explains the recurring fault** (`torch.AcceleratorError:
CUDA error: Invalid access of peer GPU memory over nvlink or a hardware
error` / `CUDA_ERROR_CONTAINED`, driver messages sometimes reading
"Sticky error detected" / `cudaErrorContained` from `cuStreamSynchronize`,
`cuMemFree_v2`, or `cuStreamGetCaptureInfo_v3`) — established through a
long, methodical elimination, not guessed:

1. **Not GPU-specific.** The identical fault reproduced on GPU 1 alone,
   GPU 7 alone, GPU 5 alone, and various pairs — every GPU on the node
   has shown it at some point. Decisively: a controlled test ran the
   exact same real forward+backward step **simultaneously on all 6
   currently-idle GPUs (1, 3, 4, 5, 6, 7)** — all 6 passed cleanly in
   the same run in which GPU 7 alone had just failed twice in a row
   moments earlier. If any GPU were actually defective, it would fail
   this test deterministically; none did. **There is no bad GPU to
   avoid.**
2. **Not "NVLink" despite the message.** `nvidia-smi topo -m` shows
   these are PCIe-only A100s with no NVLink hardware between any pair
   on this box — confirmed the fault occurs on a *single* GPU with zero
   peer/cross-device traffic involved at all, so "peer GPU memory over
   nvlink" is generic/inaccurate boilerplate for this CUDA error code,
   not a literal description of the cause.
3. **Not MIG, not another tenant's process, not GPU-level contention in
   the usual sense.** MIG is disabled on every GPU; at the moment of
   several failures the target GPU had zero other processes, 0% prior
   utilization, clean ECC/remapped-rows/retired-pages, no throttling,
   no MPS, no containers, nothing holding `/dev/nvidia<N>` open
   (`fuser`/`lsof` both empty). `dmesg`/`journalctl` kernel logs (where
   a real Xid hardware error would show up) are not readable without
   `sudo`, which was not used.
4. **Not the CUDA/driver/torch version.** The node's torch (2.14.0,
   the latest PyPI release at the time) paired with driver 595.58.03
   was a real suspect (bleeding-edge stack). Built an isolated venv
   (`.venv_torch_stable` on the cluster) with torch 2.9.0+cu126
   instead — the identical fault still occurred, just reported as an
   unrecognized error code (the older CUDA runtime has no string for
   whatever code the driver returns) instead of the named
   `cudaErrorContained`. Two different torch/CUDA-toolkit generations
   hitting the same underlying condition rules out a torch-version bug.
5. **Two genuine, different, *fixable* things were hiding inside "the
   fault" and got found by not stopping at the first plausible
   explanation:**
   - A real out-of-memory condition, confirmed directly (not inferred)
     by running with `torch.autograd.set_detect_anomaly(True)`, which
     forces synchronous CUDA error reporting: the exact same code path
     that normally raised the generic peer-access error instead raised
     a plain `torch.OutOfMemoryError` at 37.9/39.5 GB used, at
     `seq_len=4096, batch=1` on a single GPU. PyTorch's *asynchronous*
     CUDA error reporting means a real OOM on one CUDA stream is often
     only surfaced by a *later, unrelated* CUDA call, with whatever
     generic error code the now-corrupted context happens to report —
     this is why the crash site kept moving between runs and why it
     looked hardware-flavored. Fixed operationally: halved
     `max_seq_len` (4096→2048) and `kl_chunk_size` (512→256) in
     `configs/qwen3_8b.yaml`, since most of the seq_len-scaling memory
     (student-pass activations across 36 layers, the logits tensor and
     its gradient) is ~linear in seq_len, not quadratic (SDPA doesn't
     materialize a full seq² attention matrix here).
   - A real device-mismatch bug in our own code, unrelated to the
     cluster: `edgealign/init_utils.py`'s `compute_embedding_scale_stats`
     called `torch.randperm(vocab_size, generator=generator)` with no
     `device=`, defaulting to CPU, then indexed a CUDA tensor
     (`weight[idx]`) with it — an unusual pattern (implicit CPU→GPU
     index transfer right after a large bulk weight load) that a normal
     inference workload would never hit. **Fixed**: pass
     `device=weight.device` explicitly. Confirmed via a dedicated local
     resume test (see below) and via the cluster no longer failing at
     that exact line after the fix.
6. **Multi-GPU `device_map="auto"` sharding silently corrupts Qwen3-8B's
   numerics** — a separate, confirmed, real bug, not the crash itself
   but found while investigating it. Same real Wikipedia data, same
   code, same model: **single-GPU** gives correct logits
   (`teacher_logits.abs().max() ≈ 51`, teacher/student closely
   matching, as expected early in training); a **2-GPU** split (tried on
   two different, fully healthy pairs — `[1,3]` and `[5,6]` — got the
   *identical* corrupted value both times: `teacher_logits.abs().max() =
   0.98046875`, `student_logits` NaN or exact zero). This is
   deterministic given the shard boundary, not random, and it silently
   produced a `kl_loss=nan` training run earlier in this same session
   that looked like ordinary instability rather than a sharding bug.
   Root cause not fully isolated (candidate: some buffer — e.g. RoPE
   frequencies — not correctly replicated across the accelerate
   dispatch boundary), but the practical fix is clear and applied:
   **`available_gpus` is single-GPU only for this model on this
   cluster** (see `configs/qwen3_8b.yaml`) until this is actually
   root-caused. This also means the original multi-GPU memory-budget
   plan (spread `seq_len=16384, batch=4` across 4 GPUs) is not viable as
   designed — a single 40GB A100 is the real current budget.
7. **What's left unexplained, honestly**: after both real bugs above
   were fixed, the peer-access/sticky fault still recurs sometimes,
   apparently at random — same exact script, same GPU, sometimes
   passes cleanly, sometimes doesn't, with no reproducible trigger
   found despite extensive isolation (attention backend eager vs. sdpa,
   `device_map`, forward hooks, `requires_grad_(False)`, Liger Kernel
   import, batch size, sequence length bisection down to 128, real vs.
   synthetic data, `CUDA_LAUNCH_BLOCKING=1` for synchronous reporting).
   Best-supported remaining explanation: a genuinely rare, transient
   condition on this heavily shared (40-user, 60-day-uptime) node —
   not tied to any specific GPU (point 1), not OOM (point 5, when
   checked directly at the moment of failure memory usage was low,
   e.g. 16.4/42.4 GB), not a code bug found so far. **Once this fault
   does occur, the CUDA context becomes genuinely "sticky" for the rest
   of that process** — confirmed directly: an in-process retry's own
   recovery call, `torch.cuda.empty_cache()`, failed with the identical
   `CUDA_ERROR_CONTAINED`/"Sticky error detected" message. There is no
   safe in-process recovery once this happens; see "Checkpoint + restart
   recovery" below for the mitigation actually built.

**Practical takeaways for future sessions**:
- Don't re-conclude a specific GPU is bad from a handful of failures on
  it — the base rate of this fault is high enough, and it strikes
  GPU-independently often enough, that this is a strong false-positive
  trap. Re-run the *same* test on a couple of other idle GPUs before
  concluding hardware.
- Always check whether an "AcceleratorError" might actually be an OOM
  by re-running the failing step with `torch.autograd.set_detect_anomaly(True)`
  (or `CUDA_LAUNCH_BLOCKING=1`) before assuming it's external/hardware.
- Multi-GPU `device_map="auto"` sharding is NOT currently trustworthy
  for Qwen3-8B's numerics on this cluster — always spot-check
  `teacher_logits`/`student_logits` magnitude (not just "did it crash")
  after any change that touches GPU count.

## Full training config system (2026-09-16)

Extended `TrainConfig` (`edgealign/config.py`) with real knobs beyond the
original fixed-LR/no-accumulation setup, all backward compatible
(existing configs/smoke tests work unchanged with the new defaults):

- `grad_accum_steps` — micro-batches accumulated before each optimizer
  step. `max_steps`/`log_every`/`eval_every` all count *optimizer* steps
  (post-accumulation), not micro-batches, so a config's `max_steps`
  means the same thing regardless of `grad_accum_steps`.
- `optimizer` (registry key in `train.py`'s `OPTIMIZER_REGISTRY`) +
  `optimizer_kwargs`.
- `lr_scheduler` (any name `transformers.get_scheduler` supports:
  "constant", "cosine", "linear", ...) + `warmup_steps` +
  `lr_scheduler_kwargs`. Non-"constant" schedules need a concrete
  `max_steps` (not `None`) to compute their decay curve.
- `num_gpus` + `available_gpus` — `train.py`'s `main()` sets
  `CUDA_VISIBLE_DEVICES` from `available_gpus` as the very first thing,
  before any CUDA call (must happen before `FrozenLLMHarness.from_pretrained`
  — CUDA device visibility can't change after the driver initializes a
  context). `num_gpus` is a sanity cross-check against
  `len(available_gpus)`, not an independent control.

Verified locally (tiny CPU model): grad_accum=4 + cosine scheduler +
warmup=3 over 10 optimizer steps — LR correctly ramps 0→peak over the
warmup steps then cosine-decays to 0 by the final step; exactly 40
micro-batches consumed for 10 optimizer steps. All three pre-existing
smoke tests re-verified passing unchanged after the change.

## Scaling to the real training config -- a real debugging saga (2026-09-16)

User's target real config: `seq_len=16384, batch_size=4, grad_accum=4,
lr=1e-3, cosine schedule, warmup=5000, AdamW, 4 GPUs`. Getting there
required finding and fixing two genuine bugs, ruling out one
counterproductive "fix", and running into a real PyTorch/CUDA
diagnostic quirk. In order:

1. **Bug 1 (fixed): `output_hidden_states=True` retains every layer.**
   The self-embedding generator only reads 3 of 37 layer outputs, but
   requesting `output_hidden_states=True` forces the frozen model to
   retain ALL of them simultaneously (~20GB at seq_len=16384, batch=4) —
   this alone OOM'd on a single GPU before even reaching the generator.
   **Fix**: `FrozenLLMHarness.teacher_pass_selected_layers` (forward
   hooks on just the needed decoder layers, `frozen_model.py`) replaces
   `output_hidden_states=True` for any generator with
   `needs_frozen_hidden_states=True`. `GeneratorFrontend` gained
   `required_hidden_state_layers`; `SelfEmbeddingGenerator` sets it to
   `self._layer_indices`. `hidden_states` is now a `Dict[int, Tensor]`
   keyed by those same indices, not a plain sequence — `forward()`'s
   existing `hidden_states[i]` indexing needed no change. Verified: all
   three local smoke tests still pass; on the cluster, this got a
   4-GPU run past both the frozen model's forward pass AND the
   generator's own pooling (previously the earliest failure points).
2. **Bug 2 (fixed): the KL loss materializes the full vocab tensor.**
   `kl_distillation_loss` casts both logits to float32 and runs
   `log_softmax` over the real vocab (151,936) for the whole sequence at
   once — at `batch=2, seq=4096` alone this needed ~4.3GB for one
   intermediate tensor. Invisible in smoke tests (`vocab_size=512`
   there). Considered chunking (sequence-dim or vocab-dim), a plain
   bf16-no-upcast tweak, and Liger Kernel; user chose Liger Kernel, and
   specifically its **stable** `LigerKLDIVLoss` (not the unreleased
   `LigerFusedLinearKLDivLoss`, merged upstream literally the day before
   and not in any pip release — that one would avoid materializing
   logits at all via a fused linear+KL kernel, but was judged too
   bleeding-edge). **Fix**: `edgealign/losses.py` now tries
   `from liger_kernel.transformers import LigerKLDIVLoss` at import
   time; used only when the tensors are actually on CUDA (`is_cuda`
   check at call time, not just import time) — CPU/no-liger always
   falls back to the original `F.kl_div` path unchanged, so local smoke
   tests need no dependency and are untouched by this. One real API
   quirk found by testing, not assumed: Liger's kernel expects a
   **flattened `(batch*seq, vocab)`** input, not `(batch, seq, vocab)` —
   reshape before/after. Directly verified both forward (max abs diff
   ~1.2e-7) and backward (grad max abs diff ~1.9e-9) match `F.kl_div`
   exactly before trusting it. Added `liger-kernel>=0.8.2` to
   `requirements.txt`, marked cluster-only (needs Triton/CUDA).
3. **False lead, reverted: `max_memory_gib` safety margin.** After two
   consecutive load-time failures (model partially offloaded to `meta`
   device) that *looked* like external contention (the OOM error's
   "other processes on this GPU" listing named a real PID with ~36GB in
   use), added a per-GPU memory cap (`ModelConfig.max_memory_gib`,
   `FrozenLLMHarness.from_pretrained`) meant to leave headroom against
   exactly that. **It made things worse** — reverting it (`max_memory_gib:
   None`) let the same config load successfully, confirmed directly by
   toggling the one setting back and forth. Root cause of the original
   "contention": a genuine **PyTorch/NVML diagnostic quirk** — when
   `CUDA_VISIBLE_DEVICES` restricts visible devices, the OOM error's
   supplementary "other processes on this device" listing is built from
   NVML physical indices, not the CUDA-runtime-remapped local indices
   used everywhere else in the same error message. So "GPU 1" in the
   main error text correctly meant our own local index 1 (physical GPU
   4, genuinely tight on memory), while the "Process 3781298" mentioned
   alongside it was actually on unrelated *physical* GPU 1 — a
   coincidental collision between two different indexing systems both
   printing "1", not real contention. Verified directly: cross-referenced
   the PID against `nvidia-smi --query-compute-apps` + `--query-gpu=uuid`
   to confirm its true physical GPU didn't match ours. The
   `max_memory_gib` knob itself is kept in the codebase (harmless, off
   by default) since it's a real, legitimate lever in principle — just
   not the fix for this particular symptom. **Lesson for future
   debugging: don't trust the "other processes" list in a CUDA OOM
   error when CUDA_VISIBLE_DEVICES is set — cross-reference PIDs against
   `nvidia-smi --query-gpu=uuid` before concluding it's external
   contention.**
4. **Genuine remaining constraint: 2 GPUs is a tight fit, 4 has real
   margin.** With both real bugs fixed and the false lead reverted, 2
   free GPUs (`batch=2, seq_len=4096`) succeeded *intermittently* —
   same config, no code change, sometimes loads fine, sometimes hits
   the same meta-device failure moments later purely from this heavily-
   shared cluster's genuine, constantly-fluctuating other-user memory
   usage (confirmed via repeated `nvidia-smi` snapshots throughout this
   session showing different GPUs occupied every few minutes). Not
   fragile in our own code — fragile because 2×40GB is close to the
   real requirement at that scale, so any external fluctuation tips it
   over. 4 GPUs gives real headroom instead of living at the edge.

**This section's original ending (re-verifying `batch=4, seq_len=4096,
4 GPUs`) is superseded** — see "Cluster GPU topology, and the
peer-access fault" above (corrected) for what that re-verification
actually turned up: the multi-GPU sharding numerics bug, and the real
OOM at that seq_len. Current real config is single-GPU,
`seq_len=2048`, `kl_chunk_size=256` — see `configs/qwen3_8b.yaml`.

## Checkpoint + restart recovery for the sticky CUDA fault (2026-09-16)

Since the peer-access fault above is confirmed *sticky* (no safe
in-process recovery — see point 7 in the section above) but genuinely
rare/transient rather than deterministic, the practical mitigation is
process-level: checkpoint periodically, and restart a fresh process
when it happens.

- `TrainConfig.checkpoint_every` (default `0` = disabled): every this
  many *optimizer* steps, `run_training` (`edgealign/train.py`) saves
  generator + optimizer + scheduler state dicts, `global_step`, and the
  full loss `history` to `<output_dir>/checkpoint.pt`.
- `run_training` auto-resumes from that file if it exists at start —
  no separate resume flag needed. The data iterator itself is
  deliberately NOT checkpointed (it's a streaming corpus); a resumed
  run continues consuming fresh batches rather than replaying the
  exact pre-restart sequence. Acceptable for this self-distillation
  warm-start, not treated as a correctness requirement.
- `main()` catches `torch.AcceleratorError` / `torch.OutOfMemoryError`
  around the whole load+train call, prints the full traceback (so the
  real failure site is never lost, unlike the first version of this
  handler), and exits with a distinct sentinel code
  (`edgealign.train.STICKY_CUDA_FAULT_EXIT_CODE = 42`) — deliberately
  making no further CUDA calls itself (even `torch.cuda.empty_cache()`
  is unsafe post-fault, per point 7 above).
- `scripts/train_with_restart.sh <config.yaml> [max_restarts]` loops
  `python -m edgealign.train --config <config.yaml>`, restarting (after
  a short sleep) only on exit code 42; any other exit code (0 = done,
  anything else = a real bug/config error) stops immediately instead of
  masking it as a restart-worthy fault.
- Verified locally: a dedicated resume test (tiny CPU model, 3 steps +
  checkpoint, then a fresh `run_training` call with `max_steps=5`
  against the same `output_dir`) confirms it resumes from
  `global_step=3` and runs exactly 2 more steps, returning the full
  5-entry trajectory. On the cluster, `train_with_restart.sh` was
  observed correctly restarting 10/10 times on the sentinel code during
  a run of sporadic faults, and correctly not treating other errors as
  restart-worthy.

**Known limitation, not yet hit in practice but worth knowing**: if the
sticky fault recurs on *every* attempt in a row (observed once, when
the node had unusually heavy concurrent load from another user's
tensor-parallel job), restarting doesn't help — it just burns through
`max_restarts` and gives up. The restart mechanism helps with rare,
sparse faults; it is not a fix for sustained unavailability.

**Status as of this note**: single-GPU (any idle GPU; GPU 1 has also
been independently reported to be currently out of order by the
cluster's other users — avoid it specifically, separately from the
now-disproven "GPU 4" finding above), `seq_len=2048`, `batch_size=1`,
`kl_chunk_size=256`, `checkpoint_every` set, restart script in place.
Real training run to convergence has not yet been started — this was
all verification/infrastructure work. `seq_len` is still far below the
originally-requested 16384; revisiting that is blocked on the
multi-GPU sharding bug (point 6 above) being root-caused, since a
single GPU's ~40GB is the real ceiling until then.

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
