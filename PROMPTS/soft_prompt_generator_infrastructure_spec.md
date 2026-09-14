# Soft Prompt Generator — Common Infrastructure Spec (Prototype Phase)

## Purpose of this document

This is an instructional spec for an implementing agent. It describes the
infrastructure that is **shared by every generator variant** (v1 baseline
encoder, v2 self-embedding, v3 linear attention encoder — see the other
files in this directory), extracted so it only needs to be built once. The
only thing that differs between variants is a single swappable component:
the "generator front-end" that turns a prompt into N soft prompt vectors.
Build the infrastructure below so that front-end is a pluggable module,
not something baked into the training loop.

Scope: **prototype phase only**. Target the 32B frozen model, on server
GPU hardware (A100s or equivalent), with no on-device memory constraints.
Do not build for the deployment phase (8B, quantized, laptop-class
hardware) yet — that is a separate, later port, not covered by this spec.
Priority here is fast iteration and a clean validation of the method, not
compute or memory efficiency.

## 1. Frozen base model harness

- Load Qwen 3, 32B dense variant, in frozen (`no_grad` / parameters not
  trainable) mode. Its weights must never be updated at any point in this
  pipeline.
- Provide a single wrapper/interface around it that supports two forward
  passes per training example:
  - **Teacher pass**: tokenize the plain prompt (and context, if present),
    embed it normally, run through frozen Qwen 3, record the output token
    distribution (logits) at each generation step. No soft prefix involved.
  - **Student pass**: take the N soft prompt vectors produced by the
    generator front-end (see section 3), prepend them to the real prompt's
    token embeddings, run the combined sequence through the same frozen
    Qwen 3, record the output token distribution at each generation step.
- The original prompt tokens and their embeddings must never be modified
  by this process, in either pass.

## 2. Injection mechanism

- Implement prepending as a pure embedding-space operation: N generated
  soft vectors, each with dimensionality equal to Qwen 3's token embedding
  dimension, are concatenated in front of the prompt's token embedding
  sequence before that sequence is passed into Qwen 3's forward pass.
- N (number of soft prompt tokens) must be a configurable hyperparameter.
  Default the initial experiments to somewhere in the N = 16 to N = 32
  range.
- This mechanism is identical no matter which generator front-end produced
  the vectors — implement it once, independent of the front-end module.

## 3. Generator front-end interface (pluggable, not part of this spec)

- Define a clear module boundary: something that takes a prompt (plus
  optional context) as input and returns exactly N vectors of Qwen 3's
  embedding dimensionality as output.
- The internals of this module are defined by whichever variant spec is
  selected for implementation (v1, v2, or v3, in the other files in this
  directory) and are **not** specified here. This infrastructure must not
  assume a specific internal architecture (e.g. must not hardcode "there is
  a separate BERT encoder" — v2 has none, and taps Qwen 3's own hidden
  states instead, which requires an extra forward pass through Qwen 3
  itself before the student pass).
- Build the training loop, injection mechanism, and evaluation code against
  this interface only, so swapping in a different variant later requires
  no changes outside this one module.

## 4. Training objective (self-distillation warm start)

For each training example:

1. Run the teacher pass (section 1) to get the teacher output distribution.
2. Run the generator front-end (section 3) to get the N soft vectors for
   this same prompt, inject them (section 2), and run the student pass
   (section 1) to get the student output distribution.
3. Compute the loss as KL divergence between the teacher distribution and
   the student distribution, averaged over generation steps.
4. Backpropagate only into the generator front-end's trainable parameters.
   Qwen 3 must receive no gradient updates. If the selected front-end
   variant itself contains a frozen sub-component (e.g. a frozen BERT or
   Mamba encoder in v1/v3), that sub-component must also receive no
   gradient updates — only its downstream trainable MLP should.
5. Optionally add an auxiliary loss: a cosine-distance (or norm-matching)
   penalty between the mean of the generated soft vectors and the mean
   embedding norm of real Qwen 3 token embeddings, to discourage the
   generator from drifting to numerically implausible output ranges.

## 5. Initialization

- Before training starts, sample a representative set of real Qwen 3
  vocabulary token embeddings and compute their empirical mean and
  standard deviation.
- Initialize the generator front-end's final output layer (whatever that
  layer is, depending on the selected variant) so its initial output scale
  (mean and std) roughly matches that empirical measurement.
- Purpose: avoid early-training soft prefixes that are wildly out of
  distribution relative to what the frozen model expects as input, which
  could otherwise destabilize the frozen model's behavior before the
  generator has learned anything useful.

## 6. Training data pipeline

- No task-specific reward signal at this stage — this is pure
  self-distillation, teaching the generator to preserve the frozen model's
  original behavior. Any reasonably diverse, high-quality text is
  sufficient in principle, but the target agent is English-only and
  focused on coding and tool use, so weight the corpus accordingly rather
  than using generic multilingual web text.
- Build a data pipeline that mixes, in a configurable (not hardcoded)
  ratio, the three sources below. Write a download/prepare script per
  source (they have different access patterns — see each subsection), and
  a separate mixing step that samples from the three prepared sources at
  the configured ratio into the actual training stream.

### 6.1 Stack-Edu (English code coverage)

- HF dataset id: `HuggingFaceTB/stack-edu`. Not gated, but split into one
  config per language (`c`, `cpp`, `csharp`, `go`, `java`, `javascript`,
  `markdown`, `php`, `python`, `ruby`, `rust`, `shell`, `sql`, `swift`,
  `typescript`) — load the language configs relevant to the target agent
  (Python at minimum, plus whichever others the coding agent needs to
  support) and concatenate.

  ```python
  from datasets import load_dataset, concatenate_datasets

  languages = ["python", "javascript", "typescript", "shell"]  # tune as needed
  ds = concatenate_datasets([
      load_dataset("HuggingFaceTB/stack-edu", lang, split="train")
      for lang in languages
  ])
  ```

- **Important**: this dataset only ships metadata and SWHIDs/`blob_id`
  (a SHA1) per row — not the actual file content. The file content itself
  must be resolved separately from the Software Heritage public S3 mirror
  (bucket `softwareheritage`, object key `content/{blob_id}`,
  gzip-compressed, keyed by SHA1). This requires:
  - `pip install smart_open[s3] boto3`
  - AWS credentials configured (`AWS_ACCESS_KEY_ID` /
    `AWS_SECRET_ACCESS_KEY` env vars — a free-tier AWS account is
    sufficient, requests just need to be signed; the bucket itself is
    public-read).
  - Resolution code:

  ```python
  import os, boto3
  from smart_open import open as smart_open

  session = boto3.Session(
      aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
      aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
  )
  s3 = session.client("s3")

  def download_contents(row):
      s3_url = f"s3://softwareheritage/content/{row['blob_id']}"
      with smart_open(s3_url, "rb", compression=".gz",
                       transport_params={"client": s3}) as fin:
          row["content"] = fin.read().decode(row["src_encoding"])
      return row

  ds = ds.map(download_contents)
  ```

  - This is a real, non-trivial download step (millions of individual S3
    fetches) — implement it as an offline preparation script that caches
    resolved content locally (e.g. as sharded Parquet/JSONL), not as a
    live map done at training time.
  - Optionally filter by the dataset's own `int_score` (3-5) to keep only
    the higher educational-quality tier before resolving content, to cut
    down how many blobs need to be fetched.

### 6.2 OpenCodeInstruct (paired instruction + code)

- HF dataset id: `nvidia/OpenCodeInstruct`. Not gated, ~5M rows, content is
  inline (no separate resolution step needed).

  ```python
  from datasets import load_dataset

  ds = load_dataset("nvidia/OpenCodeInstruct", split="train")
  ```

- Fields: `id`, `input` (the instruction/prompt), `output` (the generated
  solution), `domain` (`generic` or `algorithmic`), `generation_algorithm`,
  `llm_judgement`, `unit_tests`, `tests_execution_status`,
  `average_test_score` (0.0-1.0).
- Recommended quality filter before use: keep rows where
  `tests_execution_status == "pass"` and/or `average_test_score` above a
  configurable threshold, so the pretraining corpus favors instruction+code
  pairs that are actually verified correct.

### 6.3 xLAM (tool-calling prompt shapes)

- HF dataset id: `Salesforce/xlam-function-calling-60k`. Not gated,
  ~60k rows, content is inline.

  ```python
  from datasets import load_dataset

  ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
  ```

- Fields: `query` (user request), `tools` (JSON-formatted function/tool
  definitions available to the model), `answers` (expected function
  call(s), JSON). Parse `tools` and `answers` as JSON per row (they are
  stored as JSON strings); fall back to the raw string if parsing fails.
- This dataset only has ~60k rows, much smaller than the other two — when
  mixing, either upsample it or accept it as a smaller-weight component;
  do not let its small size silently make tool-calling prompt shapes
  underrepresented relative to the mixing ratio actually intended.

### 6.4 Mixing

- Exact mixing ratio across the three prepared sources is a tunable
  hyperparameter, not fixed by this spec — expose it as a config value
  (e.g. target fractions per source, or per-source epoch/repeat counts to
  correct for the size imbalance between xLAM and the other two).

## 7. Evaluation / success criterion

- On a held-out validation split, compute the same KL divergence (student
  vs. teacher output distributions) used in training, and track it over
  the course of training.
- Convergence (KL divergence dropping meaningfully and staying low across
  the held-out set) marks a valid warm-start checkpoint. This checkpoint —
  not a randomly initialized generator — is what the (separately specced,
  not-yet-written) RL fine-tuning stage should start from.

## Explicitly out of scope for this spec

- The internal architecture of the generator front-end itself (that's the
  v1/v2/v3 variant specs' job).
- The deployment-phase port (8B frozen model, quantization, laptop-class
  hardware/memory constraints) — a separate, later effort once the method
  is validated here.
- The RL fine-tuning loop that follows the warm-start checkpoint — every
  variant spec references this as a "next stage, not covered."
