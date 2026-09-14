# Soft Prompt Generator Pretraining Spec (Phase 1, large model prototyping)

## Objective

Design and pretrain a soft prompt generator pipeline that produces a small set
of trainable prefix vectors for a frozen LLM, such that the frozen model's
output distribution with the soft prefix matches its output distribution
without any soft prefix, on the same prompt. This is a self-distillation /
autoencoding warm-start step, to be run before any reinforcement learning
fine-tuning of the generator.

This is the PROTOTYPING version of the spec. The goal here is to validate the whole method, does the self-distillation warm start work, does the later reward loop actually improve behavior, using a larger, more capable frozen model on server hardware where iteration is fast. This is not the deployment target.

## Frozen base model

- Model: Qwen 3, 32B parameter dense variant
- Role: frozen throughout, used twice per training example, once as teacher
  (plain prompt, no soft prefix) and once as student (soft prefix + prompt)
- Weights are never updated at any stage of this pipeline
- Target hardware: server with A100 GPUs, model and generator both run on GPU, no on-device memory constraints at this stage
- Priority at this stage is fast iteration and clean validation of the method, not compute efficiency

## Generator pipeline architecture

1. Encoder
   - A frozen BERT-family encoder (e.g. bert-base or a distilled variant)
   - Input: the raw prompt and context text
   - Output: a fixed-size embedding representing the prompt and context

2. Projection / generation network
   - A 3 layer MLP
   - Each hidden layer: 128 units
   - Activation: GELU (or ReLU if simplicity is preferred for v1)
   - Output layer: produces N soft prompt vectors, each matching the
     dimensionality of the frozen Qwen 3 (32B) token embedding space
   - N (number of soft prompt tokens) is a configurable hyperparameter,
     start with N = 16 to N = 32 for initial experiments

3. Injection point
   - The generated N soft prompt vectors are prepended as a prefix to the
     real prompt's token embeddings, before being passed into the frozen
     Qwen 3 (32B)
   - The original prompt tokens and their embeddings are left unmodified

## Pretraining objective (self-distillation warm start)

For each training example (a prompt or prompt plus context):

1. Teacher pass: run the plain prompt through frozen Qwen 3 (32B), no soft
   prefix. Record the output token distribution (logits) at each generation
   step.

2. Student pass: run the encoder plus MLP to generate the soft prefix vectors
   for this same prompt. Prepend them to the prompt's embeddings and run
   through the same frozen Qwen 3 (32B). Record the output token
   distribution.

3. Loss: KL divergence between the teacher distribution and the student
   distribution, averaged over generation steps. Only the encoder-to-MLP
   generator parameters receive gradient updates. The frozen Qwen 3 (32B)
   and the frozen BERT encoder are not updated.

4. Optional auxiliary loss: cosine distance penalty between the mean of the
   generated soft prompt vectors and the mean embedding norm of real
   Qwen 3 (32B) token embeddings, to keep generated vectors in a plausible
   numerical range early in training and avoid destabilizing the frozen
   model.

## Initialization details

- Initialize the MLP's final output layer so its initial output scale
  (mean and standard deviation) roughly matches the empirical scale of
  Qwen 3 (32B)'s real token embedding vectors, computed by sampling a set
  of real vocabulary embeddings before training starts.
- This avoids early-training outputs that are wildly out of distribution
  relative to what the frozen model expects as input.

## Training data for pretraining stage

- No task-specific reward signal is used at this stage. This stage only
  teaches the generator to produce soft prompts that preserve the frozen
  model's original behavior, so any reasonably diverse, high quality text
  works, exact dataset choice is not critical, coverage and diversity matter
  more than a specific source

- Scope note: the target agent is English-only and is meant to code and
  perform tool calls, so pretraining data should be weighted toward code and
  tool-use instruction data rather than general multilingual web text

- Selected datasets:
  - Stack-Edu (HuggingFaceTB/stack-edu): a 125B token, English, educational
    quality code dataset, filtered from the Stack v2, the same curated
    corpus used to train the StarCoder 2 models. Use this as the primary
    source of code coverage.
  - OpenCodeInstruct: a large instruction tuning dataset pairing natural
    language instructions with Python code, derived from the Stack v2 via
    OSS-Instruct, with LLM-generated unit tests for quality filtering. Use
    this to cover paired instruction-plus-code prompt shapes, closer to how
    a coding agent will actually be prompted.
  - xLAM function calling dataset: purpose-built for fine-tuning models on
    function calling and tool use, with an existing documented Hugging Face
    workflow for fine-tuning on it. Use this as the primary source of
    tool-calling prompt shapes.

- Practical recommendation: mix Stack-Edu (broad English code coverage),
  OpenCodeInstruct (paired instruction-and-code prompt shapes), and xLAM
  (tool-calling prompt shapes) for the pretraining corpus, so the generator
  sees prompt distributions close to the eventual coding-and-tool-use agent,
  rather than generic web text. Exact mixing ratios are a tunable
  hyperparameter, not fixed by this spec.
- This stage only teaches the generator to produce soft prompts that
  preserve the frozen model's original behavior

## Success criterion for pretraining stage

- Student (soft prefix) output distributions closely match teacher
  (plain prompt) output distributions across a held-out validation set,
  measured by average KL divergence
- This is the warm start checkpoint. Reinforcement learning fine-tuning
  (using task success or failure as reward, updating only the generator)
  begins from this checkpoint, not from random initialization

## Purpose of this phase

- Confirm the self-distillation pretraining objective actually converges (student KL divergence to teacher drops meaningfully)
- Confirm the downstream reward loop (specified separately) improves task success rate over the pretrained warm start
- Treat results here as validation of the approach, not as a guarantee the smaller edge model will behave identically. The compressor will likely need retuning against the smaller model's different embedding space and behavior once ported down.
- Once validated here, proceed to the deployment spec (soft_prompt_generator_spec_deploy.md) to port the method to the small, edge-sized frozen model

## Next stage (not covered in this spec)

- Reinforcement learning loop: frozen Qwen 3 (32B) plus generator perform
  real agent tool-use tasks, success or failure of tool calls is scored
  automatically, and this reward signal further fine-tunes the generator
  network only, starting from the pretrained warm-start checkpoint above
