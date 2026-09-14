# Soft Prompt Generator Pretraining Spec

## Objective

Design and pretrain a soft prompt generator pipeline that produces a small set
of trainable prefix vectors for a frozen LLM, such that the frozen model's
output distribution with the soft prefix matches its output distribution
without any soft prefix, on the same prompt. This is a self-distillation /
autoencoding warm-start step, to be run before any reinforcement learning
fine-tuning of the generator.

## Frozen base model

- Model: Qwen 3, 8B parameter dense variant
- Role: frozen throughout, used twice per training example, once as teacher
  (plain prompt, no soft prefix) and once as student (soft prefix + prompt)
- Weights are never updated at any stage of this pipeline

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
     dimensionality of the frozen Qwen 3 (8B) model's token embedding space
   - N (number of soft prompt tokens) is a configurable hyperparameter,
     start with N = 16 to N = 32 for initial experiments

3. Injection point
   - The generated N soft prompt vectors are prepended as a prefix to the
     real prompt's token embeddings, before being passed into the frozen
     Qwen 3 model
   - The original prompt tokens and their embeddings are left unmodified

## Pretraining objective (self-distillation warm start)

For each training example (a prompt or prompt plus context):

1. Teacher pass: run the plain prompt through frozen Qwen 3, no soft prefix.
   Record the output token distribution (logits) at each generation step.

2. Student pass: run the encoder plus MLP to generate the soft prefix vectors
   for this same prompt. Prepend them to the prompt's embeddings and run
   through the same frozen Qwen 3. Record the output token distribution.

3. Loss: KL divergence between the teacher distribution and the student
   distribution, averaged over generation steps. Only the encoder-to-MLP
   generator parameters receive gradient updates. The frozen Qwen 3 model
   and the frozen BERT encoder are not updated.

4. Optional auxiliary loss: cosine distance penalty between the mean of the
   generated soft prompt vectors and the mean embedding norm of real Qwen 3
   token embeddings, to keep generated vectors in a plausible numerical
   range early in training and avoid destabilizing the frozen model.

## Initialization details

- Initialize the MLP's final output layer so its initial output scale
  (mean and standard deviation) roughly matches the empirical scale of
  Qwen 3's real token embedding vectors, computed by sampling a set of
  real vocabulary embeddings before training starts.
- This avoids early-training outputs that are wildly out of distribution
  relative to what the frozen model expects as input.

## Training data for pretraining stage

- Any general text corpus of prompts and, where relevant, longer context
  passages (e.g. paired question and supporting context examples)
- No task-specific reward signal is used at this stage
- This stage only teaches the generator to produce soft prompts that
  preserve the frozen model's original behavior

## Success criterion for pretraining stage

- Student (soft prefix) output distributions closely match teacher
  (plain prompt) output distributions across a held-out validation set,
  measured by average KL divergence
- This is the warm start checkpoint. Reinforcement learning fine-tuning
  (using task success or failure as reward, updating only the generator)
  begins from this checkpoint, not from random initialization

## Next stage (not covered in this spec)

- Reinforcement learning loop: frozen Qwen 3 plus generator perform real
  agent tool-use tasks, success or failure of tool calls is scored
  automatically, and this reward signal further fine-tunes the generator
  network only, starting from the pretrained warm-start checkpoint above
