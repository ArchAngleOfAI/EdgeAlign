# Soft Prompt Generator Spec, Version 2 (Self-Embedding Generator)

## Objective

A generator design that avoids using any separate encoder model. Instead,
it reuses the frozen LLM's own internal hidden states as the input to the
generator, avoiding the redundancy of building a second, separate
representation of the same prompt.

## Core idea

The frozen LLM already builds a rich, contextual representation of the
prompt and context as part of normal processing. Rather than paying for a
separate encoder model to independently re-derive an understanding of the
same input, this design taps directly into the frozen model's own internal
hidden states and feeds them into the generator network.

## Frozen base model

- Qwen 3, 32B for prototyping, Qwen 3, 8B for deployment
- Weights are never updated at any stage of this pipeline
- The frozen model is used both as the source of hidden state embeddings
  and, after prefix injection, as the model being conditioned

## Hidden state extraction

- A first forward pass runs the plain prompt and context through the frozen
  LLM, with no soft prefix attached, purely to extract internal
  representations
- Hidden states are captured from the last three transformer layers before
  the final layer (i.e. if the model has L layers total, capture layers
  L-3, L-2, and L-1, excluding the final layer L)
- Rationale: later layers hold richer, more integrated, task-relevant
  meaning than early layers, but the very final layer can be overly
  specialized toward immediate next-token prediction. The last three layers
  before the final layer are used as a compromise, capturing high-level
  meaning while avoiding over-specialization from the last layer alone
- These three layers' hidden states are combined via concatenation (not
  averaging) into a single representation per token position, preserving
  each layer's distinct signal rather than blending them together, then
  pooled (e.g. mean pooling across token positions) into a single
  fixed-size vector representing the whole prompt and context
- Note: concatenation means the pooled input vector to the generator MLP
  is three times the width of a single layer's hidden state, increasing
  the MLP's input dimensionality accordingly

## Generator network

- Input: the pooled, concatenated hidden state vector from the three
  extracted layers
- Network: a 3 layer MLP, sized to take advantage of having no separate
  encoder to train:
  - Hidden layer 1: 256 units
  - Hidden layer 2: 256 units
  - Hidden layer 3: 128 units
  - Activation: GELU
- Output: N soft prompt vectors, each matching the dimensionality of the
  frozen model's token embedding space, start with N = 16 to N = 32
- Only this MLP is trained. The frozen LLM is never updated, and there is
  no separate encoder to train

## Injection point (second forward pass)

- The generated N soft prompt vectors are prepended as a prefix to the
  original prompt's token embeddings
- The frozen LLM is run a second time, now with the soft prefix attached,
  to produce the actual output used for pretraining loss or, later,
  real task execution

- Note: this design requires two forward passes through the frozen LLM per
  example, one to extract hidden states, one with the soft prefix attached.
  This is a real compute cost consideration, since both passes run through
  the full frozen model rather than a smaller separate encoder. This
  tradeoff should be weighed against this design's expected representation
  quality advantage.

## Pretraining objective (self-distillation warm start)

1. Teacher pass: run the plain prompt through the frozen LLM, no soft
   prefix. Record the output token distribution.
2. Student pass: extract hidden states (as above), generate the soft
   prefix via the MLP, prepend it, run through the frozen LLM again.
   Record the output token distribution.
3. Loss: KL divergence between teacher and student output distributions.
   Only the MLP's parameters receive gradient updates.
4. Optional auxiliary loss: an embedding-scale matching penalty, keeping
   generated vectors in a plausible numerical range relative to the frozen
   model's real token embeddings.

## Initialization details

- Initialize the MLP's output layer so its initial output scale roughly
  matches the empirical scale of the frozen model's real token embeddings,
  computed by sampling real vocabulary embeddings before training starts

## Training data for pretraining stage

- Stack-Edu for English code coverage, OpenCodeInstruct for paired
  instruction-and-code prompts, and xLAM for tool-calling prompt shapes,
  mixed in a tunable ratio

## Open questions specific to this version

- How exactly to pool across token positions, mean pooling is the simplest
  default, but attention-based pooling (a small learned weighting over
  token positions) may preserve more relevant information, at the cost of
  additional trainable parameters
- Whether the added compute cost of a first, hidden-state-extraction pass
  through the full frozen LLM is justified in practice by downstream task
  performance. This should be evaluated empirically during the prototyping
  phase.
- First-layer bottleneck risk: the concatenated input to the generator MLP
  is large (e.g. approximately 15,360 dimensions for Qwen 3 32B, three
  layers at 5,120 each; approximately 12,288 dimensions for Qwen 3 8B,
  three layers at 4,096 each). The MLP's first hidden layer compresses this
  down to 256 units in a single step, a substantial reduction. It is an
  open question whether 256 units is wide enough to avoid losing important
  information at this bottleneck, or whether a wider first layer, or an
  intermediate layer between the raw input and the first 256-unit layer, is
  needed. If pretraining loss (KL divergence to teacher) fails to converge
  well, this first-layer bottleneck should be one of the first suspects
  investigated, before concluding the overall approach does not work.

## Next stage (not covered in this spec)

- Reinforcement learning loop: frozen LLM plus generator perform real
  agent tool-use tasks, success or failure of tool calls is scored
  automatically, and this reward signal further fine-tunes the generator
  network only, starting from the pretrained warm-start checkpoint above
