# Soft Prompt Generator Spec, Version 3 (Linear Attention Encoder)

## Objective

A generator design that replaces a standard full-attention encoder with a
linear attention or state space model encoder (e.g. Mamba), specifically to
remove the short, fixed input length ceiling that standard encoder models
like BERT-family sentence transformers impose.

## Core idea

Encoders built on standard full attention (e.g. sentence transformer models
based on BERT or MPNet) are trained with a fixed, fairly short maximum input
length, often a few hundred tokens, because full attention's compute cost
grows sharply with sequence length. This creates a hard ceiling on how much
context the encoder can read at once, longer inputs must be chunked or
truncated before reaching the encoder.

Linear attention and state space model architectures (e.g. Mamba, RWKV,
DeltaNet) avoid this ceiling structurally. Instead of every token attending
to every other token, they maintain a fixed-size running state that updates
as the sequence is read, so compute cost per token stays constant regardless
of how long the input is. This allows the encoder to read much longer
context in one pass, without a hard architectural cutoff.

Known tradeoff: this fixed-size running state is a compression. Very
specific, exact details from far earlier in a long input can be diluted or
lost as the state keeps updating, sometimes called the lossy memory problem.
This is a real limitation to weigh against the benefit of removing the
length ceiling.

## Frozen base model

- Qwen 3, 32B for prototyping, Qwen 3, 8B for deployment
- Weights are never updated at any stage of this pipeline

## Encoder

- A pretrained linear attention or state space model, used as a frozen
  encoder, replacing the sentence transformer / BERT-family encoder used in
  other candidate designs
- Candidate model family: Mamba. Open, pretrained checkpoints are publicly
  available at multiple sizes, including 130M, 370M, 790M, 1.4B, and 2.8B
  parameters, trained on general text corpora (e.g. the Pile, SlimPajama)
- Recommended starting point: the 130M or 370M parameter checkpoint, to
  keep the encoder lightweight and consistent with this project's general
  preference for small, efficient components. Larger checkpoints remain
  available if more representational capacity is needed

- Note: rather than only choosing among the exact published checkpoint
  sizes, it is also worth considering taking one of these pretrained
  checkpoints and using a truncated version of it, cutting off some of its
  later layers rather than using the full network as published. This
  produces a smaller, cheaper encoder than any officially released size,
  at the cost of using a network that is no longer exactly the one it was
  pretrained as. In practice this partial network can still be useful,
  since its earlier layers were trained as part of the full model and
  retain meaningful learned structure, even though the checkpoint was not
  originally trained to be used with those later layers removed. This
  option should be evaluated empirically against the standard published
  checkpoint sizes during prototyping, comparing pretraining convergence
  and encoder output quality
- The encoder is frozen. Its role is purely to read the prompt and context
  and produce a representation, it is not trained further in this pipeline
- Output: the encoder's final running state, or its hidden representation
  at the last processed token, is used as the fixed-size representation of
  the full input, regardless of input length

## Generator network

- Input: the encoder's output representation
- Network: a 3 layer MLP (architecture and exact sizing to be finalized
  once the specific Mamba checkpoint size, and therefore its output
  dimension, is chosen; follow the same design principle used elsewhere in
  this project, size the MLP's first layer with enough width to avoid an
  aggressive bottleneck when compressing the encoder's output dimension
  down toward the soft prompt vector dimension)
- Output: N soft prompt vectors, each matching the dimensionality of the
  frozen model's token embedding space, start with N = 16 to N = 32
- Only this MLP is trained. Both the encoder and the frozen main LLM are
  never updated

## Injection point

- The generated N soft prompt vectors are prepended as a prefix to the
  original prompt's token embeddings, before being passed into the frozen
  main LLM
- The original prompt tokens and their embeddings are left unmodified

## Pretraining objective (self-distillation warm start)

1. Teacher pass: run the plain prompt through the frozen main LLM, no soft
   prefix. Record the output token distribution.
2. Student pass: run the prompt and context through the frozen linear
   attention encoder, generate the soft prefix via the MLP, prepend it, run
   through the frozen main LLM. Record the output token distribution.
3. Loss: KL divergence between teacher and student output distributions.
   Only the MLP's parameters receive gradient updates.
4. Optional auxiliary loss: an embedding-scale matching penalty, keeping
   generated vectors in a plausible numerical range relative to the frozen
   main LLM's real token embeddings.

## Initialization details

- Initialize the MLP's output layer so its initial output scale roughly
  matches the empirical scale of the frozen main LLM's real token
  embeddings, computed by sampling real vocabulary embeddings before
  training starts

## Training data for pretraining stage

- Stack-Edu for English code coverage, OpenCodeInstruct for paired
  instruction-and-code prompts, and xLAM for tool-calling prompt shapes,
  mixed in a tunable ratio
- Since removing the input length ceiling is the specific motivation for
  this design, the pretraining and evaluation data for this version should
  deliberately include longer context examples, beyond what a standard
  short-context sentence transformer could handle, to actually exercise
  and validate the intended benefit

## Open questions specific to this version

- Which Mamba checkpoint size gives the best tradeoff between encoder
  quality and compute cost for this project's constraints, to be
  determined empirically during prototyping
- How much the lossy memory tradeoff actually affects downstream task
  quality in practice, particularly for tasks that depend on precise,
  specific details from early in a long context, versus general thematic
  understanding
- Whether a hybrid approach, combining a linear attention encoder for long
  context with some mechanism for preserving precise local detail, is worth
  exploring, if pure linear attention proves too lossy in practice

## Next stage (not covered in this spec)

- Reinforcement learning loop: frozen main LLM plus generator perform real
  agent tool-use tasks, success or failure of tool calls is scored
  automatically, and this reward signal further fine-tunes the generator
  network only, starting from the pretrained warm-start checkpoint above
