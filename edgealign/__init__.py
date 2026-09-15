"""Shared infrastructure for the soft prompt generator, prototype phase.

Implements PROMPTS/soft_prompt_generator_infrastructure_spec.md: the
frozen-LLM harness, embedding-space injection, KL-distillation training
loop, and initialization scheme that every generator front-end variant
(v1/v2/v3) plugs into. The front-end itself is not implemented here --
see edgealign.generators.base.GeneratorFrontend for the interface.
"""
