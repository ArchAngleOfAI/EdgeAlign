"""Approximate character-budget packing of short text passages into
larger training examples (see MEMORY.md "Pretraining data actually
used" for why).

Wikipedia passages are short (~100 words) and fairly uniform. Emitting
one passage per training example would waste most of a 512-token
sequence budget on padding. Instead, concatenate several passages
(separated by the tokenizer's real EOS token, matching how the frozen
model itself was almost certainly pretrained on packed documents) until
the packed string is expected to approach max_seq_len tokens, then hand
the whole packed string off as a single "prompt" -- indistinguishable,
as far as the rest of the pipeline is concerned, from any other prompt
string. The existing tokenizer call's truncation=True (in
edgealign/train.py's training_step / edgealign/evaluate.py's evaluate)
is the safety net for the rare case this estimate overshoots; nothing
here needs to be exact.
"""
from typing import Iterator, List

# Empirically measured on the real Qwen 3 32B tokenizer against 500 real
# Wikipedia passages from wiki-18.jsonl: 318,120 chars / 75,521 tokens =
# 4.212 chars/token (see MEMORY.md). This is only ever used to decide
# how many passages to concatenate before handing off to the real
# tokenizer, which truncates exactly -- the estimate never needs to be
# exact, only roughly in the right range.
_CHARS_PER_TOKEN_ESTIMATE = 4.2


def pack_texts(
    texts: Iterator[str],
    max_seq_len: int,
    separator: str = "\n\n",
    target_fill: float = 0.85,
) -> Iterator[str]:
    """Greedily concatenates texts (joined by `separator`) until the
    running character count approaches target_fill * max_seq_len worth
    of estimated tokens, then yields the packed string and starts a new
    one. The final, possibly-short remainder is yielded too."""
    if not 0 < target_fill <= 1:
        raise ValueError(f"target_fill must be in (0, 1], got {target_fill}")

    target_chars = max_seq_len * _CHARS_PER_TOKEN_ESTIMATE * target_fill

    buffer: List[str] = []
    buffer_chars = 0

    for text in texts:
        buffer.append(text)
        buffer_chars += len(text) + len(separator)

        if buffer_chars >= target_chars:
            yield separator.join(buffer)
            buffer = []
            buffer_chars = 0

    if buffer:
        yield separator.join(buffer)
