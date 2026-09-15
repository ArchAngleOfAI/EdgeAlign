"""Tiny in-code synthetic corpus, for local smoke testing ONLY.

Never a substitute for the real Stack-Edu / OpenCodeInstruct / xLAM
pretraining mix described in the infrastructure spec section 6 -- this
exists purely so the training loop can be exercised without any dataset
download or network access.
"""
import random
from typing import Iterator, List, Optional

SYNTHETIC_EXAMPLES = [
    "def add(a, b):\n    return a + b",
    "def is_palindrome(s):\n    return s == s[::-1]",
    "Write a function that returns the factorial of n.",
    "Explain what a for loop does in Python.",
    '<user>What is the weather in Paris?</user>\n'
    '<tools>[{"name": "get_weather", "parameters": {"city": "string"}}]</tools>',
    '<user>Book a flight from NYC to LA on Friday.</user>\n'
    '<tools>[{"name": "book_flight", "parameters": {"origin": "string", '
    '"destination": "string", "date": "string"}}]</tools>',
    "class Stack:\n    def __init__(self):\n        self.items = []",
    "Fix the bug in this function: def add(a, b): return a - b",
]


def synthetic_batches(batch_size: int, num_batches: Optional[int] = None, seed: int = 0) -> Iterator[List[str]]:
    """Yields batches of prompt strings drawn (with replacement) from the
    small hand-written corpus above. If num_batches is None, yields
    forever."""
    rng = random.Random(seed)
    i = 0
    while num_batches is None or i < num_batches:
        yield rng.choices(SYNTHETIC_EXAMPLES, k=batch_size)
        i += 1
