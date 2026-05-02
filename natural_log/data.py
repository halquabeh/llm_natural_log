"""Prompt generation and small parsing helpers."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable, Sequence

from natural_log.config import MODEL_GROUPS, search_models


def numeric_interval(group: int, width: int = 20) -> range:
    """Return the log-spaced interval used by the controlled experiments."""

    if group <= 1:
        return range(1, 40)
    center = 10**group
    return range(int(center - width), int(center + width))


def generate_number_prompt(number_list: Sequence[int]) -> str:
    """Format an equation-style in-context prompt ending at the query value."""

    if not number_list:
        raise ValueError("number_list must contain at least one number")
    if len(number_list) == 1:
        return f"{number_list[-1]}="
    context = ",".join(f"{num}={num}" for num in number_list[:-1])
    return f"{context},{number_list[-1]}="


def generate_prompts_numerals(
    k: int,
    num_examples: int,
    upper_bound: int,
    group_range: Iterable[int],
    interval_function: Callable[[int], range] | None = None,
    context: str = "random",
    rng: random.Random | None = None,
) -> dict[int, list[str]]:
    """Generate controlled numeric prompts grouped by magnitude."""

    rng = rng or random
    interval_function = interval_function or numeric_interval
    groups = list(group_range)

    sampled_numbers: dict[int, list[int]] = {}
    for group in groups:
        interval = list(interval_function(group))
        sampled_numbers[group] = rng.choices(interval, k=k)

    prompts: dict[int, list[str]] = {}
    for group, numbers_in_group in sampled_numbers.items():
        prompts[group] = []
        for number in numbers_in_group:
            if context == "random":
                examples = [rng.randint(0, upper_bound) for _ in range(num_examples)]
            elif context == "fixed":
                fixed_context = [4, 54, 432, 9543, 10002, 99991]
                examples = fixed_context[:num_examples]
            elif context == "same":
                examples = numbers_in_group[:num_examples]
            else:
                raise ValueError(f"Unknown context option: {context}")
            prompts[group].append(generate_number_prompt([*examples, number]))

    return prompts


def generate_symbols(
    group_size: int,
    num_examples: int,
    base_alphabet: str = "abcdefghijklmnopqrstuvwxyz",
    rng: random.Random | None = None,
) -> list[str]:
    """Generate random symbol strings up to a specified length."""

    rng = rng or random
    return [
        "".join(rng.choices(base_alphabet, k=rng.randint(1, group_size)))
        for _ in range(num_examples)
    ]


def generate_prompts_symbols(
    k: int,
    num_examples: int,
    group_range: Iterable[int],
    context: str = "random",
    rng: random.Random | None = None,
) -> dict[int, list[str]]:
    """Generate length-matched non-numeric control prompts."""

    rng = rng or random
    groups = list(group_range)
    base_alphabet = "abcdefghijklmnopqrstuvwxyz"
    common_context_symbols = generate_symbols(
        max(groups), max(1, k * max(1, num_examples)), base_alphabet, rng=rng
    )

    prompts: dict[int, list[str]] = {}
    for group in groups:
        prompts[group] = []
        for _ in range(k):
            if context in {"random", "fixed"}:
                context_symbols = rng.sample(common_context_symbols, num_examples)
            elif context == "same":
                context_symbols = generate_symbols(group, num_examples, base_alphabet, rng=rng)
            else:
                raise ValueError(f"Unknown context option: {context}")
            query_symbol = "".join(rng.choices(base_alphabet, k=group))
            full_symbol_list = [*context_symbols, query_symbol]
            formatted = ",".join(f"{symbol}={symbol}" for symbol in full_symbol_list[:-1])
            prompts[group].append(f"{formatted},{query_symbol}=")

    return prompts


def string_to_number(text: str) -> int:
    """Convert lowercase text into a base-26 integer."""

    value = 0
    for char in text.lower():
        if "a" <= char <= "z":
            value = value * 26 + (ord(char) - ord("a"))
    return value


def strings_to_numbers(strings: Iterable[str]) -> list[int]:
    """Convert multiple strings into base-26 integers."""

    return [string_to_number(text) for text in strings]


def first_integer(text: str) -> int | None:
    """Parse the first integer from model output."""

    match = re.search(r"-?\d+", text)
    return int(match.group(0)) if match else None


__all__ = [
    "MODEL_GROUPS",
    "first_integer",
    "generate_number_prompt",
    "generate_prompts_numerals",
    "generate_prompts_symbols",
    "generate_symbols",
    "numeric_interval",
    "search_models",
    "string_to_number",
    "strings_to_numbers",
]
