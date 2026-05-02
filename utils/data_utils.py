"""Backward-compatible wrappers around ``natural_log.data``."""

from natural_log.config import MODEL_GROUPS as all_models
from natural_log.config import search_models, tokenize_model_name
from natural_log.data import (
    first_integer,
    generate_number_prompt,
    generate_prompts_numerals,
    generate_prompts_symbols,
    generate_symbols,
    numeric_interval,
    string_to_number,
    strings_to_numbers,
)


def generate_number_prompts(number_list):
    """Old pluralized function name kept for notebooks."""

    return generate_number_prompt(number_list)


def generate_symbol_prompts(symbol_list, group_size):
    """Format a symbol prompt using the older helper name."""

    context = ",".join(f"{symbol}={symbol}" for symbol in symbol_list[:-1])
    last_input = "".join(symbol_list[-group_size:])
    return f"{context},{last_input}="


__all__ = [
    "all_models",
    "first_integer",
    "generate_number_prompt",
    "generate_number_prompts",
    "generate_prompts_numerals",
    "generate_prompts_symbols",
    "generate_symbol_prompts",
    "generate_symbols",
    "numeric_interval",
    "search_models",
    "string_to_number",
    "strings_to_numbers",
    "tokenize_model_name",
]
