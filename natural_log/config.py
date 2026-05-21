"""Experiment configuration and model registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MODEL_GROUPS: dict[str, list[str]] = {
    "large": [
        "Qwen/Qwen1.5-7B",
        "meta-llama/Llama-2-7b-hf",
        "deepseek-ai/deepseek-llm-7b-base",
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Llama-3.1-8B",
    ],
    "instruct": [
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
    "rnn": [
        "tiiuae/falcon-mamba-7b",
    ],
    "small": [
        "EleutherAI/pythia-2.8b",
        "openai-community/gpt2-large",
    ],
}


def tokenize_model_name(name: str) -> list[str]:
    """Split a Hugging Face model name into searchable tokens."""

    import re

    return re.split(r"[/\-_.]", name)


def all_model_names() -> list[str]:
    """Return the flattened model registry in a stable order."""

    names: list[str] = []
    for group_names in MODEL_GROUPS.values():
        names.extend(group_names)
    return names


def search_models(query: str | None) -> str | None:
    """Search the local model registry by substring.

    If the query already looks like a full Hugging Face repository id, it is
    returned unchanged. This keeps command-line usage convenient for models not
    yet listed in the registry.
    """

    if not query:
        return None
    if "/" in query:
        return query

    q = query.lower()
    matches: list[str] = []
    for model_name in all_model_names():
        tokens = tokenize_model_name(model_name)
        if q in model_name.lower() or any(q in token.lower() for token in tokens):
            matches.append(model_name)
    return matches[0] if matches else None


@dataclass(slots=True)
class ExperimentConfig:
    """Runtime options shared by analysis scripts."""

    model_name: str = "openai-community/gpt2-large"
    transform: str = "PCA"
    Tdim: int = 1
    k: int = 30
    num_examples: int = 3
    context: str = "random"
    data: str = "numerics"
    groups: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    upper_bound: int | None = None
    save: bool = True
    plot: bool = True
    device: str = "0"
    runs: int = 3
    seed: int = 42
    save_dir: Path = Path("f_results")
    log_dir: Path = Path("logs")
    hf_token: str | None = None
    local_files_only: bool = False
    max_new_tokens: int = 10

    @property
    def resolved_upper_bound(self) -> int:
        """Upper bound for random context numbers."""

        if self.upper_bound is not None:
            return self.upper_bound
        return 10 ** max(self.groups)

    @classmethod
    def from_namespace(cls, args: Any) -> "ExperimentConfig":
        """Build a config from argparse or an older notebook-style object."""

        values: dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(args, field_name):
                values[field_name] = getattr(args, field_name)
        if "save_dir" in values:
            values["save_dir"] = Path(values["save_dir"])
        if "log_dir" in values:
            values["log_dir"] = Path(values["log_dir"])
        config = cls(**values)
        if config.upper_bound is None:
            config.upper_bound = config.resolved_upper_bound
        return config

    def slug(self) -> str:
        """Stable filename stem for this configuration."""

        model = self.model_name.replace("/", "_")
        groups = "-".join(str(group) for group in self.groups)
        return (
            f"{model}_{self.data}_k{self.k}_n{self.num_examples}"
            f"_ctx-{self.context}_g{groups}_seed{self.seed}"
        )
