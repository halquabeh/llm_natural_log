"""Model execution and experiment runners."""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

from natural_log.config import ExperimentConfig
from natural_log.data import (
    first_integer,
    generate_prompts_numerals,
    generate_prompts_symbols,
    numeric_interval,
    string_to_number,
)
from natural_log.logging_utils import setup_logging
from natural_log.transforms import (
    analyze_transformed_hidden_states,
    summarize_metric_runs,
    transform_hidden_states,
)


class RunSummary(dict):
    """Dictionary that also supports the old tuple-style result indexing."""

    def __getitem__(self, key: Any) -> Any:
        if key == 0:
            return super().__getitem__("PCA")
        if key == 1:
            return super().__getitem__("PLS")
        return super().__getitem__(key)


class Analyzer:
    """Collect hidden states, transform them, and run paper experiments."""

    def __init__(
        self,
        args: ExperimentConfig | Any,
        logger: logging.Logger | None = None,
        load_model: bool = True,
    ) -> None:
        self.config = args if isinstance(args, ExperimentConfig) else ExperimentConfig.from_namespace(args)
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logging(
            "natural_log",
            self.config.log_dir,
            filename=f"{self.config.slug()}.log",
        )

        self.data = self.config.data
        self.num_examples = self.config.num_examples
        self.k = self.config.k
        self.groups = list(self.config.groups)
        self.context = self.config.context
        self.upper_bound = self.config.resolved_upper_bound
        self.Tdim = self.config.Tdim
        self.runs = self.config.runs
        self.log: list[dict[str, Any]] = []
        self.results: Any = None
        self.states: dict[int, dict[str, Any]] | None = None

        self.device = torch.device(
            f"cuda:{self.config.device}" if torch.cuda.is_available() else "cpu"
        )
        self._set_seed(self.config.seed)

        self.tokenizer = None
        self.model = None
        if load_model:
            self._login_if_token_available()
            self.logger.info("Loading model %s on %s", self.config.model_name, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                local_files_only=self.config.local_files_only,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                local_files_only=self.config.local_files_only,
            ).to(self.device)
            self.model.eval()
            if self.tokenizer.pad_token_id is None and self.model.config.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.filename = f"{self.config.slug()}_{self.config.transform.upper()}.pkl"
        self.save_path = self.save_dir / self.filename

    def _login_if_token_available(self) -> None:
        token = (
            self.config.hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )
        if token:
            login(token=token, add_to_git_credential=False)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def save_to_file(self, obj: Any | None = None, path: str | Path | None = None) -> Path:
        """Save an object, defaulting to the most recent results."""

        payload = self.results if obj is None else obj
        output_path = Path(path or self.save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle)
        self.logger.info("Saved results to %s", output_path)
        return output_path

    def _require_model(self) -> tuple[Any, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Analyzer was created with load_model=False")
        return self.model, self.tokenizer

    def collect_states(self, seed: int | None = None) -> dict[int, dict[str, Any]]:
        """Collect last-token hidden states for controlled prompts."""

        model, tokenizer = self._require_model()
        rng = random.Random(self.config.seed if seed is None else seed)
        if self.data.lower() in {"numerics", "numbers", "numeric"}:
            prompts = generate_prompts_numerals(
                self.k,
                self.num_examples,
                self.upper_bound,
                self.groups,
                numeric_interval,
                context=self.context,
                rng=rng,
            )
        else:
            prompts = generate_prompts_symbols(
                self.k,
                self.num_examples,
                self.groups,
                context=self.context,
                rng=rng,
            )

        model_num_layers = model.config.num_hidden_layers
        results: dict[int, dict[str, Any]] = {
            layer: {"hidden_states": {}, "answers": {}} for layer in range(model_num_layers)
        }

        self.logger.info(
            "Collecting states: data=%s groups=%s k=%s examples=%s context=%s",
            self.data,
            self.groups,
            self.k,
            self.num_examples,
            self.context,
        )

        with torch.no_grad():
            for group, prompt_list in prompts.items():
                group_hidden_states = {layer: [] for layer in range(model_num_layers)}
                group_answers: list[int | str] = []

                for prompt in prompt_list:
                    inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                    outputs = model(**inputs, output_hidden_states=True)
                    last_value = prompt[prompt.rfind(",") + 1 : prompt.rfind("=")]
                    last_token_by_layer = [
                        outputs.hidden_states[layer][:, -1, :].detach().cpu()
                        for layer in range(model_num_layers)
                    ]
                    for layer in range(model_num_layers):
                        group_hidden_states[layer].append(last_token_by_layer[layer])
                    if self.data.lower() not in {"numerics", "numbers", "numeric"}:
                        last_value = string_to_number(last_value)
                    group_answers.append(last_value)

                for layer in range(model_num_layers):
                    results[layer]["hidden_states"][group] = group_hidden_states[layer]
                    results[layer]["answers"][group] = group_answers

        self.states = results
        return results

    def analyze(self, transform: str = "PCA") -> dict[int, dict[str, Any]]:
        """Transform current states and compute geometry metrics."""

        if self.states is None:
            self.states = self.collect_states()
        transformed = transform_hidden_states(
            self.states,
            method=transform,
            num_components=self.Tdim,
            logger=self.logger,
        )
        return analyze_transformed_hidden_states(transformed)

    def run_multiple(self, methods: tuple[str, ...] = ("PCA", "PLS")) -> RunSummary:
        """Repeat state collection and summarize each requested transform."""

        collected_runs: dict[str, list[dict[int, dict[str, Any]]]] = {
            method.upper(): [] for method in methods
        }
        for run_index in range(self.runs):
            self.logger.info("Starting run %s/%s", run_index + 1, self.runs)
            self.states = self.collect_states(seed=self.config.seed + run_index)
            for method in methods:
                collected_runs[method.upper()].append(self.analyze(transform=method))

        summary = RunSummary({
            method: summarize_metric_runs(run_results)
            for method, run_results in collected_runs.items()
        })
        self.results = summary
        return summary

    def string_to_number(self, text: str) -> int:
        """Backward-compatible string conversion helper."""

        return string_to_number(text)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        *,
        return_full_text: bool = True,
    ) -> str:
        """Greedily generate text from the model."""

        model, tokenizer = self._require_model()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            do_sample=False,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else model.config.eos_token_id
            ),
        )
        tokens = output[0] if return_full_text else output[0, input_ids.shape[-1] :]
        return tokenizer.decode(tokens, skip_special_tokens=True)

    def which_is_larger(self, num_dem: int) -> list[str]:
        """Backward-compatible wrapper for the numeric comparison experiment."""

        results = self.run_numeric_comparison(num_demonstrations=(num_dem,))
        lines: list[str] = []
        for row in results:
            line = f"Accuracy for group {row['group']} is: {row['accuracy']:.2f}"
            self.logger.info(line)
            lines.append(line)
        return lines

    def run_numeric_comparison(
        self,
        num_demonstrations: tuple[int, ...] = (0, 3),
        n_pairs: int = 100,
        gaps: tuple[int, ...] = (1, 5, 10),
    ) -> list[dict[str, Any]]:
        """Run the pairwise 'which is larger?' downstream check."""

        rng = random.Random(self.config.seed)
        demonstrations = [
            "Which is larger 10 or 7? 10\n",
            "Which is larger 290 or 305? 305\n",
            "Which is larger 1232 or 1124? 1232\n",
        ]
        rows: list[dict[str, Any]] = []

        for num_dem in num_demonstrations:
            prefix = "".join(demonstrations[:num_dem])
            for group in self.groups:
                interval = list(numeric_interval(group))
                correct = 0
                attempts: list[dict[str, Any]] = []

                for _ in range(n_pairs):
                    gap = rng.choice(gaps)
                    lower_candidates = [value for value in interval if value + gap in interval]
                    if not lower_candidates:
                        first, second = rng.sample(interval, 2)
                    else:
                        first = rng.choice(lower_candidates)
                        second = first + gap
                    if rng.random() < 0.5:
                        first, second = second, first

                    prompt = (
                        f"{prefix}Which is larger {first} or {second}? "
                        "Answer with one number only: "
                    )
                    response = self.generate(
                        prompt,
                        max_new_tokens=10,
                        return_full_text=False,
                    ).strip()
                    prediction = first_integer(response)
                    target = max(first, second)
                    is_correct = prediction == target
                    correct += int(is_correct)
                    attempts.append(
                        {
                            "pair": [first, second],
                            "gap": abs(first - second),
                            "target": target,
                            "prediction": prediction,
                            "response": response,
                            "correct": is_correct,
                        }
                    )

                accuracy = correct / n_pairs
                row = {
                    "model": self.config.model_name,
                    "group": group,
                    "num_demonstrations": num_dem,
                    "n_pairs": n_pairs,
                    "accuracy": accuracy,
                    "attempts": attempts,
                }
                self.logger.info(
                    "Numeric comparison: demos=%s group=%s accuracy=%.3f",
                    num_dem,
                    group,
                    accuracy,
                )
                rows.append(row)

        return rows

    def save_json(self, payload: Any, path: str | Path) -> Path:
        """Save a JSON artifact."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.logger.info("Saved JSON to %s", output_path)
        return output_path

    def collect_state_real(self, data: Any) -> dict[int, dict[str, Any]]:
        """Collect states for a dataframe-like real-world prompt/value table.

        Expected columns are ``prompts`` and ``value``. If a ``group`` column is
        present it is used to restore group structure; otherwise all examples are
        placed in a single ``real`` group.
        """

        model, tokenizer = self._require_model()
        def column(table: Any, name: str) -> Any:
            if hasattr(table, "__contains__") and name in table:
                return table[name]
            if hasattr(table, name):
                return getattr(table, name)
            raise KeyError(f"Missing required column: {name}")

        prompts = list(column(data, "prompts"))
        answers = list(column(data, "value"))
        try:
            groups = list(column(data, "group"))
        except KeyError:
            groups = ["real"] * len(prompts)

        model_num_layers = model.config.num_hidden_layers
        results: dict[int, dict[str, Any]] = {
            layer: {"hidden_states": {}, "answers": {}} for layer in range(model_num_layers)
        }

        with torch.no_grad():
            for group in sorted(set(groups), key=str):
                group_hidden_states = {layer: [] for layer in range(model_num_layers)}
                group_answers: list[Any] = []
                for prompt, answer, row_group in zip(prompts, answers, groups):
                    if row_group != group:
                        continue
                    inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                    outputs = model(**inputs, output_hidden_states=True)
                    for layer in range(model_num_layers):
                        group_hidden_states[layer].append(
                            outputs.hidden_states[layer][:, -1, :].detach().cpu()
                        )
                    group_answers.append(answer)

                for layer in range(model_num_layers):
                    results[layer]["hidden_states"][group] = group_hidden_states[layer]
                    results[layer]["answers"][group] = group_answers

        self.states = results
        return results
