# Natural Log

[Number representations in LLMs ↗](https://openreview.net/forum?id=m3m5sNVNDe)

## Install

```bash
pip install -r requirements.txt
```

If a model is gated, set a Hugging Face token before running experiments:

```bash
export HF_TOKEN=...
```

## Project Structure

- `natural_log/`: reusable package code.
- `scripts/`: one runnable script per paper result.

## Reproduce Paper

Main PCA table plus appendix PLS table:

```bash
python scripts/make_table1.py --model gpt2 --runs 3 --device 0
```

PCA projection figure with systematic magnitude colors:

```bash
python scripts/make_pca_figure.py --model gpt2 --data numerics --layers best --device 0
```

Context-example sweep:

```bash
python scripts/run_context_sweep.py --models qwen,llama,mistral,deepseek --device 0
```

Motivating downstream comparison result:

```bash
python scripts/run_numeric_comparison.py --model gpt2 --groups 1,2,3,4,5 --device 0
```

All scripts log to the terminal and to `logs/*.log`. Results are written under `results/`.

```bibtex
@article{alquboj2025number,
  title={Number representations in llms: A computational parallel to human perception},
  author={AlquBoj, HV and AlQuabeh, Hilal and Bojkovic, Velibor and Hiraoka, Tatsuya and El-Shangiti, Ahmed Oumar and Nwadike, Munachiso and Inui, Kentaro},
  journal={arXiv preprint arXiv:2502.16147},
  year={2025}
}
```
