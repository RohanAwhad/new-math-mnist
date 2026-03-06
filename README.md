# New Math Ops

Synthetic benchmark for strict left-to-right arithmetic evaluation with deterministic generation.

- 50/50 family split: `normal` (`+`, `-`, `*`, `/`) and `new` (`##`, `@@`, `$$`)
- Shared difficulty levels: `L1` (1-5 ops), `L2` (6-10 ops), `L3` (11-20 ops)
- Equal difficulty mix (1/3 each) inside both families
- Division uses floor division and never samples division by zero

## Files

- `generate_dataset.py` - dataset generator (`dataset.jsonl` + `manifest.json`)
- `prompts.py` - system/user prompts and prompt version
- `llm_client.py` - async LiteLLM client wrapper (`litellm.acompletion`)
- `evaluate.py` - async evaluation runner and metrics writer
- `tests/` - unit tests for generation, prompts, client, and evaluator

## Generate

```bash
python benchmarks/new_math_ops/generate_dataset.py \
  --output-dir benchmarks/data/new_math_ops_v2 \
  --seed 20260304 \
  --num-examples 70300
```

`--num-examples` must be even to preserve the strict 50/50 family split.

## Evaluate

```bash
python benchmarks/new_math_ops/evaluate.py \
  --dataset benchmarks/data/new_math_ops_v2/dataset.jsonl \
  --manifest benchmarks/data/new_math_ops_v2/manifest.json \
  --model openai/gpt-4o-mini \
  --temperature 1.0 \
  --concurrency 20
```

## Dev setup

```bash
uv sync
uv run pre-commit install
```

This installs a git hook that runs ruff (format + lint) and mypy on every commit.

## Typing and tests

```bash
uv run mypy contracts.py generate_dataset.py prompts.py llm_client.py evaluate.py tests
uv run python -m unittest discover -s tests -p "test_*.py"
```
