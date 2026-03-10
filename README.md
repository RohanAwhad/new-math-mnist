# New Math Ops

Synthetic benchmark for strict left-to-right arithmetic evaluation with deterministic generation.

- 50/50 family split: `normal` (`+`, `-`, `*`, `/`) and `new` (`##`, `@@`, `$$`)
- Shared difficulty levels: `L1` (1-2 ops), `L2` (3-6 ops), `L3` (6-10 ops)
- Equal difficulty mix (1/3 each) inside both families
- Division uses floor division and never samples division by zero

## Files

- `new_math_ops/generate_dataset.py` - dataset generator (`dataset.jsonl` + `manifest.json`)
- `new_math_ops/prompts.py` - system/user prompts and prompt version
- `new_math_ops/llm_client.py` - async LiteLLM client wrapper (`litellm.acompletion`)
- `new_math_ops/evaluate.py` - async evaluation runner and metrics writer
- `tests/` - unit tests for generation, prompts, client, and evaluator

## Generate

```bash
python -m new_math_ops.generate_dataset \
  --output-dir benchmarks/data/new_math_ops_v2 \
  --seed 20260304 \
  --num-examples 70300
```

`--num-examples` must be even to preserve the strict 50/50 family split.

## Evaluate

```bash
python -m new_math_ops.evaluate \
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

## Install as dependency

```bash
uv add "new-math-ops @ git+https://github.com/RohanAwhad/new-math-mnist.git@<commit_sha>"
```

Then import from Python:

```python
import new_math_ops

messages = new_math_ops.build_messages("9 / 2 ## 3")
answer = new_math_ops.parse_final_answer("<final_answer>1</final_answer>")
```

Top-level imports/scripts (for example `import prompts` or `python generate_dataset.py`) are no longer supported.

## Typing and tests

```bash
uv run mypy new_math_ops tests
uv run python -m unittest discover -s tests -p "test_*.py"
```
