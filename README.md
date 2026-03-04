# New Math Ops

Synthetic benchmark for symbolic operator execution with deterministic generation.

## Files

- `generate_dataset.py` - dataset generator (`dataset.jsonl` + `manifest.json`)
- `prompts.py` - system/user prompts and prompt version
- `llm_client.py` - async LiteLLM client wrapper (`litellm.acompletion`)
- `evaluate.py` - async evaluation runner and metrics writer
- `tests/` - unit tests for generation, prompts, client, and evaluator

## Generate

```bash
python benchmarks/new_math_ops/generate_dataset.py \
  --output-dir benchmarks/data/new_math_ops_v1 \
  --seed 20260304 \
  --num-examples 70300
```

## Evaluate

```bash
python benchmarks/new_math_ops/evaluate.py \
  --dataset benchmarks/data/new_math_ops_v1/dataset.jsonl \
  --manifest benchmarks/data/new_math_ops_v1/manifest.json \
  --model openai/gpt-4o-mini \
  --temperature 1.0 \
  --concurrency 20
```
