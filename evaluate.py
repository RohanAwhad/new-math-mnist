from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import llm_client
import prompts


FINAL_ANSWER_PATTERN = re.compile(r"^\s*<final_answer>\s*([0-9])\s*</final_answer>\s*$")


def parse_final_answer(text: str) -> int | None:
    match = FINAL_ANSWER_PATTERN.match(text)
    if match is None:
        return None
    return int(match.group(1))


def load_dataset_rows(
    dataset_path: Path, limit: int | None = None
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _init_bucket() -> dict[str, float | int]:
    return {
        "total": 0,
        "correct": 0,
        "format_errors": 0,
        "accuracy": 0.0,
        "format_error_rate": 0.0,
    }


def compute_metrics(predictions: list[dict[str, object]]) -> dict[str, object]:
    total = len(predictions)
    correct = sum(1 for row in predictions if bool(row["is_correct"]))
    format_errors = sum(1 for row in predictions if bool(row["format_error"]))

    by_difficulty: dict[str, dict[str, float | int]] = {}
    by_n_ops: dict[str, dict[str, float | int]] = {}

    for row in predictions:
        difficulty = str(row["difficulty_level"])
        n_ops_key = str(row["n_ops"])

        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = _init_bucket()
        if n_ops_key not in by_n_ops:
            by_n_ops[n_ops_key] = _init_bucket()

        for bucket in (by_difficulty[difficulty], by_n_ops[n_ops_key]):
            bucket["total"] = int(bucket["total"]) + 1
            if bool(row["is_correct"]):
                bucket["correct"] = int(bucket["correct"]) + 1
            if bool(row["format_error"]):
                bucket["format_errors"] = int(bucket["format_errors"]) + 1

    for bucket in [*by_difficulty.values(), *by_n_ops.values()]:
        bucket_total = int(bucket["total"])
        if bucket_total == 0:
            continue
        bucket["accuracy"] = int(bucket["correct"]) / bucket_total
        bucket["format_error_rate"] = int(bucket["format_errors"]) / bucket_total

    accuracy = (correct / total) if total else 0.0
    format_error_rate = (format_errors / total) if total else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "format_errors": format_errors,
        "format_error_rate": format_error_rate,
        "by_difficulty": by_difficulty,
        "by_n_ops": by_n_ops,
    }


async def evaluate_dataset_rows(
    *,
    rows: list[dict[str, object]],
    client: object,
    concurrency: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_row(row: dict[str, object]) -> dict[str, object]:
        async with semaphore:
            expression = str(row["input"])
            expected_output = int(row["expected_output"])
            metadata = dict(row["metadata"])

            start = time.perf_counter()
            raw_response = await client.complete(prompts.build_messages(expression))
            latency = time.perf_counter() - start

            parsed = parse_final_answer(raw_response)
            format_error = parsed is None
            is_correct = parsed == expected_output

            return {
                "id": str(metadata["id"]),
                "input": expression,
                "expected_output": expected_output,
                "predicted_output": parsed,
                "is_correct": is_correct,
                "format_error": format_error,
                "raw_response": raw_response,
                "difficulty_level": str(metadata["difficulty_level"]),
                "n_ops": int(metadata["n_ops"]),
                "latency_seconds": round(latency, 6),
            }

    predictions = await asyncio.gather(*(evaluate_row(row) for row in rows))
    metrics = compute_metrics(predictions)
    return predictions, metrics


def write_run_artifacts(
    *,
    output_root: Path,
    predictions: list[dict[str, object]],
    metrics: dict[str, object],
    run_config: dict[str, object],
    run_id: str | None = None,
) -> Path:
    resolved_run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as file:
        for row in predictions:
            file.write(json.dumps(row, sort_keys=True))
            file.write("\n")

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    config_path = run_dir / "run_config.json"
    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate models on New Math Ops dataset"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/new_math_ops/runs")
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    return parser


async def _run() -> None:
    args = build_parser().parse_args()

    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if args.limit is not None and args.limit < 0:
        raise ValueError("limit must be non-negative")

    rows = load_dataset_rows(args.dataset, args.limit)

    client = llm_client.LiteLLMClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    predictions, metrics = await evaluate_dataset_rows(
        rows=rows,
        client=client,
        concurrency=args.concurrency,
    )

    run_config = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "dataset": str(args.dataset),
        "manifest": str(args.manifest) if args.manifest else None,
        "limit": args.limit,
        "prompt_version": prompts.PROMPT_VERSION,
    }

    run_dir = write_run_artifacts(
        output_root=args.output_dir,
        predictions=predictions,
        metrics=metrics,
        run_config=run_config,
        run_id=args.run_id,
    )

    print(f"wrote run artifacts to {run_dir}")
    print(
        json.dumps(
            {
                "total": metrics["total"],
                "correct": metrics["correct"],
                "accuracy": metrics["accuracy"],
                "format_error_rate": metrics["format_error_rate"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
