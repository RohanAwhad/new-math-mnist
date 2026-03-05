from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import llm_client
import prompts
from contracts import (
    BucketCounts,
    BucketMetrics,
    LLMCompletionClient,
    DatasetRow,
    DifficultyLevel,
    Metrics,
    PredictionRow,
    RunConfig,
)
from tqdm import tqdm


FINAL_ANSWER_PATTERN = re.compile(r"<final_answer>\s*([0-9])\s*</final_answer>")


def parse_final_answer(text: str) -> int | None:
    match = FINAL_ANSWER_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1))


def load_dataset_rows(dataset_path: Path, limit: int | None = None) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            row: DatasetRow = json.loads(line)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _init_bucket() -> BucketCounts:
    return {
        "total": 0,
        "correct": 0,
        "format_errors": 0,
    }


def _finalize_bucket(bucket: BucketCounts) -> BucketMetrics:
    bucket_total = bucket["total"]
    if bucket_total == 0:
        accuracy = 0.0
        format_error_rate = 0.0
    else:
        accuracy = bucket["correct"] / bucket_total
        format_error_rate = bucket["format_errors"] / bucket_total

    return {
        "total": bucket["total"],
        "correct": bucket["correct"],
        "format_errors": bucket["format_errors"],
        "accuracy": accuracy,
        "format_error_rate": format_error_rate,
    }


def compute_metrics(predictions: list[PredictionRow]) -> Metrics:
    total = len(predictions)
    correct = sum(1 for row in predictions if bool(row["is_correct"]))
    format_errors = sum(1 for row in predictions if bool(row["format_error"]))

    by_difficulty: defaultdict[DifficultyLevel, BucketCounts] = defaultdict(
        _init_bucket
    )
    by_n_ops: defaultdict[int, BucketCounts] = defaultdict(_init_bucket)

    for row in predictions:
        difficulty = row["difficulty_level"]
        n_ops_key = row["n_ops"]

        for bucket in (by_difficulty[difficulty], by_n_ops[n_ops_key]):
            bucket["total"] += 1
            bucket["correct"] += int(row["is_correct"])
            if bool(row["format_error"]):
                bucket["format_errors"] += 1

    by_difficulty_metrics: dict[DifficultyLevel, BucketMetrics] = {}
    for key, bucket in by_difficulty.items():
        by_difficulty_metrics[key] = _finalize_bucket(bucket)

    by_n_ops_metrics: dict[int, BucketMetrics] = {}
    for key, bucket in by_n_ops.items():
        by_n_ops_metrics[key] = _finalize_bucket(bucket)

    accuracy = (correct / total) if total else 0.0
    format_error_rate = (format_errors / total) if total else 0.0

    metrics: Metrics = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "format_errors": format_errors,
        "format_error_rate": format_error_rate,
        "by_difficulty": by_difficulty_metrics,
        "by_n_ops": by_n_ops_metrics,
    }
    return metrics


def _serialize_metrics(metrics: Metrics) -> dict[str, object]:
    serialized_metrics: dict[str, object] = dict(metrics)
    serialized_metrics["by_n_ops"] = {
        str(n_ops): bucket for n_ops, bucket in metrics["by_n_ops"].items()
    }
    return serialized_metrics


async def evaluate_dataset_rows(
    *,
    rows: list[DatasetRow],
    client: LLMCompletionClient,
    concurrency: int,
) -> tuple[list[PredictionRow], Metrics]:
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_row(row: DatasetRow) -> PredictionRow:
        async with semaphore:
            expression = row["input"]
            expected_output = row["expected_output"]
            metadata = row["metadata"]

            start = time.perf_counter()
            raw_response = await client.complete(prompts.build_messages(expression))
            latency = time.perf_counter() - start

            parsed = parse_final_answer(raw_response)
            format_error = parsed is None
            is_correct = parsed == expected_output

            return {
                "id": metadata["id"],
                "input": expression,
                "expected_output": expected_output,
                "predicted_output": parsed,
                "is_correct": is_correct,
                "format_error": format_error,
                "raw_response": raw_response,
                "difficulty_level": metadata["difficulty_level"],
                "n_ops": metadata["n_ops"],
                "latency_seconds": round(latency, 6),
            }

    predictions: list[PredictionRow] = []
    pending = [evaluate_row(row) for row in rows]
    for completed in tqdm(
        asyncio.as_completed(pending),
        total=len(pending),
        desc="Evaluating",
        unit="sample",
        disable=not sys.stderr.isatty(),
    ):
        predictions.append(await completed)

    metrics = compute_metrics(predictions)
    return predictions, metrics


def write_run_artifacts(
    *,
    output_root: Path,
    predictions: list[PredictionRow],
    metrics: Metrics,
    run_config: RunConfig,
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
    serialized_metrics = _serialize_metrics(metrics)
    metrics_path.write_text(
        json.dumps(serialized_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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

    run_config: RunConfig = {
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
