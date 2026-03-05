from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from contracts import (
    DIFFICULTY_LEVELS,
    LEVEL_S1,
    LEVEL_S2,
    LEVEL_S3,
    OPERATORS,
    DatasetRow,
    DifficultyLevel,
    Manifest,
    Operator,
)

DIGITS: tuple[int, ...] = tuple(range(10))
MAX_S1_EXPRESSIONS = len(DIGITS) * len(OPERATORS) * len(DIGITS)

LEVEL_ORDER: tuple[DifficultyLevel, ...] = DIFFICULTY_LEVELS

# This mix comes from the previous recommendation with S4 removed and ratios
# re-normalized.
DEFAULT_LEVEL_MIX: dict[DifficultyLevel, float] = {
    LEVEL_S1: 0.004,
    LEVEL_S2: 0.498,
    LEVEL_S3: 0.498,
}


def apply_operator(lhs: int, operator: Operator, rhs: int) -> int:
    if operator is Operator.ABS_DIFF:
        return abs(lhs - rhs)
    if operator is Operator.MAX:
        return max(lhs, rhs)
    if operator is Operator.MIN:
        return min(lhs, rhs)
    raise ValueError(f"unknown operator: {operator}")


def evaluate_expression(numbers: list[int], operators: list[Operator]) -> int:
    if len(numbers) != len(operators) + 1:
        raise ValueError("numbers must be exactly one longer than operators")

    value = numbers[0]
    for operator, rhs in zip(operators, numbers[1:], strict=True):
        value = apply_operator(value, operator, rhs)
    return value


def render_expression(numbers: list[int], operators: list[Operator]) -> str:
    tokens: list[str] = [str(numbers[0])]
    for operator, rhs in zip(operators, numbers[1:], strict=True):
        tokens.append(operator)
        tokens.append(str(rhs))
    return " ".join(tokens)


def sample_expression(
    rng: random.Random, min_ops: int, max_ops: int
) -> tuple[list[int], list[Operator]]:
    n_ops = rng.randint(min_ops, max_ops)
    numbers = [rng.randint(0, 9) for _ in range(n_ops + 1)]
    operators: list[Operator] = [rng.choice(OPERATORS) for _ in range(n_ops)]
    return numbers, operators


def allocate_by_ratio(
    total: int,
    ratios: dict[DifficultyLevel, float],
    levels: tuple[DifficultyLevel, ...],
) -> dict[DifficultyLevel, int]:
    if total < 0:
        raise ValueError("total must be non-negative")

    raw: dict[DifficultyLevel, float] = {
        level: total * ratios[level] for level in levels
    }
    counts: dict[DifficultyLevel, int] = {level: int(raw[level]) for level in levels}
    remainder = total - sum(counts.values())

    for level in sorted(
        levels, key=lambda level: raw[level] - counts[level], reverse=True
    )[:remainder]:
        counts[level] += 1

    return counts


def allocate_level_counts(num_examples: int) -> dict[DifficultyLevel, int]:
    counts = allocate_by_ratio(num_examples, DEFAULT_LEVEL_MIX, LEVEL_ORDER)

    if counts[LEVEL_S1] <= MAX_S1_EXPRESSIONS:
        return counts

    remaining_total = num_examples - MAX_S1_EXPRESSIONS
    remaining_levels = (LEVEL_S2, LEVEL_S3)
    remaining_weight = DEFAULT_LEVEL_MIX[LEVEL_S2] + DEFAULT_LEVEL_MIX[LEVEL_S3]
    remaining_mix: dict[DifficultyLevel, float] = {
        LEVEL_S2: DEFAULT_LEVEL_MIX[LEVEL_S2] / remaining_weight,
        LEVEL_S3: DEFAULT_LEVEL_MIX[LEVEL_S3] / remaining_weight,
    }
    remaining_counts = allocate_by_ratio(
        remaining_total, remaining_mix, remaining_levels
    )

    return {
        LEVEL_S1: MAX_S1_EXPRESSIONS,
        LEVEL_S2: remaining_counts[LEVEL_S2],
        LEVEL_S3: remaining_counts[LEVEL_S3],
    }


@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: str
    difficulty_level: DifficultyLevel
    input: str
    expected_output: int
    n_ops: int
    op_seq: tuple[Operator, ...]

    def to_dict(self) -> DatasetRow:
        return {
            "input": self.input,
            "expected_output": self.expected_output,
            "metadata": {
                "id": self.sample_id,
                "difficulty_level": self.difficulty_level,
                "n_ops": self.n_ops,
                "op_seq": list(self.op_seq),
            },
        }


def generate_s1_primitive(
    size: int, rng: random.Random, seen: set[str]
) -> list[Sample]:
    if size > MAX_S1_EXPRESSIONS:
        raise ValueError(f"S1 size cannot exceed {MAX_S1_EXPRESSIONS}")

    pool: list[tuple[str, int, Operator]] = []
    for lhs in DIGITS:
        for operator in OPERATORS:
            for rhs in DIGITS:
                expression = f"{lhs} {operator} {rhs}"
                target = apply_operator(lhs, operator, rhs)
                pool.append((expression, target, operator))

    rng.shuffle(pool)

    samples: list[Sample] = []
    for expression, target, operator in pool:
        if len(samples) == size:
            break

        if expression in seen:
            continue

        samples.append(
            Sample(
                sample_id=f"s1_{len(samples):06d}",
                difficulty_level=LEVEL_S1,
                input=expression,
                expected_output=target,
                n_ops=1,
                op_seq=(operator,),
            )
        )
        seen.add(expression)

    return samples


def generate_random_level(
    *,
    difficulty_level: DifficultyLevel,
    sample_prefix: str,
    size: int,
    min_ops: int,
    max_ops: int,
    rng: random.Random,
    seen: set[str],
) -> list[Sample]:
    samples: list[Sample] = []
    attempts = 0
    max_attempts = max(size * 500, 100_000)

    while len(samples) < size:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                "generation stalled for "
                f"{difficulty_level}; requested={size}, got={len(samples)}"
            )

        numbers, operators = sample_expression(rng, min_ops, max_ops)
        expression = render_expression(numbers, operators)
        if expression in seen:
            continue

        target = evaluate_expression(numbers, operators)

        samples.append(
            Sample(
                sample_id=f"{sample_prefix}_{len(samples):06d}",
                difficulty_level=difficulty_level,
                input=expression,
                expected_output=target,
                n_ops=len(operators),
                op_seq=tuple(operators),
            )
        )
        seen.add(expression)

    return samples


def build_label_histogram(samples: list[Sample]) -> dict[str, int]:
    counts = {digit: 0 for digit in DIGITS}
    for sample in samples:
        counts[sample.expected_output] += 1
    return {str(label): count for label, count in counts.items()}


def write_jsonl(path: Path, rows: list[DatasetRow]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True))
            file.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic new-math synthetic benchmark"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260304)
    parser.add_argument("--num-examples", type=int, default=70_300)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.num_examples < 0:
        raise ValueError("num-examples must be non-negative")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    counts_by_level = allocate_level_counts(args.num_examples)

    seen_expressions: set[str] = set()
    rng_s1 = random.Random(args.seed + 11)
    rng_s2 = random.Random(args.seed + 29)
    rng_s3 = random.Random(args.seed + 47)
    rng_shuffle = random.Random(args.seed + 101)

    s1_samples = generate_s1_primitive(
        counts_by_level[LEVEL_S1], rng_s1, seen_expressions
    )
    s2_samples = generate_random_level(
        difficulty_level=LEVEL_S2,
        sample_prefix="s2",
        size=counts_by_level[LEVEL_S2],
        min_ops=2,
        max_ops=8,
        rng=rng_s2,
        seen=seen_expressions,
    )
    s3_samples = generate_random_level(
        difficulty_level=LEVEL_S3,
        sample_prefix="s3",
        size=counts_by_level[LEVEL_S3],
        min_ops=8,
        max_ops=20,
        rng=rng_s3,
        seen=seen_expressions,
    )

    all_samples = [*s1_samples, *s2_samples, *s3_samples]
    rng_shuffle.shuffle(all_samples)

    dataset_path = output_dir / "dataset.jsonl"
    write_jsonl(dataset_path, [sample.to_dict() for sample in all_samples])

    manifest: Manifest = {
        "benchmark_name": "new_math_ops_v1",
        "seed": args.seed,
        "rules": {
            "##": "abs(a-b)",
            "@@": "max(a,b)",
            "$$": "min(a,b)",
            "evaluation": "left_to_right",
            "digits": "0..9",
        },
        "requested_num_examples": args.num_examples,
        "generated_num_examples": len(all_samples),
        "default_level_mix": DEFAULT_LEVEL_MIX,
        "counts_by_difficulty": counts_by_level,
        "label_histogram": build_label_histogram(all_samples),
        "row_fields": {
            "input": "str",
            "expected_output": "int",
            "metadata": {
                "id": "str",
                "difficulty_level": "str",
                "n_ops": "int",
                "op_seq": "list[str]",
            },
        },
        "files": ["dataset.jsonl", "manifest.json"],
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"wrote dataset to {output_dir}")
    print(json.dumps({"rows": len(all_samples), **counts_by_level}, sort_keys=True))


if __name__ == "__main__":
    main()
