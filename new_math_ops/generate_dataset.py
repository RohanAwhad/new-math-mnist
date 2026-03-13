from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .contracts import (
    ARITHMETIC_FAMILIES,
    DIFFICULTY_LEVELS,
    NEW_OPERATORS,
    NORMAL_OPERATORS,
    ArithmeticFamily,
    DatasetRow,
    DifficultyLevel,
    Manifest,
    Operator,
)

DIGITS: tuple[int, ...] = tuple(range(10))

DEFAULT_FAMILY_MIX: dict[ArithmeticFamily, float] = {
    ArithmeticFamily.NORMAL: 0.5,
    ArithmeticFamily.NEW: 0.5,
}

DEFAULT_LEVEL_MIX: dict[DifficultyLevel, float] = {
    DifficultyLevel.L1: 1 / 3,
    DifficultyLevel.L2: 1 / 3,
    DifficultyLevel.L3: 1 / 3,
}

LEVEL_BOUNDS: dict[DifficultyLevel, tuple[int, int]] = {
    DifficultyLevel.L1: (1, 2),
    DifficultyLevel.L2: (3, 6),
    DifficultyLevel.L3: (6, 10),
}

OPERATORS_BY_FAMILY: dict[ArithmeticFamily, tuple[Operator, ...]] = {
    ArithmeticFamily.NORMAL: NORMAL_OPERATORS,
    ArithmeticFamily.NEW: NEW_OPERATORS,
}


def evaluate_expression(
    numbers: list[int],
    operators: list[Operator],
    arithmetic_family: ArithmeticFamily,
) -> int:
    if len(numbers) != len(operators) + 1:
        raise ValueError("numbers must be exactly one longer than operators")

    if arithmetic_family is ArithmeticFamily.NORMAL:
        expr = render_expression(numbers, operators).replace(" / ", " // ")
        return int(eval(expr))

    # NEW family: left-to-right, no precedence
    value = numbers[0]
    for operator, rhs in zip(operators, numbers[1:], strict=True):
        value = operator.apply(value, rhs)
    return value


def render_expression(numbers: list[int], operators: list[Operator]) -> str:
    tokens: list[str] = [str(numbers[0])]
    for operator, rhs in zip(operators, numbers[1:], strict=True):
        tokens.append(operator)
        tokens.append(str(rhs))
    return " ".join(tokens)


def sample_expression(
    rng: random.Random,
    min_ops: int,
    max_ops: int,
    operator_pool: tuple[Operator, ...],
) -> tuple[list[int], list[Operator]]:
    n_ops = rng.randint(min_ops, max_ops)
    numbers: list[int] = [rng.randint(0, 9)]
    operators: list[Operator] = []

    for _ in range(n_ops):
        operator = rng.choice(operator_pool)
        rhs = rng.randint(1 if operator is Operator.FLOOR_DIVIDE else 0, 9)
        operators.append(operator)
        numbers.append(rhs)

    return numbers, operators


def allocate_by_ratio(
    total: int,
    ratios: dict[DifficultyLevel, float],
    buckets: tuple[DifficultyLevel, ...],
) -> dict[DifficultyLevel, int]:
    if total < 0:
        raise ValueError("total must be non-negative")

    raw: dict[DifficultyLevel, float] = {bucket: total * ratios[bucket] for bucket in buckets}
    counts: dict[DifficultyLevel, int] = {bucket: int(raw[bucket]) for bucket in buckets}
    remainder = total - sum(counts.values())

    for level in sorted(buckets, key=lambda level: raw[level] - counts[level], reverse=True)[:remainder]:
        counts[level] += 1

    return counts


def allocate_family_counts(num_examples: int) -> dict[ArithmeticFamily, int]:
    if num_examples % 2 != 0:
        raise ValueError("num-examples must be even to enforce strict 50/50 families")

    per_family = num_examples // 2
    return {
        ArithmeticFamily.NORMAL: per_family,
        ArithmeticFamily.NEW: per_family,
    }


def allocate_level_counts(num_examples: int) -> dict[DifficultyLevel, int]:
    return allocate_by_ratio(num_examples, DEFAULT_LEVEL_MIX, DIFFICULTY_LEVELS)


@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: str
    arithmetic_family: ArithmeticFamily
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
                "arithmetic_family": self.arithmetic_family,
                "difficulty_level": self.difficulty_level,
                "n_ops": self.n_ops,
                "op_seq": list(self.op_seq),
            },
        }


def generate_random_level(
    *,
    arithmetic_family: ArithmeticFamily,
    difficulty_level: DifficultyLevel,
    sample_prefix: str,
    size: int,
    rng: random.Random,
    seen: set[str],
) -> list[Sample]:
    min_ops, max_ops = LEVEL_BOUNDS[difficulty_level]
    operator_pool = OPERATORS_BY_FAMILY[arithmetic_family]

    samples: list[Sample] = []
    attempts = 0
    max_attempts = max(size * 500, 100_000)

    while len(samples) < size:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(f"generation stalled for {difficulty_level}; requested={size}, got={len(samples)}")

        numbers, operators = sample_expression(
            rng,
            min_ops,
            max_ops,
            operator_pool,
        )
        expression = render_expression(numbers, operators)
        if expression in seen:
            continue

        target = evaluate_expression(numbers, operators, arithmetic_family)

        samples.append(
            Sample(
                sample_id=f"{sample_prefix}_{len(samples):06d}",
                arithmetic_family=arithmetic_family,
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
    counter = Counter(sample.expected_output for sample in samples)
    return {str(label): counter[label] for label in sorted(counter)}


def write_jsonl(path: Path, rows: list[DatasetRow]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True))
            file.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic new-math synthetic benchmark")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260304)
    parser.add_argument("--num-examples", type=int, default=70_300)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.num_examples < 0:
        raise ValueError("num-examples must be non-negative")
    if args.num_examples % 2 != 0:
        raise ValueError("num-examples must be even for strict 50/50 family split")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    counts_by_family = allocate_family_counts(args.num_examples)
    counts_by_family_and_difficulty: dict[ArithmeticFamily, dict[DifficultyLevel, int]] = {
        family: allocate_level_counts(counts_by_family[family]) for family in ARITHMETIC_FAMILIES
    }
    counts_by_difficulty: dict[DifficultyLevel, int] = {
        level: sum(counts_by_family_and_difficulty[family][level] for family in ARITHMETIC_FAMILIES)
        for level in DIFFICULTY_LEVELS
    }

    seen_expressions: set[str] = set()
    rng_shuffle = random.Random(args.seed + 101)

    all_samples: list[Sample] = []
    for family_index, family in enumerate(ARITHMETIC_FAMILIES):
        for level_index, difficulty_level in enumerate(DIFFICULTY_LEVELS):
            bucket_size = counts_by_family_and_difficulty[family][difficulty_level]
            bucket_seed = args.seed + (family_index * 1000) + (level_index * 100) + 11
            rng_bucket = random.Random(bucket_seed)
            sample_prefix = f"{family.value}_{difficulty_level.value.lower()}"
            all_samples.extend(
                generate_random_level(
                    arithmetic_family=family,
                    difficulty_level=difficulty_level,
                    sample_prefix=sample_prefix,
                    size=bucket_size,
                    rng=rng_bucket,
                    seen=seen_expressions,
                )
            )

    rng_shuffle.shuffle(all_samples)

    dataset_path = output_dir / "dataset.jsonl"
    write_jsonl(dataset_path, [sample.to_dict() for sample in all_samples])

    manifest: Manifest = {
        "benchmark_name": "new_math_ops_v2",
        "seed": args.seed,
        "rules": {
            "+": "a+b",
            "-": "a-b",
            "*": "a*b",
            "/": "floor(a/b), b!=0",
            "##": "abs(a-b)",
            "@@": "max(a,b)",
            "$$": "min(a,b)",
            "evaluation": "normal=standard_precedence, new=left_to_right",
        },
        "requested_num_examples": args.num_examples,
        "generated_num_examples": len(all_samples),
        "default_family_mix": DEFAULT_FAMILY_MIX,
        "default_level_mix": DEFAULT_LEVEL_MIX,
        "counts_by_family": counts_by_family,
        "counts_by_difficulty": counts_by_difficulty,
        "counts_by_family_and_difficulty": counts_by_family_and_difficulty,
        "label_histogram": build_label_histogram(all_samples),
        "row_fields": {
            "input": "str",
            "expected_output": "int",
            "metadata": {
                "id": "str",
                "arithmetic_family": "str",
                "difficulty_level": "str",
                "n_ops": "int",
                "op_seq": "list[str]",
            },
        },
        "files": ["dataset.jsonl", "manifest.json"],
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote dataset to {output_dir}")
    print(
        json.dumps(
            {
                "rows": len(all_samples),
                "counts_by_family": counts_by_family,
                "counts_by_difficulty": counts_by_difficulty,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
