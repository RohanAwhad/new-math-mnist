from __future__ import annotations

from new_math_ops.evaluate import (
    FINAL_ANSWER_PATTERN,
    build_parser,
    compute_metrics,
    evaluate_dataset_rows,
    load_dataset_rows,
    main,
    parse_final_answer,
    write_run_artifacts,
)

__all__ = [
    "FINAL_ANSWER_PATTERN",
    "parse_final_answer",
    "load_dataset_rows",
    "compute_metrics",
    "evaluate_dataset_rows",
    "write_run_artifacts",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    main()
