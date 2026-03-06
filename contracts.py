from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Literal, Protocol, TypedDict


class ArithmeticFamily(str, Enum):
    NORMAL = "normal"
    NEW = "new"


class Operator(str, Enum):
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    FLOOR_DIVIDE = "/"
    ABS_DIFF = "##"
    MAX = "@@"
    MIN = "$$"


class DifficultyLevel(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


ChatRole = Literal["system", "user"]

ARITHMETIC_FAMILIES: tuple[ArithmeticFamily, ...] = (
    ArithmeticFamily.NORMAL,
    ArithmeticFamily.NEW,
)

NORMAL_OPERATORS: tuple[Operator, ...] = (
    Operator.ADD,
    Operator.SUBTRACT,
    Operator.MULTIPLY,
    Operator.FLOOR_DIVIDE,
)
NEW_OPERATORS: tuple[Operator, ...] = (
    Operator.ABS_DIFF,
    Operator.MAX,
    Operator.MIN,
)
OPERATORS: tuple[Operator, ...] = (*NORMAL_OPERATORS, *NEW_OPERATORS)

DIFFICULTY_LEVELS: tuple[DifficultyLevel, ...] = (
    DifficultyLevel.L1,
    DifficultyLevel.L2,
    DifficultyLevel.L3,
)


# NOTE: Intentional duplication: keep flat manifest rule keys close to Operator values.
# Approved for now; FIXME: if this pattern appears in more than two places, refactor
# to a shared serializer/adapter boundary.
Rules = TypedDict(
    "Rules",
    {
        "+": str,
        "-": str,
        "*": str,
        "/": str,
        "##": str,
        "@@": str,
        "$$": str,
        "evaluation": str,
    },
)


class ChatMessage(TypedDict):
    role: ChatRole
    content: str


class DatasetMetadata(TypedDict):
    id: str
    arithmetic_family: ArithmeticFamily
    difficulty_level: DifficultyLevel
    n_ops: int
    op_seq: list[Operator]


class DatasetRow(TypedDict):
    input: str
    expected_output: int
    metadata: DatasetMetadata


class PredictionRow(TypedDict):
    id: str
    input: str
    expected_output: int
    predicted_output: int | None
    is_correct: bool
    format_error: bool
    raw_response: str
    arithmetic_family: ArithmeticFamily
    difficulty_level: DifficultyLevel
    n_ops: int
    latency_seconds: float


class BucketCounts(TypedDict):
    total: int
    correct: int
    format_errors: int


class BucketMetrics(TypedDict):
    total: int
    correct: int
    format_errors: int
    accuracy: float
    format_error_rate: float


class Metrics(TypedDict):
    total: int
    correct: int
    accuracy: float
    format_errors: int
    format_error_rate: float
    by_family: dict[ArithmeticFamily, BucketMetrics]
    by_difficulty: dict[DifficultyLevel, BucketMetrics]
    by_n_ops: dict[int, BucketMetrics]


class ManifestMetadataFields(TypedDict):
    id: str
    arithmetic_family: str
    difficulty_level: str
    n_ops: str
    op_seq: str


class ManifestRowFields(TypedDict):
    input: str
    expected_output: str
    metadata: ManifestMetadataFields


class Manifest(TypedDict):
    benchmark_name: str
    seed: int
    rules: Rules
    requested_num_examples: int
    generated_num_examples: int
    default_family_mix: dict[ArithmeticFamily, float]
    default_level_mix: dict[DifficultyLevel, float]
    counts_by_family: dict[ArithmeticFamily, int]
    counts_by_difficulty: dict[DifficultyLevel, int]
    counts_by_family_and_difficulty: dict[ArithmeticFamily, dict[DifficultyLevel, int]]
    label_histogram: dict[str, int]
    row_fields: ManifestRowFields
    files: list[str]


class RunConfig(TypedDict):
    model: str
    temperature: float
    max_tokens: int
    concurrency: int
    dataset: str
    manifest: str | None
    limit: int | None
    prompt_version: str


class LLMCompletionClient(Protocol):
    async def complete(self, messages: Sequence[ChatMessage]) -> str: ...
