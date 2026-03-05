from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Literal, Protocol, TypedDict


class Operator(str, Enum):
    ABS_DIFF = "##"
    MAX = "@@"
    MIN = "$$"


class DifficultyLevel(str, Enum):
    S1_PRIMITIVE = "S1_Primitive"
    S2_COMPOSITION = "S2_Composition"
    S3_LENGTH_OOD = "S3_LengthOOD"


ChatRole = Literal["system", "user"]

OPERATORS: tuple[Operator, ...] = (Operator.ABS_DIFF, Operator.MAX, Operator.MIN)
LEVEL_S1: DifficultyLevel = DifficultyLevel.S1_PRIMITIVE
LEVEL_S2: DifficultyLevel = DifficultyLevel.S2_COMPOSITION
LEVEL_S3: DifficultyLevel = DifficultyLevel.S3_LENGTH_OOD
DIFFICULTY_LEVELS: tuple[DifficultyLevel, ...] = (LEVEL_S1, LEVEL_S2, LEVEL_S3)


# NOTE: Intentional duplication: keep flat manifest rule keys close to Operator values.
# Approved for now; FIXME: if this pattern appears in more than two places, refactor
# to a shared serializer/adapter boundary.
Rules = TypedDict(
    "Rules",
    {
        "##": str,
        "@@": str,
        "$$": str,
        "evaluation": str,
        "digits": str,
    },
)


class ChatMessage(TypedDict):
    role: ChatRole
    content: str


class DatasetMetadata(TypedDict):
    id: str
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
    by_difficulty: dict[DifficultyLevel, BucketMetrics]
    by_n_ops: dict[int, BucketMetrics]


class ManifestMetadataFields(TypedDict):
    id: str
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
    default_level_mix: dict[DifficultyLevel, float]
    counts_by_difficulty: dict[DifficultyLevel, int]
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
