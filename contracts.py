from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, TypedDict


Operator = Literal["##", "@@", "$$"]
DifficultyLevel = Literal["S1_Primitive", "S2_Composition", "S3_LengthOOD"]
ChatRole = Literal["system", "user"]


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


class SupportsComplete(Protocol):
    async def complete(self, messages: Sequence[ChatMessage]) -> str: ...
