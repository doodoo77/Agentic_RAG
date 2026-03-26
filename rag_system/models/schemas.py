from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, field_validator
from typing_extensions import Annotated


ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
FeedbackType = Literal['thumbs_up', 'thumbs_down']


class NormalizedInput(BaseModel):
    project_id: str
    image_path: str
    user_initial_diagnosis: Optional[str] = None

    @field_validator('project_id')
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('project_id is required.')
        return value

    @field_validator('image_path')
    @classmethod
    def validate_image_path(cls, value: str) -> str:
        path = Path(value)
        if not path.exists():
            raise ValueError(f'image file does not exist: {value}')
        if not path.is_file():
            raise ValueError(f'image path is not a file: {value}')
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f'unsupported image extension: {path.suffix}. Allowed: {sorted(ALLOWED_EXTENSIONS)}'
            )
        return str(path)

    @field_validator('user_initial_diagnosis')
    @classmethod
    def normalize_user_initial_diagnosis(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        return value


class RetrievedResult(TypedDict):
    error_type: str
    check_item: str
    improvement_text: str
    improvement_code: str


class InitialDiagnosisResult(TypedDict):
    error_type: str
    check_item: str
    improvement_text: str
    improvement_code: str


class DiagnosisRecord(TypedDict):
    error_type: str
    check_item: str
    improvement_text: str
    improvement_code: str
    image_path: str


class LongTermMemoryValue(TypedDict):
    current_diagnosis: DiagnosisRecord
    referenced_diagnosis: DiagnosisRecord


class LongTermMemorySearchItem(TypedDict):
    memory_key: str
    memory_value: LongTermMemoryValue
    text_similarity: float
    image_similarity: float
    final_score: float


class EarlyExitResult(TypedDict):
    early_exit_triggered: bool
    selected_memory: Optional[LongTermMemorySearchItem]
    selected_similarity: Optional[float]
    diagnosis_result: Optional[RetrievedResult]


class CaseItem(TypedDict, total=False):
    project_id: str
    retrieved_image_path: str
    retrieved_result: RetrievedResult
    source_meta: Dict[str, Any]


class RetrievalCandidate(TypedDict):
    case_image_path: str
    final_score: float
    image_similarity: float
    text_similarity: float
    image_weight: float
    text_weight: float
    retrieved_result: RetrievedResult


class GradedCandidate(TypedDict):
    candidate: RetrievalCandidate
    is_relevant: bool
    grader_score: float
    reason: str


class GradeResult(TypedDict):
    has_relevant_candidate: bool
    selected_candidate: Optional[RetrievalCandidate]
    graded_candidates: List[GradedCandidate]
    grade_reason: str


class RewriteResult(TypedDict):
    rewritten_query: str


class MemorySaveResult(TypedDict):
    memory_saved: bool
    namespace: Tuple[str, str]
    memory_key: Optional[str]
    memory_value: Optional[LongTermMemoryValue]


class PipelineState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # request / raw input
    project_id: str
    image_path: str
    user_initial_diagnosis: Optional[str]
    feedback: Optional[FeedbackType]
    allowed_pairs_xlsx_path: Optional[str]
    allowed_pairs_sheet_name: Optional[str]

    # step 1
    normalized_input: Dict[str, Any]

    # step 2
    allowed_pairs: List[Tuple[str, str]]
    allowed_pairs_text: str
    initial_diagnosis_prompt: str
    initial_diagnosis_result: InitialDiagnosisResult
    initial_diagnosis_pair_valid: bool

    # memory payload
    retrieved_image_path: Optional[str]
    retrieved_result: Optional[RetrievedResult]
    memory_record: Optional[MemorySaveResult]
    memory_saved: Optional[bool]

    # memory early exit
    memory_candidates: List[LongTermMemorySearchItem]
    early_exit_threshold: float
    early_exit_result: EarlyExitResult
    early_exit_triggered: bool
    selected_memory: Optional[LongTermMemorySearchItem]
    selected_similarity: Optional[float]
    diagnosis_result: Optional[RetrievedResult]

    # retrieval
    top_k: int
    retrieval_candidates: List[RetrievalCandidate]
    retrieval_query_text: str
    retrieval_mode: str
    rewritten_query: Optional[str]

    # grader
    graded_candidates: List[GradedCandidate]
    has_relevant_candidate: bool
    selected_candidate: Optional[RetrievalCandidate]
    grade_reason: Optional[str]

    # control
    rewrite_count: int
    max_rewrite_count: int
    final_output: Optional[Dict[str, Any]]
