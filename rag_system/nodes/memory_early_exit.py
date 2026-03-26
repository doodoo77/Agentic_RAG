from __future__ import annotations

from pathlib import Path
from typing import Optional

from rag_system.memory.langgraph_postgres import LangGraphPostgresMemoryClient
from rag_system.models.schemas import (
    DiagnosisRecord,
    EarlyExitResult,
    InitialDiagnosisResult,
    LongTermMemorySearchItem,
    RetrievedResult,
)
from rag_system.nodes.feedback import build_diagnosis_record
from rag_system.nodes.retrieve import cosine_similarity_torch, embed_image, embed_text

TEXT_WEIGHT = 0.4
IMAGE_WEIGHT = 0.6


def build_text_payload(diagnosis: DiagnosisRecord) -> str:
    return ' | '.join(
        [
            f"error_type: {diagnosis['error_type']}",
            f"check_item: {diagnosis['check_item']}",
            f"improvement_text: {diagnosis['improvement_text']}",
        ]
    )


def compute_text_similarity(current_diagnosis: DiagnosisRecord, memory_diagnosis: DiagnosisRecord) -> float:
    current_vec = embed_text(build_text_payload(current_diagnosis))
    memory_vec = embed_text(build_text_payload(memory_diagnosis))
    similarity = cosine_similarity_torch(current_vec, memory_vec)
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def compute_image_similarity(current_image_path: str, memory_image_path: str) -> float:
    if not Path(current_image_path).exists() or not Path(memory_image_path).exists():
        return 0.0
    current_vec = embed_image(current_image_path)
    memory_vec = embed_image(memory_image_path)
    similarity = cosine_similarity_torch(current_vec, memory_vec)
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def compute_final_similarity(current_diagnosis: DiagnosisRecord, memory_diagnosis: DiagnosisRecord) -> tuple[float, float, float]:
    text_similarity = compute_text_similarity(current_diagnosis, memory_diagnosis)
    image_similarity = compute_image_similarity(
        current_diagnosis['image_path'],
        memory_diagnosis['image_path'],
    )
    final_score = (TEXT_WEIGHT * text_similarity) + (IMAGE_WEIGHT * image_similarity)
    return text_similarity, image_similarity, final_score


def _to_retrieved_result(diagnosis: DiagnosisRecord) -> RetrievedResult:
    return {
        'error_type': diagnosis['error_type'],
        'check_item': diagnosis['check_item'],
        'improvement_text': diagnosis['improvement_text'],
        'improvement_code': diagnosis['improvement_code'],
    }


def search_memory_candidates(
    memory_client: LangGraphPostgresMemoryClient,
    project_id: str,
    current_diagnosis: DiagnosisRecord,
    top_k: int = 10,
) -> list[LongTermMemorySearchItem]:
    rows = memory_client.list_project_memories(project_id, limit=max(top_k * 20, top_k))

    candidates: list[LongTermMemorySearchItem] = []
    for row in rows:
        memory_value = row.get('value') or {}
        memory_current = memory_value.get('current_diagnosis')
        memory_reference = memory_value.get('referenced_diagnosis')
        if not isinstance(memory_current, dict) or not isinstance(memory_reference, dict):
            continue
        if not memory_current.get('image_path') or not memory_reference.get('image_path'):
            continue

        text_similarity, image_similarity, final_score = compute_final_similarity(
            current_diagnosis=current_diagnosis,
            memory_diagnosis=memory_current,
        )
        candidates.append(
            {
                'memory_key': row.get('key', ''),
                'memory_value': memory_value,
                'text_similarity': text_similarity,
                'image_similarity': image_similarity,
                'final_score': final_score,
            }
        )

    candidates.sort(key=lambda item: item['final_score'], reverse=True)
    return candidates[:top_k]


def run_memory_retrieval_early_exit(
    memory_client: LangGraphPostgresMemoryClient,
    project_id: str,
    image_path: str,
    user_initial_diagnosis: Optional[str],
    initial_diagnosis_result: InitialDiagnosisResult,
    early_exit_threshold: float = 0.90,
    top_k: int = 10,
) -> tuple[EarlyExitResult, list[LongTermMemorySearchItem]]:
    del user_initial_diagnosis

    current_diagnosis = build_diagnosis_record(
        diagnosis_result=initial_diagnosis_result,
        image_path=image_path,
    )
    memory_items = search_memory_candidates(
        memory_client=memory_client,
        project_id=project_id,
        current_diagnosis=current_diagnosis,
        top_k=top_k,
    )

    selected = next((item for item in memory_items if item['final_score'] >= early_exit_threshold), None)
    if selected is None:
        return {
            'early_exit_triggered': False,
            'selected_memory': None,
            'selected_similarity': None,
            'diagnosis_result': None,
        }, memory_items

    referenced_diagnosis = selected['memory_value']['referenced_diagnosis']
    return {
        'early_exit_triggered': True,
        'selected_memory': selected,
        'selected_similarity': selected['final_score'],
        'diagnosis_result': _to_retrieved_result(referenced_diagnosis),
    }, memory_items
