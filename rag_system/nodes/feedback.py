from __future__ import annotations

import hashlib
from typing import Optional

from rag_system.memory.langgraph_postgres import LangGraphPostgresMemoryClient
from rag_system.models.schemas import (
    DiagnosisRecord,
    FeedbackType,
    InitialDiagnosisResult,
    LongTermMemoryValue,
    MemorySaveResult,
    RetrievedResult,
)


def _stable_case_id(project_id: str, diagnosis: DiagnosisRecord) -> str:
    raw = "||".join(
        [
            project_id,
            diagnosis['error_type'],
            diagnosis['check_item'],
            diagnosis['improvement_text'],
            diagnosis['improvement_code'],
            diagnosis['image_path'],
        ]
    )
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]


def build_diagnosis_record(
    *,
    diagnosis_result: InitialDiagnosisResult | RetrievedResult,
    image_path: str,
) -> DiagnosisRecord:
    return {
        'error_type': diagnosis_result['error_type'],
        'check_item': diagnosis_result['check_item'],
        'improvement_text': diagnosis_result['improvement_text'],
        'improvement_code': diagnosis_result['improvement_code'],
        'image_path': image_path,
    }


def build_memory_value(
    *,
    current_diagnosis_result: InitialDiagnosisResult,
    current_image_path: str,
    referenced_diagnosis_result: RetrievedResult,
    referenced_image_path: str,
) -> LongTermMemoryValue:
    return {
        'current_diagnosis': build_diagnosis_record(
            diagnosis_result=current_diagnosis_result,
            image_path=current_image_path,
        ),
        'referenced_diagnosis': build_diagnosis_record(
            diagnosis_result=referenced_diagnosis_result,
            image_path=referenced_image_path,
        ),
    }


def build_memory_key(project_id: str, memory_value: LongTermMemoryValue) -> str:
    past_case_id = _stable_case_id(project_id, memory_value['referenced_diagnosis'])
    current_case_id = _stable_case_id(project_id, memory_value['current_diagnosis'])
    return f'{past_case_id}:{current_case_id}'


def save_long_term_memory(
    memory_client: LangGraphPostgresMemoryClient,
    *,
    project_id: str,
    current_diagnosis_result: InitialDiagnosisResult,
    current_image_path: str,
    referenced_diagnosis_result: RetrievedResult,
    referenced_image_path: str,
    feedback: FeedbackType,
) -> MemorySaveResult:
    namespace = memory_client.namespace(project_id)
    if feedback != 'thumbs_up':
        return {
            'memory_saved': False,
            'namespace': namespace,
            'memory_key': None,
            'memory_value': None,
        }

    memory_value = build_memory_value(
        current_diagnosis_result=current_diagnosis_result,
        current_image_path=current_image_path,
        referenced_diagnosis_result=referenced_diagnosis_result,
        referenced_image_path=referenced_image_path,
    )
    memory_key = build_memory_key(project_id, memory_value)
    memory_client.save_mapping(
        project_id=project_id,
        memory_key=memory_key,
        memory_value=memory_value,
    )
    return {
        'memory_saved': True,
        'namespace': namespace,
        'memory_key': memory_key,
        'memory_value': memory_value,
    }
