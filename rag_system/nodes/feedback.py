
from __future__ import annotations

from typing import Any

from rag_system.models.schemas import FeedbackType, MemoryEvent, MemorySaveResult, RetrievedResult


def build_memory_event(
    project_id: str,
    query_image_path: str,
    retrieved_image_path: str,
    retrieved_result: RetrievedResult,
    feedback: FeedbackType,
) -> MemoryEvent:
    return {
        'project_id': project_id,
        'query_image_path': query_image_path,
        'retrieved_image_path': retrieved_image_path,
        'retrieved_result': retrieved_result,
        'feedback': feedback,
    }


def build_mem0_memory_text(event: MemoryEvent) -> str:
    result = event['retrieved_result']
    return '\n'.join(
        [
            f"project_id: {event['project_id']}",
            f"query_image_path: {event['query_image_path']}",
            f"retrieved_image_path: {event['retrieved_image_path']}",
            f"check_item: {result['check_item']}",
            f"error_type: {result['error_type']}",
            f"improvement_text: {result['improvement_text']}",
            f"improvement_code: {result['improvement_code']}",
            f"feedback: {event['feedback']}",
        ]
    )


def save_memory_event_with_mem0(memory_client: Any, event: MemoryEvent) -> None:
    memory_text = build_mem0_memory_text(event)
    memory_client.add(
        memory_text,
        user_id=event['project_id'],
        metadata={
            'project_id': event['project_id'],
            'memory_event': event,
        },
    )


def save_long_term_memory(
    memory_client: Any,
    project_id: str,
    query_image_path: str,
    retrieved_image_path: str,
    retrieved_result: RetrievedResult,
    feedback: FeedbackType,
) -> MemorySaveResult:
    event = build_memory_event(
        project_id=project_id,
        query_image_path=query_image_path,
        retrieved_image_path=retrieved_image_path,
        retrieved_result=retrieved_result,
        feedback=feedback,
    )
    save_memory_event_with_mem0(memory_client=memory_client, event=event)
    return {'memory_saved': True, 'memory_event': event}
