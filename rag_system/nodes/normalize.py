from __future__ import annotations

from typing import Optional

from rag_system.models.schemas import NormalizedInput


def normalize_input(
    project_id: str,
    image_path: str,
    user_initial_diagnosis: Optional[str] = None,
) -> NormalizedInput:
    return NormalizedInput(
        project_id=project_id,
        image_path=image_path,
        user_initial_diagnosis=user_initial_diagnosis,
    )
