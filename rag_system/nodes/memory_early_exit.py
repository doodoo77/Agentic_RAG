from __future__ import annotations

from typing import Any, Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from rag_system.models.schemas import EarlyExitResult, InitialDiagnosisResult, Mem0SearchItem, MemoryEvent

_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_clip_processor = None
_clip_model = None
_clip_device = None


def _get_clip_components():
    global _clip_processor, _clip_model, _clip_device
    if _clip_processor is None or _clip_model is None:
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME).to(_clip_device)
        _clip_model.eval()
    return _clip_processor, _clip_model, _clip_device


def _load_rgb_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _normalize_tensor(vec: torch.Tensor) -> torch.Tensor:
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    denom = vec.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    vec = vec / denom
    return vec[0].detach().cpu()


def _extract_image_embedding(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return _normalize_tensor(output)

    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return _normalize_tensor(output.image_embeds)

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return _normalize_tensor(output.pooler_output)

    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        pooled = output.last_hidden_state.mean(dim=1)
        return _normalize_tensor(pooled)

    raise TypeError(f"Unsupported image feature output type: {type(output)}")


def _embed_image(image_path: str) -> torch.Tensor:
    processor, model, device = _get_clip_components()
    image = _load_rgb_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        try:
            output = model.get_image_features(**inputs)
        except Exception:
            output = model.vision_model(pixel_values=inputs["pixel_values"])

    return _extract_image_embedding(output)


def compute_image_similarity(current_image_path: str, memory_query_image_path: str) -> float:
    current_vec = _embed_image(current_image_path)
    memory_vec = _embed_image(memory_query_image_path)
    similarity = torch.dot(current_vec, memory_vec).item()
    similarity_01 = (similarity + 1.0) / 2.0
    return max(0.0, min(1.0, float(similarity_01)))


def build_mem0_search_query(
    user_initial_diagnosis: Optional[str],
    initial_diagnosis_result: InitialDiagnosisResult,
) -> str:
    parts = [
        f"check_item: {initial_diagnosis_result['check_item']}",
        f"error_type: {initial_diagnosis_result['error_type']}",
        f"improvement_text: {initial_diagnosis_result['improvement_text']}",
    ]
    user_text = (user_initial_diagnosis or "").strip()
    if user_text:
        parts.append(f"user_initial_diagnosis: {user_text}")
    return " | ".join(parts)


def _normalize_search_results(results: Any) -> list[dict]:
    if isinstance(results, dict) and "results" in results:
        return results["results"]
    if isinstance(results, list):
        return results
    return []


def search_memory_candidates_with_mem0(
    memory_client: Any,
    project_id: str,
    user_initial_diagnosis: Optional[str],
    initial_diagnosis_result: InitialDiagnosisResult,
    top_k: int = 10,
) -> list[Mem0SearchItem]:
    query = build_mem0_search_query(user_initial_diagnosis, initial_diagnosis_result)
    results = memory_client.search(query, user_id=project_id, limit=top_k)
    results = _normalize_search_results(results)

    out: list[Mem0SearchItem] = []
    for item in results:
        metadata = item.get("metadata", {}) or {}
        if metadata.get("project_id") != project_id:
            continue
        event = metadata.get("memory_event")
        if not event:
            continue
        out.append({"memory_event": event, "mem0_score": item.get("score")})
    return out


def score_early_exit_candidate(similarity: float, feedback: str) -> float:
    if feedback == "thumbs_down":
        return -1.0
    if feedback == "thumbs_up":
        return similarity + 0.03
    return similarity


def select_early_exit_candidate(
    image_path: str,
    memory_items: list[Mem0SearchItem],
    early_exit_threshold: float = 0.90,
):
    candidates = []
    for item in memory_items:
        event: MemoryEvent = item["memory_event"]
        similarity = compute_image_similarity(image_path, event["query_image_path"])
        if similarity < early_exit_threshold:
            continue
        final_score = score_early_exit_candidate(similarity, event["feedback"])
        if final_score < 0:
            continue
        candidates.append(
            {
                "memory_event": event,
                "similarity": similarity,
                "final_score": final_score,
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    best = candidates[0]
    event: MemoryEvent = best["memory_event"]

    return {
        "matched": True,
        "similarity": best["similarity"],
        "feedback": event["feedback"],
        "result": event.get("final_diagnosis_result", event["retrieved_result"]),
        "memory_event": event,
    }


def run_memory_retrieval_early_exit(
    memory_client: Any,
    project_id: str,
    image_path: str,
    user_initial_diagnosis: Optional[str],
    initial_diagnosis_result: InitialDiagnosisResult,
    early_exit_threshold: float = 0.90,
    top_k: int = 10,
) -> tuple[EarlyExitResult, list[Mem0SearchItem]]:
    memory_items = search_memory_candidates_with_mem0(
        memory_client=memory_client,
        project_id=project_id,
        user_initial_diagnosis=user_initial_diagnosis,
        initial_diagnosis_result=initial_diagnosis_result,
        top_k=top_k,
    )

    selected = select_early_exit_candidate(
        image_path=image_path,
        memory_items=memory_items,
        early_exit_threshold=early_exit_threshold,
    )

    if selected is None:
        return {
            "early_exit_triggered": False,
            "selected_memory": None,
            "selected_similarity": None,
            "diagnosis_result": None,
        }, memory_items

    return {
        "early_exit_triggered": True,
        "selected_memory": selected["memory_event"],
        "selected_similarity": selected["similarity"],
        "diagnosis_result": selected["memory_event"]["retrieved_result"],
    }, memory_items

