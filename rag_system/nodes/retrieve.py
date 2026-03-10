from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, List, Optional

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from rag_system.models.schemas import InitialDiagnosisResult, RetrievalCandidate, RetrievedResult

_CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
_TEXT_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
_clip_processor = None
_clip_model = None
_clip_device = None
_text_model = None

def _resolve_case_image_path(case_image_path: str, project_id: str) -> Path:
    raw_path = Path(case_image_path)
    if raw_path.exists():
        return raw_path

    marker = f"rag_case_store/{project_id}/"
    normalized = case_image_path.replace("\\", "/")
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidate = Path("./rag_case_store") / project_id / suffix
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Case image does not exist: {case_image_path}")

def _get_clip_components():
    global _clip_processor, _clip_model, _clip_device
    if _clip_processor is None or _clip_model is None:
        _clip_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME).to(_clip_device)
        _clip_model.eval()
    return _clip_processor, _clip_model, _clip_device


def _get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(_TEXT_MODEL_NAME)
    return _text_model


def _load_rgb_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert('RGB')


def _normalize_tensor(vec: torch.Tensor) -> torch.Tensor:
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    denom = vec.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    vec = vec / denom
    return vec[0].detach().cpu()


def _extract_image_embedding(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return _normalize_tensor(output)
    if hasattr(output, 'image_embeds') and output.image_embeds is not None:
        return _normalize_tensor(output.image_embeds)
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return _normalize_tensor(output.pooler_output)
    if hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
        pooled = output.last_hidden_state.mean(dim=1)
        return _normalize_tensor(pooled)
    raise TypeError(f'Unsupported image feature output type: {type(output)}')


def embed_image(image_path: str) -> torch.Tensor:
    processor, model, device = _get_clip_components()
    image = _load_rgb_image(image_path)
    inputs = processor(images=image, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        try:
            output = model.get_image_features(**inputs)
        except Exception:
            output = model.vision_model(pixel_values=inputs['pixel_values'])
    return _extract_image_embedding(output)


def embed_text(text: str) -> torch.Tensor:
    model = _get_text_model()
    vec = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    return vec.detach().cpu()


def cosine_similarity_torch(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    return float(torch.dot(vec_a, vec_b).item())


def build_query_text(user_initial_diagnosis: Optional[str], initial_diagnosis_result: InitialDiagnosisResult) -> str:
    parts = [
        f"check_item: {initial_diagnosis_result['check_item']}",
        f"error_type: {initial_diagnosis_result['error_type']}",
        f"improvement_text: {initial_diagnosis_result['improvement_text']}",
    ]
    user_text = (user_initial_diagnosis or '').strip()
    if user_text:
        parts.append(f'user_initial_diagnosis: {user_text}')
    return ' | '.join(parts)


def build_case_text(retrieved_result: RetrievedResult, extra_text: str = '') -> str:
    parts = [
        f"check_item: {retrieved_result['check_item']}",
        f"error_type: {retrieved_result['error_type']}",
        f"improvement_text: {retrieved_result['improvement_text']}",
    ]
    if extra_text.strip():
        parts.append(extra_text.strip())
    return ' | '.join(parts)


def softmax_channel_weights(image_similarity: float, text_similarity: float) -> tuple[float, float]:
    exp_img = math.exp(image_similarity*2.0)
    exp_txt = math.exp(text_similarity)
    denom = exp_img + exp_txt
    return exp_img / denom, exp_txt / denom


def compute_weighted_similarity(image_similarity: float, text_similarity: float) -> dict:
    image_weight, text_weight = softmax_channel_weights(image_similarity, text_similarity)
    final_score = image_weight * image_similarity + text_weight * text_similarity
    return {
        'image_similarity': image_similarity,
        'text_similarity': text_similarity,
        'image_weight': image_weight,
        'text_weight': text_weight,
        'final_score': final_score,
    }


def _case_store_root() -> Path:
    return Path(os.getenv('RAG_CASE_STORE_ROOT', './rag_case_store'))


def _vector_store_path(project_id: str) -> Path:
    return _case_store_root() / project_id / 'vector_store.json'


def _tensor_from_list(values: list[float]) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float32)
    return _normalize_tensor(tensor)


def load_project_vector_store(project_id: str) -> list[dict]:
    vector_path = _vector_store_path(project_id)
    if not vector_path.exists():
        raise FileNotFoundError(
            f'vector_store.json not found: {vector_path}. 먼저 build_case_db()를 실행해서 과거진단이력 벡터 DB를 생성하세요.'
        )

    raw = json.loads(vector_path.read_text(encoding='utf-8'))
    if not isinstance(raw, list):
        raise ValueError(f'vector_store.json must contain a list: {vector_path}')

    rows: list[dict] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f'vector_store row must be object: index={idx}')

        case_image_path = str(row.get('case_image_path') or '')
        retrieved_result = row.get('retrieved_result')
        image_embedding = row.get('image_embedding')
        text_embedding = row.get('text_embedding')
        if not case_image_path or not isinstance(retrieved_result, dict):
            raise ValueError(f'Invalid vector_store row index={idx}: missing case_image_path/retrieved_result')
        if not isinstance(image_embedding, list) or not isinstance(text_embedding, list):
            raise ValueError(f'Invalid vector_store row index={idx}: missing embeddings')

        image_path = _resolve_case_image_path(case_image_path, project_id)

        rows.append(
            {
                'id': row.get('id', idx),
                'project_id': str(row.get('project_id') or project_id),
                'case_image_path': str(image_path),
                'retrieved_result': {
                    'error_type': str(retrieved_result['error_type']),
                    'check_item': str(retrieved_result['check_item']),
                    'improvement_text': str(retrieved_result['improvement_text']),
                    'improvement_code': str(retrieved_result['improvement_code']),
                },
                'source_meta': row.get('source_meta') if isinstance(row.get('source_meta'), dict) else {},
                'text_payload': str(row.get('text_payload') or ''),
                'image_embedding': _tensor_from_list([float(x) for x in image_embedding]),
                'text_embedding': _tensor_from_list([float(x) for x in text_embedding]),
            }
        )
    return rows


def retrieve_cases_weighted_fusion(
    project_id: str,
    image_path: str,
    user_initial_diagnosis: Optional[str],
    initial_diagnosis_result: InitialDiagnosisResult,
    rewritten_query: Optional[str] = None,
    top_k: int = 5,
) -> tuple[list[RetrievalCandidate], str]:
    query_text = rewritten_query or build_query_text(user_initial_diagnosis, initial_diagnosis_result)
    query_image_vec = embed_image(image_path)
    query_text_vec = embed_text(query_text)

    vector_rows = load_project_vector_store(project_id)
    candidates: list[RetrievalCandidate] = []

    for row in vector_rows:
        image_similarity = cosine_similarity_torch(query_image_vec, row['image_embedding'])
        text_similarity = cosine_similarity_torch(query_text_vec, row['text_embedding'])
        score_info = compute_weighted_similarity(image_similarity, text_similarity)
        candidates.append(
            {
                'case_image_path': row['case_image_path'],
                'image_similarity': score_info['image_similarity'],
                'text_similarity': score_info['text_similarity'],
                'image_weight': score_info['image_weight'],
                'text_weight': score_info['text_weight'],
                'final_score': score_info['final_score'],
                'retrieved_result': row['retrieved_result'],
            }
        )

    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return candidates[:top_k], query_text
