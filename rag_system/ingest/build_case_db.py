from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rag_system.nodes.retrieve import embed_image, embed_text
from rag_system.preprocess.a11y_preprocess import OpenAIVLMBackend, preprocess_input

SUPPORTED_INPUTS = {'.pptx', '.pdf'}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_input_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob('*')):
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUTS:
            yield path


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _join_text_parts(record: Dict[str, Any]) -> str:
    check_item = str(record.get('inspection_item') or '').strip()
    error_type = str(record.get('error_type') or '').strip()

    rationale = record.get('rationale_bullets') or []
    improvement_text = '\n'.join(
        str(x).strip() for x in rationale if str(x).strip()
    ).strip()

    improvement_code = str(record.get('fix_code') or '').strip()

    parts = [
        f"check_item: {check_item}",
        f"error_type: {error_type}",
        f"improvement_text: {improvement_text}",
        f"improvement_code: {improvement_code}",
    ]
    return ' | '.join(parts)


def _to_case_items(project_id: str, source_file: Path, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for record in records:
        check_item = (record.get('inspection_item') or '').strip()
        error_type = (record.get('error_type') or '').strip()
        if not check_item or not error_type:
            continue
        rationale = [str(x).strip() for x in (record.get('rationale_bullets') or []) if str(x).strip()]
        improvement_text = '\n'.join(rationale).strip()
        improvement_code = (record.get('fix_code') or '').strip()
        error_code_text = (record.get('error_code_text') or '').strip()
        source_page = int(record.get('page_index') or 0)
        source_text = _join_text_parts(record)
        for image_path in record.get('error_region_images') or []:
            cases.append(
                {
                    'project_id': project_id,
                    'retrieved_image_path': str(Path(image_path).resolve()),
                    'retrieved_result': {
                        'error_type': error_type,
                        'check_item': check_item,
                        'improvement_text': improvement_text,
                        'improvement_code': improvement_code,
                    },
                    'source_meta': {
                        'source_file': str(source_file.resolve()),
                        'page_index': source_page,
                        'error_code_text': error_code_text,
                        'source_text': source_text,
                    },
                }
            )
    return cases


def _build_vector_rows(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases):
        source_meta = case.get('source_meta') or {}
        text_payload = str(source_meta.get('source_text') or '')
        rows.append(
            {
                'id': idx,
                'project_id': case['project_id'],
                'case_image_path': case['retrieved_image_path'],
                'retrieved_result': case['retrieved_result'],
                'source_meta': source_meta,
                'text_payload': text_payload,
                'image_embedding': embed_image(case['retrieved_image_path']).tolist(),
                'text_embedding': embed_text(text_payload).tolist(),
            }
        )
    return rows


def build_case_db(
    project_id: str,
    input_dir: Path,
    case_store_root: Path,
    preprocess_root: Optional[Path] = None,
    model: str = 'gpt-4.1-mini',
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    dpi: int = 200,
) -> Dict[str, Any]:
    input_dir = Path(input_dir)
    case_store_root = Path(case_store_root)
    preprocess_root = Path(preprocess_root) if preprocess_root else (case_store_root / project_id / 'preprocessed')
    project_root = case_store_root / project_id
    ensure_dir(project_root)
    ensure_dir(preprocess_root)

    files = list(_iter_input_files(input_dir))
    if not files:
        raise FileNotFoundError(f'No .pptx or .pdf files found in {input_dir}')

    vlm = OpenAIVLMBackend(model=model, temperature=temperature, api_key=(api_key or os.getenv('OPENAI_API_KEY', '')).strip())

    all_cases: List[Dict[str, Any]] = []
    preprocess_outputs: List[str] = []
    for file_path in files:
        file_out_dir = preprocess_root / file_path.stem
        records_path = preprocess_input(file_path, file_out_dir, vlm, dpi=dpi)
        preprocess_outputs.append(str(records_path))
        records = _read_jsonl(records_path)
        all_cases.extend(_to_case_items(project_id, file_path, records))

    cases_path = project_root / 'cases.json'
    cases_path.write_text(json.dumps(all_cases, ensure_ascii=False, indent=2), encoding='utf-8')

    vector_rows = _build_vector_rows(all_cases)
    vector_path = project_root / 'vector_store.json'
    vector_path.write_text(json.dumps(vector_rows, ensure_ascii=False), encoding='utf-8')

    manifest = {
        'project_id': project_id,
        'input_dir': str(input_dir.resolve()),
        'preprocess_outputs': preprocess_outputs,
        'cases_path': str(cases_path.resolve()),
        'vector_store_path': str(vector_path.resolve()),
        'source_file_count': len(files),
        'case_count': len(all_cases),
        'vector_row_count': len(vector_rows),
    }
    manifest_path = project_root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', required=True)
    parser.add_argument('--input-dir', required=True, help='과거진단이력 pptx/pdf 폴더')
    parser.add_argument('--case-store-root', default=os.getenv('RAG_CASE_STORE_ROOT', './rag_case_store'))
    parser.add_argument('--preprocess-root', default='')
    parser.add_argument('--model', default=os.getenv('VLM_MODEL', 'gpt-4.1-mini'))
    parser.add_argument('--temperature', type=float, default=float(os.getenv('VLM_TEMPERATURE', '0.0')))
    parser.add_argument('--api-key', default='')
    parser.add_argument('--dpi', type=int, default=int(os.getenv('RENDER_DPI', '200')))
    args = parser.parse_args()

    manifest = build_case_db(
        project_id=args.project_id,
        input_dir=Path(args.input_dir).resolve(),
        case_store_root=Path(args.case_store_root).resolve(),
        preprocess_root=Path(args.preprocess_root).resolve() if args.preprocess_root else None,
        model=args.model,
        temperature=args.temperature,
        api_key=args.api_key or os.getenv('OPENAI_API_KEY'),
        dpi=args.dpi,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
