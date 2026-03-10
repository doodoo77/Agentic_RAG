
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from rag_system.models.schemas import InitialDiagnosisResult


def _load_allowed_pairs_from_excel(xlsx_path: str, sheet_name: str) -> List[Tuple[str, str]]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if '지침' not in df.columns or '오류 유형' not in df.columns:
        raise RuntimeError(f'golden_text.xlsx columns mismatch: {list(df.columns)}')

    seen: Set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for _, row in df[['지침', '오류 유형']].iterrows():
        ci = str(row['지침']).strip()
        et = str(row['오류 유형']).strip()
        if not ci or ci.lower() == 'nan' or not et or et.lower() == 'nan':
            continue
        key = (ci, et)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)

    if not out:
        raise ValueError(f'No allowed pairs found in Excel: path={xlsx_path}, sheet={sheet_name}')
    return out


def resolve_allowed_pairs(
    allowed_pairs: List[Tuple[str, str]] | None = None,
    xlsx_path: str | None = None,
    sheet_name: str = 'Sheet1',
) -> List[Tuple[str, str]]:
    if allowed_pairs:
        return allowed_pairs
    if not xlsx_path:
        raise ValueError('Either allowed_pairs or allowed_pairs_xlsx_path must be provided.')
    return _load_allowed_pairs_from_excel(xlsx_path=xlsx_path, sheet_name=sheet_name)


def _format_allowed_pairs(pairs: List[Tuple[str, str]]) -> str:
    if not pairs:
        return '(no allowed pairs)'
    return '\n'.join([f'- check_item: {ci} | error_type: {et}' for (ci, et) in pairs])


def _pair_set(pairs: List[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    return set(pairs)


def _validate_or_fallback(
    parsed: Dict[str, Any],
    allowed: Set[Tuple[str, str]],
    fallback_pair: Tuple[str, str],
) -> Tuple[Dict[str, Any], bool]:
    ci = str(parsed.get('check_item') or '')
    et = str(parsed.get('error_type') or '')
    ok = (ci, et) in allowed
    if ok:
        return parsed, True

    parsed['check_item'] = fallback_pair[0]
    parsed['error_type'] = fallback_pair[1]
    return parsed, False


def build_initial_diagnosis_prompt(
    user_initial_diagnosis: str | None,
    allowed_pairs_text: str,
) -> str:
    user_text = (user_initial_diagnosis or '').strip()
    if not user_text:
        user_text = '(사용자 초기 진단 없음)'

    return f"""
[User Initial Diagnosis]
{user_text}

[Allowed pairs (check_item / error_type는 반드시 아래 조합 중 하나만 선택)]
{allowed_pairs_text}

요구사항:
0) check_item과 error_type은 반드시 위 Allowed pairs 안에 있는 조합 그대로 선택한다.
새 값을 만들거나, 다른 조합으로 바꾸거나, 추론해서 생성하면 안 된다.

1) 이미지를 최우선으로 보고 현재 화면의 문제를 초기 진단하라.
2) 사용자 초기 진단 텍스트는 보조 근거로만 활용하라.
3) improvement_text에는
   - 무엇이 문제인지
   - 왜 문제인지
   - 어떻게 고칠지를
   2~3문장, 300자 이내로 작성하라.
4) improvement_code에는 개선 방향의 HTML 예시 코드를 작성하라.
5) 반드시 JSON만 출력하라.

반드시 아래 형식으로만 출력:
{{
  "error_type": "...",
  "check_item": "...",
  "improvement_text": "...",
  "improvement_code": "..."
}}
""".strip()


def invoke_multimodal_json(llm: Any, image_path: str, prompt: str) -> Dict[str, Any]:
    return llm.invoke_json(prompt=prompt, image_paths=[image_path])


def run_initial_diagnosis(
    llm: Any,
    image_path: str,
    user_initial_diagnosis: str | None,
    allowed_pairs: List[Tuple[str, str]],
) -> tuple[InitialDiagnosisResult, bool, str]:
    if not allowed_pairs:
        raise ValueError('allowed_pairs must not be empty. Check your Excel file and sheet name.')

    allowed_pairs_text = _format_allowed_pairs(allowed_pairs)
    prompt = build_initial_diagnosis_prompt(user_initial_diagnosis, allowed_pairs_text)
    parsed = invoke_multimodal_json(llm=llm, image_path=image_path, prompt=prompt)

    allowed = _pair_set(allowed_pairs)
    fallback_pair = allowed_pairs[0]
    parsed, is_valid = _validate_or_fallback(parsed=parsed, allowed=allowed, fallback_pair=fallback_pair)
    return parsed, is_valid, prompt
