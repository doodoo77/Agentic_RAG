
from __future__ import annotations

from typing import Any


def build_query_rewrite_prompt(
    user_initial_diagnosis: str | None,
    initial_diagnosis_result: dict,
    graded_candidates: list[dict],
) -> str:
    user_text = (user_initial_diagnosis or '').strip()
    if not user_text:
        user_text = '(사용자 초기 진단 없음)'

    top_candidates = graded_candidates[:3]

    return f"""
[Current User Initial Diagnosis]
{user_text}

[Current Initial Diagnosis Result]
{initial_diagnosis_result}

[Graded Candidates]
{top_candidates}

너의 역할:
현재 입력에 대해 retrieval을 다시 수행할 수 있도록,
검색 친화적인 rewritten query를 작성하라.

목표:
- 현재 사용자가 문제로 삼는 오류 영역을 더 잘 찾을 수 있는 질의를 만든다.
- 기존 retrieval에서 관련성이 낮았던 후보들을 반복해서 찾지 않도록 질의를 보강한다.
- 새 진단을 만드는 것이 아니라, retrieval용 질의를 다시 쓰는 것이다.

작성 규칙:
1. 현재 오류 영역을 가장 우선해서 반영하라.
2. initial_diagnosis_result의 check_item, error_type, improvement_text를 핵심 단서로 사용하라.
3. user_initial_diagnosis가 있으면 보조 단서로 반영하라.
4. graded_candidates에서 낮은 관련성으로 판단된 이유를 보고, 너무 넓거나 엉뚱한 표현은 줄여라.
5. retrieval에 유리하도록 짧지만 충분히 구체적인 한국어 질의문으로 작성하라.
6. 불필요한 설명, 메타 발언, 평가 문장은 쓰지 마라.
7. 반드시 JSON만 출력하라.

반드시 아래 형식으로만 출력:
{{
  "rewritten_query": "..."
}}
""".strip()


def invoke_rewriter_json(llm: Any, prompt: str) -> dict:
    return llm.invoke_json(prompt=prompt, image_paths=None)


def run_query_rewrite(
    llm: Any,
    user_initial_diagnosis: str | None,
    initial_diagnosis_result: dict,
    graded_candidates: list[dict],
) -> dict:
    prompt = build_query_rewrite_prompt(
        user_initial_diagnosis=user_initial_diagnosis,
        initial_diagnosis_result=initial_diagnosis_result,
        graded_candidates=graded_candidates,
    )
    return invoke_rewriter_json(llm=llm, prompt=prompt)
