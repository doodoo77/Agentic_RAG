
from __future__ import annotations

from typing import Any

from rag_system.models.schemas import GradeResult, GradedCandidate


def build_grader_prompt(
    user_initial_diagnosis: str | None,
    initial_diagnosis_result: dict,
    retrieval_candidate: dict,
) -> str:
    user_text = (user_initial_diagnosis or '').strip()
    if not user_text:
        user_text = '(사용자 초기 진단 없음)'

    candidate_result = retrieval_candidate['retrieved_result']
    final_score = retrieval_candidate['final_score']

    return f"""
[Current User Initial Diagnosis]
{user_text}

[Current Initial Diagnosis Result]
{initial_diagnosis_result}

[Retrieved Candidate Final Score]
{final_score}

[Retrieved Candidate Result]
{candidate_result}

너의 역할:
현재 입력 이미지와 검색된 과거 사례 이미지를 비교하고,
이 후보가 현재 사용자의 오류 영역을 진단하고 해결하는 데 실제로 의미 있는 후보인지 평가하라.

평가 시 반드시 고려할 것:
1. 현재 이미지에서 문제된 오류 영역과 이 후보 사례가 직접 관련 있는가
2. 후보의 retrieved_result가 현재 문제에 재사용 가능한가
3. retrieval의 최종 점수(final_score)가 이 판단을 어느 정도 뒷받침하는가
4. 단순히 표면적으로 비슷한 것이 아니라 실제 진단/개선에 쓸 수 있는 후보인가

출력 규칙:
- 반드시 JSON만 출력한다
- grader_score는 0.0 ~ 1.0 사이 float
- is_relevant는 true 또는 false
- reason은 한두 문장으로 작성한다

반드시 아래 형식으로만 출력:
{{
  "is_relevant": true,
  "grader_score": 0.0,
  "reason": "..."
}}
""".strip()


def invoke_grader_json(llm: Any, image_path: str, case_image_path: str, prompt: str) -> dict:
    return llm.invoke_json(prompt=prompt, image_paths=[image_path, case_image_path])


def grade_retrieval_candidate(
    llm: Any,
    image_path: str,
    user_initial_diagnosis: str | None,
    initial_diagnosis_result: dict,
    retrieval_candidate: dict,
) -> GradedCandidate:
    prompt = build_grader_prompt(user_initial_diagnosis, initial_diagnosis_result, retrieval_candidate)
    parsed = invoke_grader_json(
        llm=llm,
        image_path=image_path,
        case_image_path=retrieval_candidate['case_image_path'],
        prompt=prompt,
    )
    return {
        'candidate': retrieval_candidate,
        'is_relevant': bool(parsed['is_relevant']),
        'grader_score': float(parsed['grader_score']),
        'reason': str(parsed['reason']),
    }


def run_grader(
    llm: Any,
    image_path: str,
    user_initial_diagnosis: str | None,
    initial_diagnosis_result: dict,
    retrieval_candidates: list[dict],
) -> GradeResult:
    graded_candidates: list[GradedCandidate] = []
    for candidate in retrieval_candidates:
        graded_candidates.append(
            grade_retrieval_candidate(
                llm=llm,
                image_path=image_path,
                user_initial_diagnosis=user_initial_diagnosis,
                initial_diagnosis_result=initial_diagnosis_result,
                retrieval_candidate=candidate,
            )
        )

    graded_candidates.sort(key=lambda x: x['grader_score'], reverse=True)
    relevant_candidates = [x for x in graded_candidates if x['is_relevant']]

    if not relevant_candidates:
        return {
            'has_relevant_candidate': False,
            'selected_candidate': None,
            'graded_candidates': graded_candidates,
            'grade_reason': '검색된 후보들이 현재 오류 영역과 직접 관련된 사례라고 보기 어렵습니다.',
        }

    selected = relevant_candidates[0]
    return {
        'has_relevant_candidate': True,
        'selected_candidate': selected['candidate'],
        'graded_candidates': graded_candidates,
        'grade_reason': selected['reason'],
    }
