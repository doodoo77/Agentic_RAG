# Colab 실행 안내

## 압축 해제 후 폴더
압축을 풀면 `/content/rag_system_bundle_real_fixed_v2/` 폴더가 생깁니다.

## main_pipeline.ipynb 위치
- `/content/rag_system_bundle_real_fixed_v2/main_pipeline.ipynb`

## 엑셀 파일 위치
가장 간단한 위치:
- `/content/rag_system_bundle_real_fixed_v2/golden_text.xlsx`

기본 템플릿:
- `/content/rag_system_bundle_real_fixed_v2/golden_text_template.xlsx`

템플릿을 복사해서 `golden_text.xlsx`로 바꿔 쓰면 됩니다.

## 엑셀 컬럼명
- `지침`
- `오류 유형`

## 과거진단이력 PPTX/PDF 넣는 위치
프로젝트별로 아래 폴더를 만들어 그 안에 넣으면 됩니다.
- `/content/rag_case_store/<project_id>/past_diagnosis_history/`

예시:
- `/content/rag_case_store/demo_project/past_diagnosis_history/report_01.pptx`
- `/content/rag_case_store/demo_project/past_diagnosis_history/report_02.pptx`

## 과거진단이력 전처리 + vectorDB 생성
아래 스크립트를 먼저 실행하면, PPTX/PDF 안의 오류 이미지와 텍스트를 분리해서 `cases.json`과 `vector_store.json`을 생성합니다.

```bash
cd /content/rag_system_bundle_real_fixed_v2
python -m rag_system.ingest.build_case_db \
  --project-id demo_project \
  --input-dir /content/rag_case_store/demo_project/past_diagnosis_history \
  --case-store-root /content/rag_case_store
```

생성 결과:
- `/content/rag_case_store/<project_id>/cases.json`
- `/content/rag_case_store/<project_id>/vector_store.json`
- `/content/rag_case_store/<project_id>/manifest.json`
- `/content/rag_case_store/<project_id>/preprocessed/...`

## 검색 동작
`rag_system.nodes.retrieve.retrieve_cases_weighted_fusion()`는 아래 우선순위로 검색합니다.
1. `vector_store.json`이 있으면 사전 계산된 이미지/텍스트 임베딩으로 검색
2. 없으면 `cases.json`을 읽어서 런타임에 임베딩 계산

최종 점수는 이미지 유사도와 텍스트 유사도를 softmax 가중치로 합성합니다.

## state별 출력
노트북의 `print_stream_updates(app, state)` 셀이 각 노드별 state 갱신 내용을 출력합니다.
