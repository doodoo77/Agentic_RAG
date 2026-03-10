# Accessibility Diagnosis Agentic RAG

> Search past diagnosis history, reuse validated fixes, and continuously improve with project-level memory.

A multimodal Agentic RAG system for **UI accessibility diagnosis**.
Instead of generating a diagnosis from scratch every time, this project retrieves similar cases from past reports, grades the candidates, rewrites the query when retrieval fails, and stores user feedback as long-term memory for future runs.

## Why this project exists

In accessibility QA workflows, the same UI issues appear repeatedly across screens, features, and releases.
But in practice, historical reports are often buried in PPTX/PDF files, which makes it hard to:

- find similar past cases quickly
- reuse proven improvement guidance and code
- maintain diagnosis consistency across projects
- improve retrieval quality from user feedback over time

This repository turns scattered diagnosis history into a **searchable multimodal case base** and runs an **agentic retrieval loop** on top of it.

## Highlights

- **Agentic RAG pipeline for diagnosis history search**  
  Generates an initial diagnosis, retrieves relevant historical cases, grades retrieved candidates, and rewrites the query when needed.

- **Project-level long-term memory**  
  Stores validated cases separately by project and checks memory first for fast reuse through early exit.

- **Feedback-driven quality improvement loop**  
  Saves `thumbs_up` and `thumbs_down` signals so retrieval behavior improves over repeated usage.

- **Cross-modal retrieval over image + text**  
  Combines image similarity and text similarity to retrieve cases that are visually and semantically aligned.

## What it does

### 1. Build a searchable case database from past diagnosis reports
The ingestion pipeline parses historical `.pptx` and `.pdf` reports and extracts:

- error-region images
- diagnosis text
- check items
- error types
- improvement text
- improvement code

Artifacts generated per project:

- `cases.json`
- `vector_store.json`
- `manifest.json`
- `preprocessed/`

### 2. Create an initial diagnosis from the current input
Given a current UI image and a user-written diagnosis note, the system generates a structured first-pass diagnosis.

Typical fields:

- `error_type`
- `check_item`
- `improvement_text`
- `improvement_code`

It also constrains outputs using the allowed combinations defined in `golden_text.xlsx`.

### 3. Check long-term memory first
Before running full retrieval, the system looks up previously validated project memory.
If a highly similar historical case is found, it can reuse that result immediately.

This reduces:

- repeated retrieval cost
- unnecessary LLM calls
- latency for recurring issue patterns

### 4. Retrieve similar cases using image and text together
The retriever uses both channels:

- **image retrieval** for visually similar UI issue regions
- **text retrieval** for semantically similar diagnosis intent

This is important because accessibility issues are often similar in layout and wording, not just one or the other.

### 5. Grade retrieved candidates with an LLM
High vector similarity does not always mean the case is actually reusable.
So the pipeline runs a grading step that checks whether retrieved candidates are truly relevant to the current issue.

### 6. Rewrite the query and retry when retrieval is weak
If the retrieval result is poor, the system rewrites the search query based on the failed candidates and tries again.

This forms the core agentic loop:

**diagnose → retrieve → grade → rewrite → retrieve again**

### 7. Store user feedback for future runs
When a user approves or rejects the final result, that signal is stored in long-term memory and used in later runs.

Over time, the system becomes better at:

- reusing good cases
- avoiding bad cases
- responding faster to repeated issue types

## System overview

```text
Past diagnosis reports (PPTX/PDF)
  -> preprocess
  -> structured cases + vector store

Current UI image + initial diagnosis note
  -> normalize
  -> initial_diagnosis
  -> memory_early_exit
      -> if matched: return reused result
      -> else: retrieve
  -> grader
      -> if relevant: finalize
      -> else: rewrite -> retrieve again
  -> feedback memory update
```

## Repository structure

```text
rag_system_bundle_real_fixed_v2/
├── main_pipeline_fixed.ipynb
├── README_COLAB_RUN.md
├── START_HERE.txt
├── requirements_colab.txt
├── golden_text.xlsx
├── golden_text_template.xlsx
├── example_input/
│   ├── README.txt
│   ├── test_img.png
│   └── initial_note_example.txt
├── past_diagnosis_history/
│   └── *.pptx | *.pdf
├── rag_system/
│   ├── clients/
│   │   └── openai_responses.py
│   ├── graph/
│   │   └── build_graph.py
│   ├── ingest/
│   │   └── build_case_db.py
│   ├── models/
│   │   └── schemas.py
│   ├── nodes/
│   │   ├── normalize.py
│   │   ├── initial_diagnosis.py
│   │   ├── memory_early_exit.py
│   │   ├── retrieve.py
│   │   ├── grader.py
│   │   ├── rewrite.py
│   │   └── feedback.py
│   └── preprocess/
│       └── a11y_preprocess.py
└── rag_case_store/
    └── <project_id>/
        ├── cases.json
        ├── vector_store.json
        ├── manifest.json
        └── preprocessed/
```

## Core modules

### `rag_system.ingest.build_case_db`
Builds the case database from historical reports.

### `rag_system.nodes.initial_diagnosis`
Creates the first structured diagnosis from the current image and note.

### `rag_system.nodes.memory_early_exit`
Checks project-level long-term memory and exits early when a validated similar case already exists.

### `rag_system.nodes.retrieve`
Runs cross-modal retrieval using image and text signals.

### `rag_system.nodes.grader`
Verifies whether retrieved candidates are actually reusable for the current issue.

### `rag_system.nodes.rewrite`
Rewrites the retrieval query when candidate quality is weak.

### `rag_system.nodes.feedback`
Stores user feedback into long-term memory for later reuse.

### `rag_system.graph.build_graph`
Connects the full execution flow as a graph-based pipeline.

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements_colab.txt
```

### 2. Prepare past diagnosis history
Place historical diagnosis files under the project-specific history directory.

```text
/content/rag_case_store/<project_id>/past_diagnosis_history/
```

Supported formats:

- `.pptx`
- `.pdf`

### 3. Build the case database

```bash
cd /content/rag_system_bundle_real_fixed_v2
python -m rag_system.ingest.build_case_db \
  --project-id demo_project \
  --input-dir /content/rag_case_store/demo_project/past_diagnosis_history \
  --case-store-root /content/rag_case_store
```

### 4. Prepare current input
Add the following files to `example_input/`:

- one or more target UI images
- one or more initial diagnosis notes

The notebook uses the first image file and the first text file it finds.

### 5. Run the main pipeline
Open and execute:

```text
main_pipeline_fixed.ipynb
```

The pipeline returns the selected diagnosis result and the retrieved reference case.

## Input schema

- `project_id`: project-scoped key for case storage and memory
- `image_path`: target UI image path
- `user_initial_diagnosis`: user-written diagnosis note
- `feedback`: optional, `thumbs_up` or `thumbs_down`

## Output schema

- `project_id`
- `image_path`
- `diagnosis_result`
- `early_exit_triggered`
- `selected_similarity`
- `grade_reason`
- `retrieved_image_path`

## Key design choices

### Multimodal retrieval instead of text-only RAG
Accessibility diagnosis depends heavily on **visual layout context** and **diagnosis language** together.
This project retrieves using both.

### Agentic loop instead of one-shot retrieval
The pipeline does not stop at top-k retrieval.
It evaluates candidate quality and retries with rewritten intent when needed.

### Long-term memory instead of stateless retrieval
Validated cases are remembered per project so repeated issue patterns become faster to solve.

### Feedback-aware operation instead of static ranking
The system learns from practical usage by storing explicit user preference signals.

## Example use cases

- accessibility QA assistant
- diagnosis history search for design and publishing review
- screenshot-based issue case recommendation
- standardization of repetitive UI diagnosis workflows

## Limitations

- historical diagnosis quality directly affects retrieval quality
- parsing quality may vary depending on report format consistency
- threshold tuning may need adjustment per project domain
- current implementation is notebook-first rather than service-first

## Roadmap

- API and batch inference support
- project-specific threshold auto-tuning
- stronger reranking with feedback-aware scoring
- visual evidence rendering for retrieved cases
- automatic diagnosis report generation

## Security note

Before publishing this repository, make sure no API keys, tokens, or private report files remain in notebooks, configs, or cached artifacts.

## Related files

- `README_COLAB_RUN.md`: Colab-oriented execution guide
- `START_HERE.txt`: shortest path to first run
- `golden_text_template.xlsx`: template for allowed diagnosis combinations
=======
# Multimodal Diagnostic RAG System (LangGraph + Python + Colab)

프로젝트 단위 long-term memory를 우선으로 사용하는 멀티모달 진단 이력 검색 시스템 MVP입니다.

## 포함 기능
- 이미지 단독 / 이미지+텍스트 / 이미지+초기진단 입력 처리
- 초기 진단 생성 (검색용 확장 질의 포함)
- project-level long-term memory 조회
- 유사도 + 만족도 기반 early exit
- 멀티모달 retrieval 2안 지원
  - `integrated`: 이미지/텍스트를 공통 임베딩 공간에서 검색
  - `split_fusion`: 이미지/텍스트 채널 분리 후 weighted fusion
- retrieval grader
- query rewrite 후 재검색
- sparse feedback 로깅 (기본 5회 중 1회)
- memory update
- LangGraph 상태 기반 오케스트레이션

## 디렉터리 구조
```text
rag_project/
  README.md
  requirements.txt
  .env.example
  notebooks/
    colab_quickstart.py
  data/
    projects/demo_project/
      cases.jsonl
  src/rag_system/
    main.py
    graph.py
    state.py
    settings.py
    config/prompts.py
    models/schemas.py
    models/encoders.py
    memory/store.py
    retrieval/scoring.py
    retrieval/retriever.py
    nodes/*.py
    utils/io.py
    utils/time.py
  tests/test_smoke.py
```

## 빠른 시작
```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -m rag_system.main
```

## Colab 사용
`notebooks/colab_quickstart.py`를 Colab에 올린 뒤 셀 단위로 실행하면 됩니다.

## 주의
이 저장소는 바로 확장 가능한 MVP 골격입니다.
- 실제 초기 진단 모델은 `InitialDiagnosisGenerator`에서 교체
- 실제 이미지/텍스트 임베딩 모델은 `EncoderFactory`에서 교체
- 실제 벡터DB는 현재 JSONL + numpy 기반 예제로 되어 있으며 FAISS/Qdrant/pgvector 등으로 쉽게 교체 가능