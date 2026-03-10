from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from mem0 import Memory

from rag_system.clients.openai_responses import OpenAIResponsesJSONClient
from rag_system.graph.build_graph import make_graph
from rag_system.models.schemas import PipelineState
from rag_system.nodes.feedback import save_long_term_memory

st.set_page_config(page_title="Agentic RAG Review", layout="wide")


def _safe_json(value: Any):
    try:
        st.json(value)
    except Exception:
        st.write(value)


def _render_input_summary(state: dict):
    st.subheader("0. Input")
    left, right = st.columns([1, 1.2])

    with left:
        image_path = state.get("image_path")
        if image_path and Path(image_path).exists():
            st.image(image_path, caption="uploaded image", use_container_width=True)

    with right:
        rows = [
            {
                "project_id": state.get("project_id", ""),
                "top_k": state.get("top_k"),
                "early_exit_threshold": state.get("early_exit_threshold"),
                "max_rewrite_count": state.get("max_rewrite_count"),
                "allowed_pairs_xlsx_path": state.get("allowed_pairs_xlsx_path", ""),
                "allowed_pairs_sheet_name": state.get("allowed_pairs_sheet_name", ""),
            }
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.markdown("**user_initial_diagnosis**")
        st.write(state.get("user_initial_diagnosis") or "(없음)")


def _render_initial_diagnosis(node_state: dict):
    st.subheader("1. Initial Diagnosis")
    result = node_state.get("initial_diagnosis_result") or {}

    df = pd.DataFrame(
        [
            {
                "check_item": result.get("check_item", ""),
                "error_type": result.get("error_type", ""),
                "improvement_text": result.get("improvement_text", ""),
                "improvement_code": result.get("improvement_code", ""),
                "pair_valid": node_state.get("initial_diagnosis_pair_valid"),
            }
        ]
    )
    st.dataframe(df, use_container_width=True)

    with st.expander("initial_diagnosis_prompt", expanded=False):
        st.code(node_state.get("initial_diagnosis_prompt", ""), language="text")


def _render_memory_early_exit(node_state: dict):
    st.subheader("2. Memory Early Exit")

    early_exit_result = node_state.get("early_exit_result") or {}
    memory_candidates = node_state.get("memory_candidates") or []

    top_mem0 = []
    for idx, item in enumerate(memory_candidates[:3], start=1):
        event = item.get("memory_event") or {}
        retrieved = event.get("retrieved_result") or {}
        top_mem0.append(
            {
                "rank": idx,
                "mem0_score": item.get("mem0_score"),
                "feedback": event.get("feedback"),
                "retrieved_image_path": event.get("retrieved_image_path"),
                "check_item": retrieved.get("check_item"),
                "error_type": retrieved.get("error_type"),
            }
        )

    summary_df = pd.DataFrame(
        [
            {
                "memory_candidate_count": len(memory_candidates),
                "early_exit_triggered": early_exit_result.get("early_exit_triggered"),
                "selected_similarity": early_exit_result.get("selected_similarity"),
                "selected_memory_exists": early_exit_result.get("selected_memory") is not None,
            }
        ]
    )
    st.dataframe(summary_df, use_container_width=True)

    if top_mem0:
        st.markdown("**top memory candidates**")
        st.dataframe(pd.DataFrame(top_mem0), use_container_width=True)

    selected_memory = early_exit_result.get("selected_memory")
    if selected_memory:
        with st.expander("selected memory detail", expanded=True):
            left, right = st.columns([1, 1.2])
            with left:
                mem_img = selected_memory.get("retrieved_image_path")
                if mem_img and Path(mem_img).exists():
                    st.image(mem_img, caption="selected memory image", use_container_width=True)
            with right:
                _safe_json(selected_memory)


def _render_retrieve(node_state: dict):
    st.subheader("3. Retrieve")

    query_text = node_state.get("retrieval_query_text", "")
    if query_text:
        st.markdown("**retrieval_query_text**")
        st.code(query_text, language="text")

    candidates = node_state.get("retrieval_candidates") or []
    if not candidates:
        st.info("retrieval 결과가 없습니다.")
        return

    rows = []
    for idx, c in enumerate(candidates[:3], start=1):
        rr = c.get("retrieved_result") or {}
        rows.append(
            {
                "rank": idx,
                "case_image_path": c.get("case_image_path", ""),
                "check_item": rr.get("check_item", ""),
                "error_type": rr.get("error_type", ""),
                "improvement_text": rr.get("improvement_text", ""),
                "improvement_code": rr.get("improvement_code", ""),
                "image_similarity": c.get("image_similarity"),
                "text_similarity": c.get("text_similarity"),
                "image_weight": c.get("image_weight"),
                "text_weight": c.get("text_weight"),
                "final_score": c.get("final_score"),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    for idx, c in enumerate(candidates[:3], start=1):
        with st.expander(f"retrieval candidate #{idx}", expanded=(idx == 1)):
            left, right = st.columns([1, 1.25])
            with left:
                image_path = c.get("case_image_path")
                if image_path and Path(image_path).exists():
                    st.image(image_path, caption=image_path, use_container_width=True)
            with right:
                _safe_json(c)


def _render_grader(node_state: dict):
    st.subheader("4. Grader")

    graded = node_state.get("graded_candidates") or []
    if not graded:
        st.info("grader 결과가 없습니다.")
        return

    rows = []
    for idx, g in enumerate(graded[:3], start=1):
        candidate = g.get("candidate") or {}
        rows.append(
            {
                "rank": idx,
                "case_image_path": candidate.get("case_image_path", ""),
                "grader_score": g.get("grader_score"),
                "is_relevant": g.get("is_relevant"),
                "reason": g.get("reason", ""),
                "selected": idx == 1 and bool(node_state.get("has_relevant_candidate")),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if node_state.get("grade_reason"):
        st.markdown("**grade_reason**")
        st.write(node_state.get("grade_reason"))

    selected = node_state.get("selected_candidate")
    if selected:
        with st.expander("selected candidate detail", expanded=True):
            left, right = st.columns([1, 1.25])
            with left:
                image_path = selected.get("case_image_path")
                if image_path and Path(image_path).exists():
                    st.image(image_path, caption="selected candidate", use_container_width=True)
            with right:
                _safe_json(selected)


def _render_rewrite(node_state: dict):
    st.subheader("5. Rewrite")
    df = pd.DataFrame(
        [
            {
                "rewrite_count": node_state.get("rewrite_count"),
                "rewritten_query": node_state.get("rewritten_query", ""),
            }
        ]
    )
    st.dataframe(df, use_container_width=True)


def _render_finalize(node_state: dict):
    st.subheader("6. Finalize")

    final_output = node_state.get("final_output") or {}
    diagnosis_result = final_output.get("diagnosis_result") or {}

    left, right = st.columns([1, 1.2])

    with left:
        image_path = final_output.get("retrieved_image_path")
        if image_path and Path(image_path).exists():
            st.image(image_path, caption="final selected image", use_container_width=True)
        else:
            st.info("선택된 이미지가 없습니다.")

    with right:
        df = pd.DataFrame(
            [
                {
                    "check_item": diagnosis_result.get("check_item", ""),
                    "error_type": diagnosis_result.get("error_type", ""),
                    "improvement_text": diagnosis_result.get("improvement_text", ""),
                    "improvement_code": diagnosis_result.get("improvement_code", ""),
                    "early_exit_triggered": final_output.get("early_exit_triggered"),
                    "selected_similarity": final_output.get("selected_similarity"),
                }
            ]
        )
        st.dataframe(df, use_container_width=True)

    with st.expander("final_output raw", expanded=False):
        _safe_json(final_output)


def _render_feedback_trace(node_state: dict):
    st.subheader("7. Feedback")
    if node_state.get("memory_saved"):
        st.success("mem0 저장 완료")
    else:
        st.info("아직 전문가 피드백이 입력되지 않아 mem0 저장 안 됨")

    df = pd.DataFrame(
        [
            {
                "feedback": node_state.get("feedback"),
                "memory_saved": node_state.get("memory_saved"),
            }
        ]
    )
    st.dataframe(df, use_container_width=True)


@st.cache_resource(show_spinner=False)
def _build_runtime(diagnosis_model: str, grader_model: str, rewriter_model: str):
    llm_diagnosis = OpenAIResponsesJSONClient(model=diagnosis_model)
    llm_grader = OpenAIResponsesJSONClient(model=grader_model)
    llm_rewriter = OpenAIResponsesJSONClient(model=rewriter_model)
    memory_client = Memory()

    app = make_graph(
        llm_diagnosis=llm_diagnosis,
        llm_grader=llm_grader,
        llm_rewriter=llm_rewriter,
        memory_client=memory_client,
    )
    return app, memory_client


def _render_expert_feedback(final_state: dict, memory_client: Any):
    st.markdown("---")
    st.subheader("전문가 피드백 저장")

    diagnosis_result = final_state.get("diagnosis_result")
    retrieved_image_path = final_state.get("retrieved_image_path")
    normalized_input = final_state.get("normalized_input") or {}

    if not diagnosis_result or not retrieved_image_path:
        st.warning("저장할 최종 결과가 없습니다.")
        return

    comment = st.text_area(
        "전문가 코멘트",
        placeholder="필요하면 짧게 남기세요.",
        key="expert_feedback_comment",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div style='text-align:center;font-size:88px;'>👍</div>", unsafe_allow_html=True)
        approve = st.button("전문가 승인", use_container_width=True, key="approve_btn")

    with col2:
        st.markdown("<div style='text-align:center;font-size:88px;'>👎</div>", unsafe_allow_html=True)
        reject = st.button("수정 필요", use_container_width=True, key="reject_btn")

    if approve or reject:
        feedback_value = "thumbs_up" if approve else "thumbs_down"

        try:
            save_result = save_long_term_memory(
                memory_client=memory_client,
                project_id=normalized_input["project_id"],
                query_image_path=normalized_input["image_path"],
                retrieved_image_path=retrieved_image_path,
                retrieved_result=diagnosis_result,
                feedback=feedback_value,
            )

            st.success("mem0 저장 완료")

            if comment.strip():
                st.info(f"전문가 코멘트: {comment.strip()}")

            with st.expander("저장된 memory_event", expanded=True):
                _safe_json(save_result)
        except Exception as e:
            st.error(f"mem0 저장 실패: {e}")


def main():
    st.title("Agentic RAG Review App")

    with st.sidebar:
        st.header("설정")
        rag_case_store_root = st.text_input("RAG_CASE_STORE_ROOT", value="./rag_case_store")
        project_id = st.text_input("project_id", value="default_project")
        allowed_pairs_xlsx_path = st.text_input("allowed_pairs_xlsx_path", value="./golden_text.xlsx")
        allowed_pairs_sheet_name = st.text_input("allowed_pairs_sheet_name", value="Sheet1")
        top_k = st.number_input("top_k", min_value=1, max_value=20, value=5, step=1)
        early_exit_threshold = st.number_input("early_exit_threshold", value=0.85, step=0.01, format="%.2f")
        max_rewrite_count = st.number_input("max_rewrite_count", min_value=0, max_value=10, value=2, step=1)
        diagnosis_model = st.text_input("diagnosis_model", value="gpt-4.1")
        grader_model = st.text_input("grader_model", value="gpt-4.1")
        rewriter_model = st.text_input("rewriter_model", value="gpt-4.1-mini")

    user_initial_diagnosis = st.text_area("초기 진단 텍스트", height=120)
    uploaded_image = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg", "webp"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="uploaded image", use_container_width=True)

    run = st.button("실행", type="primary", use_container_width=True)

    if not run:
        return

    if uploaded_image is None:
        st.error("이미지를 업로드하세요.")
        return

    os.environ["RAG_CASE_STORE_ROOT"] = rag_case_store_root

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_image.name).suffix or ".png") as tmp:
        tmp.write(uploaded_image.getbuffer())
        image_path = tmp.name

    state: PipelineState = {
        "project_id": project_id,
        "image_path": image_path,
        "user_initial_diagnosis": user_initial_diagnosis,
        "allowed_pairs_xlsx_path": allowed_pairs_xlsx_path,
        "allowed_pairs_sheet_name": allowed_pairs_sheet_name,
        "top_k": int(top_k),
        "early_exit_threshold": float(early_exit_threshold),
        "max_rewrite_count": int(max_rewrite_count),
        "diagnosis_model": diagnosis_model,
        "grader_model": grader_model,
        "rewriter_model": rewriter_model,
    }

    _render_input_summary(state)

    try:
        app, memory_client = _build_runtime(
            diagnosis_model=diagnosis_model,
            grader_model=grader_model,
            rewriter_model=rewriter_model,
        )
    except Exception as e:
        st.error(f"그래프 준비 실패: {e}")
        return

    seen_nodes = []
    final_state = dict(state)

    try:
        for event in app.stream(state):
            for node_name, node_state in event.items():
                seen_nodes.append(node_name)
                if isinstance(node_state, dict):
                    final_state.update(node_state)

                st.markdown("---")
                st.caption(f"node: {node_name}")

                if node_name == "input_processor":
                    pass
                elif node_name == "initial_diagnosis":
                    _render_initial_diagnosis(node_state)
                elif node_name == "memory_early_exit":
                    _render_memory_early_exit(node_state)
                elif node_name == "retrieve":
                    _render_retrieve(node_state)
                elif node_name == "grader":
                    _render_grader(node_state)
                elif node_name == "rewrite":
                    _render_rewrite(node_state)
                elif node_name == "finalize":
                    _render_finalize(node_state)
                elif node_name == "feedback":
                    _render_feedback_trace(node_state)
                else:
                    st.subheader(node_name)
                    _safe_json(node_state)

    except Exception as e:
        st.error(f"파이프라인 실행 실패: {e}")
        return

    st.markdown("---")
    st.subheader("Pipeline Trace")
    st.write(" → ".join(seen_nodes) if seen_nodes else "실행된 노드 없음")

    with st.expander("final_state raw", expanded=False):
        _safe_json(final_state)

    _render_expert_feedback(final_state, memory_client)


if __name__ == "__main__":
    main()