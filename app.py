from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from rag_system.graph import build_graph
from rag_system.models.schemas import PipelineState
from rag_system.nodes.feedback import save_long_term_memory


st.set_page_config(page_title="RAG System Review App", layout="wide")


def _safe_json(value: Any):
    try:
        st.json(value)
    except Exception:
        st.write(value)


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
            }
        ]
    )
    st.dataframe(df, use_container_width=True)


def _render_memory_early_exit(node_state: dict):
    st.subheader("2. Memory Early Exit")
    df = pd.DataFrame(
        [
            {
                "memory candidate count": len(node_state.get("memory_candidates") or []),
                "best similarity": node_state.get("best_memory_similarity"),
                "threshold": node_state.get("early_exit_threshold"),
                "early_exit_triggered": node_state.get("memory_early_exit_triggered"),
            }
        ]
    )
    st.dataframe(df, use_container_width=True)


def _render_retrieve(node_state: dict):
    st.subheader("3. Retrieve")
    candidates = (node_state.get("retrieval_candidates") or [])[:3]
    if not candidates:
        st.info("retrieval 결과가 없습니다.")
        return

    rows = []
    for idx, c in enumerate(candidates, start=1):
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

    for idx, c in enumerate(candidates, start=1):
        with st.expander(f"retrieval candidate #{idx}", expanded=(idx == 1)):
            left, right = st.columns([1, 1.4])
            with left:
                image_path = c.get("case_image_path")
                if image_path and Path(image_path).exists():
                    st.image(image_path, caption=image_path, use_container_width=True)
                else:
                    st.warning("case 이미지 경로를 찾을 수 없습니다.")
            with right:
                _safe_json(c)


def _render_grader(node_state: dict):
    st.subheader("4. Grader")
    graded = (node_state.get("graded_candidates") or [])[:3]
    if not graded:
        st.info("grader 결과가 없습니다.")
        return

    rows = []
    for idx, g in enumerate(graded, start=1):
        rows.append(
            {
                "rank": idx,
                "case_image_path": g.get("case_image_path", ""),
                "grader_score": g.get("grader_score"),
                "is_relevant": g.get("is_relevant"),
                "reason": g.get("reason", ""),
                "selected": g.get("selected"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_finalize(node_state: dict):
    st.subheader("5. Finalize")
    final_output = node_state.get("final_output") or {}
    selected = node_state.get("selected_candidate") or {}

    left, right = st.columns([1, 1.2])

    with left:
        image_path = selected.get("case_image_path")
        if image_path and Path(image_path).exists():
            st.image(image_path, caption="selected case image", use_container_width=True)
        else:
            st.info("선택된 case 이미지가 없습니다.")

    with right:
        df = pd.DataFrame(
            [
                {
                    "check_item": final_output.get("check_item", ""),
                    "error_type": final_output.get("error_type", ""),
                    "improvement_text": final_output.get("improvement_text", ""),
                    "improvement_code": final_output.get("improvement_code", ""),
                }
            ]
        )
        st.dataframe(df, use_container_width=True)

    with st.expander("final_output raw"):
        _safe_json(final_output)


def _render_feedback_trace(node_state: dict):
    st.subheader("6. Feedback")
    df = pd.DataFrame(
        [
            {
                "feedback": node_state.get("human_feedback"),
                "memory_saved": node_state.get("memory_saved"),
            }
        ]
    )
    st.dataframe(df, use_container_width=True)


def _feedback_buttons(final_state: dict):
    st.markdown("---")
    st.subheader("전문가 피드백 저장")

    comment = st.text_area(
        "피드백 코멘트",
        placeholder="필요하면 간단한 코멘트를 입력하세요.",
        key="expert_feedback_comment",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div style='text-align:center;font-size:72px;'>👍</div>", unsafe_allow_html=True)
        approve = st.button("전문가 승인", use_container_width=True, key="approve_btn")

    with col2:
        st.markdown("<div style='text-align:center;font-size:72px;'>👎</div>", unsafe_allow_html=True)
        reject = st.button("수정 필요", use_container_width=True, key="reject_btn")

    if approve or reject:
        feedback_state = dict(final_state)
        feedback_state["human_feedback"] = "thumbs_up" if approve else "thumbs_down"
        feedback_state["human_feedback_comment"] = comment

        try:
            save_long_term_memory(feedback_state)
            st.success("mem0 저장 완료")
        except Exception as e:
            st.error(f"mem0 저장 실패: {e}")


def main():
    st.title("RAG System Review App")

    with st.sidebar:
        st.header("설정")
        rag_case_store_root = st.text_input("RAG_CASE_STORE_ROOT", value="./rag_case_store")
        project_id = st.text_input("project_id", value="default_project")
        allowed_pairs_xlsx_path = st.text_input("allowed_pairs_xlsx_path", value="./golden_text.xlsx")
        allowed_pairs_sheet_name = st.text_input("allowed_pairs_sheet_name", value="Sheet1")
        top_k = st.number_input("top_k", min_value=1, max_value=20, value=5, step=1)
        early_exit_threshold = st.number_input("early_exit_threshold", value=0.85, step=0.01, format="%.2f")
        max_rewrite_count = st.number_input("max_rewrite_count", min_value=0, max_value=10, value=2, step=1)
        diagnosis_model = st.text_input("diagnosis_model", value="gpt-4o-mini")
        grader_model = st.text_input("grader_model", value="gpt-4o-mini")
        rewriter_model = st.text_input("rewriter_model", value="gpt-4o-mini")

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

    with st.expander("입력 파라미터", expanded=False):
        _safe_json(state)

    try:
        app = build_graph()
    except Exception as e:
        st.error(f"그래프 생성 실패: {e}")
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

                if node_name == "initial_diagnosis":
                    _render_initial_diagnosis(node_state)
                elif node_name == "memory_early_exit":
                    _render_memory_early_exit(node_state)
                elif node_name == "retrieve":
                    _render_retrieve(node_state)
                elif node_name == "grader":
                    _render_grader(node_state)
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

    _feedback_buttons(final_state)


if __name__ == "__main__":
    main()