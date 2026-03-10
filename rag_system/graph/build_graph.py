
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from rag_system.models.schemas import PipelineState
from rag_system.nodes.feedback import save_long_term_memory
from rag_system.nodes.initial_diagnosis import resolve_allowed_pairs, run_initial_diagnosis
from rag_system.nodes.memory_early_exit import run_memory_retrieval_early_exit
from rag_system.nodes.normalize import normalize_input
from rag_system.nodes.retrieve import retrieve_cases_weighted_fusion
from rag_system.nodes.grader import run_grader
from rag_system.nodes.rewrite import run_query_rewrite


def make_graph(llm_diagnosis: Any, llm_grader: Any, llm_rewriter: Any, memory_client: Any):
    def input_processor(state: PipelineState) -> PipelineState:
        normalized = normalize_input(
            project_id=state['project_id'],
            image_path=state['image_path'],
            user_initial_diagnosis=state.get('user_initial_diagnosis'),
        )
        return {**state, 'normalized_input': normalized.model_dump()}

    def initial_diagnosis_node(state: PipelineState) -> PipelineState:
        allowed_pairs = resolve_allowed_pairs(
            allowed_pairs=state.get('allowed_pairs'),
            xlsx_path=state.get('allowed_pairs_xlsx_path'),
            sheet_name=state.get('allowed_pairs_sheet_name', 'Sheet1') or 'Sheet1',
        )
        result, is_valid, prompt = run_initial_diagnosis(
            llm=llm_diagnosis,
            image_path=state['normalized_input']['image_path'],
            user_initial_diagnosis=state['normalized_input'].get('user_initial_diagnosis'),
            allowed_pairs=allowed_pairs,
        )
        return {
            **state,
            'allowed_pairs': allowed_pairs,
            'allowed_pairs_text': '\n'.join([f'- check_item: {ci} | error_type: {et}' for ci, et in allowed_pairs]),
            'initial_diagnosis_prompt': prompt,
            'initial_diagnosis_result': result,
            'initial_diagnosis_pair_valid': is_valid,
        }

    def memory_early_exit_node(state: PipelineState) -> PipelineState:
        early_exit_result, memory_candidates = run_memory_retrieval_early_exit(
            memory_client=memory_client,
            project_id=state['normalized_input']['project_id'],
            image_path=state['normalized_input']['image_path'],
            user_initial_diagnosis=state['normalized_input'].get('user_initial_diagnosis'),
            initial_diagnosis_result=state['initial_diagnosis_result'],
            early_exit_threshold=state.get('early_exit_threshold', 0.90),
            top_k=state.get('top_k', 10),
        )
        out = {
            **state,
            'memory_candidates': memory_candidates,
            'early_exit_result': early_exit_result,
            'early_exit_triggered': early_exit_result['early_exit_triggered'],
            'selected_memory': early_exit_result['selected_memory'],
            'selected_similarity': early_exit_result['selected_similarity'],
            'diagnosis_result': early_exit_result['diagnosis_result'],
        }
        if early_exit_result['early_exit_triggered'] and early_exit_result['selected_memory']:
            out['retrieved_image_path'] = early_exit_result['selected_memory']['retrieved_image_path']
            out['retrieved_result'] = early_exit_result['selected_memory']['retrieved_result']
        return out

    def retrieve_node(state: PipelineState) -> PipelineState:
        retrieval_candidates, query_text = retrieve_cases_weighted_fusion(
            project_id=state['normalized_input']['project_id'],
            image_path=state['normalized_input']['image_path'],
            user_initial_diagnosis=state['normalized_input'].get('user_initial_diagnosis'),
            initial_diagnosis_result=state['initial_diagnosis_result'],
            rewritten_query=state.get('rewritten_query'),
            top_k=state.get('top_k', 5),
        )
        return {
            **state,
            'retrieval_candidates': retrieval_candidates,
            'retrieval_query_text': query_text,
            'retrieval_mode': 'weighted_fusion',
        }

    def grader_node(state: PipelineState) -> PipelineState:
        grade_result = run_grader(
            llm=llm_grader,
            image_path=state['normalized_input']['image_path'],
            user_initial_diagnosis=state['normalized_input'].get('user_initial_diagnosis'),
            initial_diagnosis_result=state['initial_diagnosis_result'],
            retrieval_candidates=state['retrieval_candidates'],
        )
        out = {
            **state,
            'graded_candidates': grade_result['graded_candidates'],
            'has_relevant_candidate': grade_result['has_relevant_candidate'],
            'selected_candidate': grade_result['selected_candidate'],
            'grade_reason': grade_result['grade_reason'],
        }
        if grade_result['has_relevant_candidate']:
            out['diagnosis_result'] = grade_result['selected_candidate']['retrieved_result']
            out['retrieved_image_path'] = grade_result['selected_candidate']['case_image_path']
            out['retrieved_result'] = grade_result['selected_candidate']['retrieved_result']
        return out

    def rewrite_node(state: PipelineState) -> PipelineState:
        rewrite_result = run_query_rewrite(
            llm=llm_rewriter,
            user_initial_diagnosis=state['normalized_input'].get('user_initial_diagnosis'),
            initial_diagnosis_result=state['initial_diagnosis_result'],
            graded_candidates=state.get('graded_candidates', []),
        )
        return {
            **state,
            'rewritten_query': rewrite_result['rewritten_query'],
            'rewrite_count': state.get('rewrite_count', 0) + 1,
        }

    def finalize_node(state: PipelineState) -> PipelineState:
        final_output = {
            'project_id': state['normalized_input']['project_id'],
            'image_path': state['normalized_input']['image_path'],
            'diagnosis_result': state.get('diagnosis_result'),
            'early_exit_triggered': state.get('early_exit_triggered', False),
            'selected_similarity': state.get('selected_similarity'),
            'grade_reason': state.get('grade_reason'),
            'retrieved_image_path': state.get('retrieved_image_path'),
        }
        return {**state, 'final_output': final_output}

    def feedback_node(state: PipelineState) -> PipelineState:
        feedback = state.get('feedback')
        retrieved_image_path = state.get('retrieved_image_path')
        retrieved_result = state.get('retrieved_result') or state.get('diagnosis_result')
        if not feedback or not retrieved_image_path or not retrieved_result:
            return {**state, 'memory_saved': False, 'memory_event': None}

        save_result = save_long_term_memory(
            memory_client=memory_client,
            project_id=state['normalized_input']['project_id'],
            query_image_path=state['normalized_input']['image_path'],
            retrieved_image_path=retrieved_image_path,
            retrieved_result=retrieved_result,
            feedback=feedback,
        )
        return {
            **state,
            'memory_saved': save_result['memory_saved'],
            'memory_event': save_result['memory_event'],
        }

    def route_after_early_exit(state: PipelineState) -> str:
        return 'finalize' if state.get('early_exit_triggered') else 'retrieve'

    def route_after_grader(state: PipelineState) -> str:
        if state.get('has_relevant_candidate'):
            return 'finalize'
        if state.get('rewrite_count', 0) >= state.get('max_rewrite_count', 1):
            return 'finalize'
        return 'rewrite'

    graph_builder = StateGraph(PipelineState)
    graph_builder.add_node('input_processor', input_processor)
    graph_builder.add_node('initial_diagnosis', initial_diagnosis_node)
    graph_builder.add_node('memory_early_exit', memory_early_exit_node)
    graph_builder.add_node('retrieve', retrieve_node)
    graph_builder.add_node('grader', grader_node)
    graph_builder.add_node('rewrite', rewrite_node)
    graph_builder.add_node('finalize', finalize_node)
    graph_builder.add_node('feedback', feedback_node)

    graph_builder.add_edge(START, 'input_processor')
    graph_builder.add_edge('input_processor', 'initial_diagnosis')
    graph_builder.add_edge('initial_diagnosis', 'memory_early_exit')
    graph_builder.add_conditional_edges(
        'memory_early_exit',
        route_after_early_exit,
        {'finalize': 'finalize', 'retrieve': 'retrieve'},
    )
    graph_builder.add_edge('retrieve', 'grader')
    graph_builder.add_conditional_edges(
        'grader',
        route_after_grader,
        {'finalize': 'finalize', 'rewrite': 'rewrite'},
    )
    graph_builder.add_edge('rewrite', 'retrieve')
    graph_builder.add_edge('finalize', 'feedback')
    graph_builder.add_edge('feedback', END)

    return graph_builder.compile()
