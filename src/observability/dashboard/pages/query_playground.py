"""Interactive query playground for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.observability.dashboard.services.query_service import QueryExecutionResult, QueryService
from src.observability.dashboard.ui import (
    MetricCard,
    format_ms,
    render_empty_state,
    render_metric_cards,
    render_page_header,
    render_section_header,
)


EXAMPLE_QUERIES = [
    "纳瓦尔关于财富自由的核心观点是什么？",
    "这份文档主要讲了哪些章节？",
    "书里提到了哪些与幸福相关的建议？",
]


@st.cache_resource
def _get_query_service() -> QueryService:
    """Return a cached query service instance for the dashboard."""
    return QueryService()


def _result_to_state(result: QueryExecutionResult) -> dict[str, Any]:
    """Convert a query result into a Streamlit session-state friendly payload."""
    return {
        "query": result.query,
        "collection": result.collection,
        "top_k": result.top_k,
        "response_content": result.response.content,
        "response_metadata": result.response.metadata,
        "citations": [citation.to_dict() for citation in result.response.citations],
        "final_results": [item.to_dict() for item in result.final_results],
        "dense_results": [item.to_dict() for item in result.dense_results],
        "sparse_results": [item.to_dict() for item in result.sparse_results],
        "processed_keywords": result.processed_keywords,
        "dense_error": result.dense_error,
        "sparse_error": result.sparse_error,
        "used_fallback": result.used_fallback,
        "rerank_applied": result.rerank_applied,
        "rerank_used_fallback": result.rerank_used_fallback,
        "rerank_fallback_reason": result.rerank_fallback_reason,
        "reranker_type": result.reranker_type,
        "elapsed_ms": result.elapsed_ms,
    }


def _render_result_list(
    results: list[dict[str, Any]],
    empty_message: str,
    key_prefix: str,
) -> None:
    """Render a list of retrieval results as expanders."""
    if not results:
        render_empty_state("No results", empty_message)
        return

    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {}) or {}
        score = float(result.get("score", 0.0))
        source_name = Path(str(metadata.get("source_path", "unknown"))).name or "unknown"
        label = f"#{index} · {source_name} · score={score:.4f}"

        with st.expander(label, expanded=(index == 1 and len(results) <= 3)):
            cards = [
                MetricCard(label="Score", value=f"{score:.4f}", caption="Current ranking score"),
            ]
            if metadata.get("page") is not None or metadata.get("page_num") is not None:
                cards.append(
                    MetricCard(
                        label="Page",
                        value=metadata.get("page") or metadata.get("page_num"),
                        caption="Source page",
                    )
                )
            if metadata.get("chunk_index") is not None:
                cards.append(
                    MetricCard(
                        label="Chunk Index",
                        value=metadata.get("chunk_index"),
                        caption="Chunk position in document",
                    )
                )
            if metadata.get("original_score") is not None:
                cards.append(
                    MetricCard(
                        label="Original Score",
                        value=f"{float(metadata['original_score']):.4f}",
                        caption="Pre-rerank score",
                    )
                )
            if metadata.get("rerank_score") is not None:
                cards.append(
                    MetricCard(
                        label="Rerank Score",
                        value=f"{float(metadata['rerank_score']):.4f}",
                        caption="Post-rerank score",
                    )
                )
            render_metric_cards(cards, columns=min(len(cards), 4))

            st.caption(f"Source: `{metadata.get('source_path', 'unknown')}`")
            st.text_area(
                "Chunk text",
                value=result.get("text", ""),
                height=220,
                disabled=True,
                key=(
                    f"query_playground_result_{key_prefix}_{index}_"
                    f"{result.get('chunk_id', 'unknown')}"
                ),
                label_visibility="collapsed",
            )
            with st.expander("Metadata", expanded=False):
                st.json(metadata)


def render() -> None:
    """Render the Query Playground page."""
    render_page_header(
        title="Query Playground",
        subtitle=(
            "直接在浏览器里发起查询，查看 dense / sparse / fusion / rerank 的召回效果，"
            "不用切回命令行。"
        ),
        badge="Interactive Retrieval",
    )

    render_section_header(
        "Run Query",
        "输入问题、选择 collection，并决定是否启用 rerank。页面会记录 trace，便于后续在 Query Traces 中复盘。",
    )

    service = _get_query_service()

    with st.form("query_playground_form"):
        query = st.text_area(
            "Query",
            value=st.session_state.get("query_playground_query", EXAMPLE_QUERIES[0]),
            height=110,
            placeholder="Ask a question about your indexed documents…",
        )
        control_col1, control_col2, control_col3, control_col4 = st.columns([1.2, 0.8, 0.9, 1.1])
        with control_col1:
            collection = st.text_input(
                "Collection",
                value=st.session_state.get("query_playground_collection", "default"),
            )
        with control_col2:
            top_k = st.number_input("Top-K", min_value=1, max_value=20, value=5)
        with control_col3:
            use_rerank = st.checkbox("Use rerank", value=False)
        with control_col4:
            example_query = st.selectbox("Example", options=EXAMPLE_QUERIES, index=0)

        example_clicked = st.form_submit_button("Use Example")
        submitted = st.form_submit_button("🔎 Run Query", type="primary")

    if example_clicked:
        st.session_state["query_playground_query"] = example_query
        st.session_state["query_playground_collection"] = collection.strip() or "default"
        st.rerun()

    if submitted:
        st.session_state["query_playground_query"] = query
        st.session_state["query_playground_collection"] = collection.strip() or "default"
        if not query.strip():
            st.warning("请输入查询内容后再运行。")
        else:
            with st.spinner("Running hybrid retrieval…"):
                try:
                    result = service.run_query(
                        query=query,
                        collection=collection.strip() or "default",
                        top_k=int(top_k),
                        use_rerank=use_rerank,
                    )
                except Exception as exc:
                    st.error(f"Query failed: {exc}")
                else:
                    st.session_state["query_playground_result"] = _result_to_state(result)

    state = st.session_state.get("query_playground_result")
    if not state:
        render_empty_state(
            "No query yet",
            "先执行一次查询，下面会展示格式化检索摘要、最终结果，以及 dense / sparse 召回细节。",
        )
        return

    render_metric_cards(
        [
            MetricCard(
                label="Final Results",
                value=len(state.get("final_results", [])),
                caption=f"Collection: {state.get('collection', 'default')}",
            ),
            MetricCard(
                label="Dense Hits",
                value=len(state.get("dense_results", [])),
                caption="Semantic retrieval candidates",
            ),
            MetricCard(
                label="Sparse Hits",
                value=len(state.get("sparse_results", [])),
                caption="Keyword retrieval candidates",
            ),
            MetricCard(
                label="Latency",
                value=format_ms(state.get("elapsed_ms")),
                caption="End-to-end query execution",
            ),
        ],
        columns=4,
    )

    info_bits = [
        f"Fallback used: {'Yes' if state.get('used_fallback') else 'No'}",
        f"Rerank: {'On' if state.get('rerank_applied') else 'Off'}",
        f"Reranker: {state.get('reranker_type', 'none')}",
    ]
    if state.get("processed_keywords"):
        info_bits.append("Keywords: " + ", ".join(state["processed_keywords"]))
    st.caption(" · ".join(info_bits))

    tab_summary, tab_final, tab_dense, tab_sparse, tab_debug = st.tabs(
        ["Summary", "Final Results", "Dense", "Sparse", "Debug"]
    )

    with tab_summary:
        st.markdown(state.get("response_content", ""))
        citations = state.get("citations", [])
        if citations:
            render_section_header("Citations", "格式化展示当前检索结果对应的引用来源。")
            st.dataframe(citations, use_container_width=True, hide_index=True)

    with tab_final:
        _render_result_list(
            state.get("final_results", []),
            "当前 query 没有最终召回结果。",
            key_prefix="final",
        )

    with tab_dense:
        if state.get("dense_error"):
            st.warning(f"Dense retrieval warning: {state['dense_error']}")
        _render_result_list(
            state.get("dense_results", []),
            "Dense retrieval 没有返回候选结果。",
            key_prefix="dense",
        )

    with tab_sparse:
        if state.get("sparse_error"):
            st.warning(f"Sparse retrieval warning: {state['sparse_error']}")
        _render_result_list(
            state.get("sparse_results", []),
            "Sparse retrieval 没有返回候选结果。",
            key_prefix="sparse",
        )

    with tab_debug:
        st.json(
            {
                "query": state.get("query"),
                "collection": state.get("collection"),
                "top_k": state.get("top_k"),
                "processed_keywords": state.get("processed_keywords", []),
                "used_fallback": state.get("used_fallback"),
                "dense_error": state.get("dense_error"),
                "sparse_error": state.get("sparse_error"),
                "rerank_applied": state.get("rerank_applied"),
                "rerank_used_fallback": state.get("rerank_used_fallback"),
                "rerank_fallback_reason": state.get("rerank_fallback_reason"),
                "reranker_type": state.get("reranker_type"),
                "elapsed_ms": state.get("elapsed_ms"),
                "response_metadata": state.get("response_metadata", {}),
            }
        )
