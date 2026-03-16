"""Query Traces page – browse query trace history with stage waterfall."""

from __future__ import annotations

import logging
from statistics import mean
from typing import Any

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService
from src.observability.dashboard.ui import (
    MetricCard,
    format_ms,
    render_empty_state,
    render_metric_cards,
    render_page_header,
    render_section_header,
)

logger = logging.getLogger(__name__)


def _matches_trace(trace: dict[str, Any], keyword: str, only_reranked: bool) -> bool:
    """Return whether a trace matches the active UI filters."""
    if only_reranked and not any(
        stage.get("stage") == "rerank" for stage in trace.get("stages", [])
    ):
        return False

    if not keyword:
        return True

    haystack = " ".join(
        [
            str(trace.get("metadata", {})),
            str(trace.get("stages", [])),
            str(trace.get("trace_id", "")),
        ]
    ).lower()
    return keyword.lower() in haystack


def render() -> None:
    """Render the Query Traces page."""
    render_page_header(
        title="Query Traces",
        subtitle=(
            "查看检索链路每一步的耗时、稠密/稀疏召回差异以及 rerank 影响，"
            "更快定位回答质量与性能问题。"
        ),
        badge="Retrieval Debugger",
    )

    svc = TraceService()
    traces = svc.list_traces(trace_type="query")

    if not traces:
        render_empty_state(
            "No query traces",
            "还没有记录到查询执行，先跑一次检索或问答请求即可看到结果。",
        )
        return

    filter_col1, filter_col2, filter_col3 = st.columns([2.2, 1.2, 1])
    with filter_col1:
        keyword = st.text_input(
            "Search traces",
            value="",
            key="qt_keyword",
            placeholder="Search query text, metadata or stage payload…",
        )
    with filter_col2:
        only_reranked = st.checkbox("Only reranked", value=False, key="qt_only_reranked")
    with filter_col3:
        limit = st.selectbox("Show recent", options=[5, 10, 20, 50, 100], index=2, key="qt_limit")

    filtered = [trace for trace in traces if _matches_trace(trace, keyword.strip(), only_reranked)]
    visible_traces = filtered[:limit]
    elapsed_values = [trace.get("elapsed_ms") for trace in visible_traces if trace.get("elapsed_ms") is not None]
    reranked_count = sum(
        1
        for trace in visible_traces
        if any(stage.get("stage") == "rerank" for stage in trace.get("stages", []))
    )

    render_metric_cards(
        [
            MetricCard(
                label="Visible Traces",
                value=len(visible_traces),
                caption=f"Filtered from {len(traces)} total query traces",
            ),
            MetricCard(
                label="Average Latency",
                value=format_ms(mean(elapsed_values)) if elapsed_values else "—",
                caption="Across current result set",
            ),
            MetricCard(
                label="Reranked Queries",
                value=reranked_count,
                caption="Queries that executed the rerank stage",
            ),
        ],
        columns=3,
    )

    render_section_header(
        "Query History",
        "将每条 trace 拆成时间线、检索对比、metadata 和评估四个视图，减少信息堆叠。",
    )

    if not visible_traces:
        render_empty_state(
            "No matching traces",
            "当前筛选条件下没有命中 query trace，试试清空关键词或关闭 rerank 过滤。",
        )
        return

    for idx, trace in enumerate(visible_traces):
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        total_label = format_ms(trace.get("elapsed_ms"))
        timings = svc.get_stage_timings(trace)
        meta = trace.get("metadata", {})
        query = meta.get("query", "")
        title = query[:72] + ("…" if len(query) > 72 else "") if query else f"Trace {trace_id[:12]}…"

        with st.expander(
            f"**{title}** — {started} — total: {total_label}",
            expanded=(idx == 0),
        ):
            if query:
                st.markdown(f"**Query**\n\n{query}")

            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Stages", len(timings))
            with summary_cols[1]:
                st.metric("Latency", total_label)
            with summary_cols[2]:
                st.metric("Collection", meta.get("collection", "—"))
            with summary_cols[3]:
                st.metric("Top-K", meta.get("top_k", "—"))

            tab_timing, tab_retrieval, tab_meta, tab_eval = st.tabs(
                ["Timing", "Retrieval", "Metadata", "Evaluate"]
            )

            with tab_timing:
                if timings:
                    chart_col, table_col = st.columns([1.2, 1])
                    with chart_col:
                        st.markdown("**Stage Waterfall**")
                        chart_data = {item["stage_name"]: item["elapsed_ms"] for item in timings}
                        st.bar_chart(chart_data, horizontal=True)
                    with table_col:
                        st.markdown("**Exact Timings**")
                        st.table(
                            [
                                {
                                    "Stage": item["stage_name"],
                                    "Elapsed (ms)": round(item["elapsed_ms"], 2),
                                }
                                for item in timings
                            ]
                        )
                else:
                    render_empty_state(
                        "No timing data",
                        "这条 trace 没有阶段耗时详情，可能来自旧日志或异常中断流程。",
                    )

            with tab_retrieval:
                dense = _find_stage(timings, "dense_retrieval")
                sparse = _find_stage(timings, "sparse_retrieval")
                rerank = _find_stage(timings, "rerank")

                render_metric_cards(
                    [
                        MetricCard(
                            label="Dense Retrieval",
                            value=format_ms(dense["elapsed_ms"]) if dense else "—",
                            caption="Embedding-based recall",
                        ),
                        MetricCard(
                            label="Sparse Retrieval",
                            value=format_ms(sparse["elapsed_ms"]) if sparse else "—",
                            caption="Keyword / BM25 recall",
                        ),
                        MetricCard(
                            label="Rerank",
                            value=format_ms(rerank["elapsed_ms"]) if rerank else "—",
                            caption="Final ordering refinement",
                        ),
                    ],
                    columns=3,
                )

                if dense or sparse:
                    dense_col, sparse_col = st.columns(2)
                    with dense_col:
                        if dense and dense["data"]:
                            st.markdown("**Dense payload**")
                            st.json(dense["data"])
                    with sparse_col:
                        if sparse and sparse["data"]:
                            st.markdown("**Sparse payload**")
                            st.json(sparse["data"])

                if rerank:
                    rerank_cols = st.columns(3)
                    with rerank_cols[0]:
                        st.metric("Elapsed", format_ms(rerank["elapsed_ms"]))
                    with rerank_cols[1]:
                        st.metric("Input count", rerank["data"].get("input_count", "—"))
                    with rerank_cols[2]:
                        st.metric("Output count", rerank["data"].get("output_count", "—"))

            with tab_meta:
                if meta:
                    st.json(meta)
                else:
                    render_empty_state(
                        "No metadata",
                        "这条 query trace 没有附带 metadata。",
                    )

            with tab_eval:
                _render_evaluate_button(trace, idx)


def _render_evaluate_button(trace: dict[str, Any], idx: int) -> None:
    """Render a Ragas evaluate button for a single query trace."""
    meta = trace.get("metadata", {})
    query = meta.get("query", "")
    if not query:
        render_empty_state(
            "Evaluation unavailable",
            "当前 trace 缺少 query 文本，因此无法重放检索并做 Ragas 评分。",
        )
        return

    st.caption(
        "Uses Ragas to score faithfulness, answer relevancy and context precision. "
        "This may call an external LLM and take a few seconds."
    )
    clicked = st.button(
        "📏 Ragas Evaluate",
        key=f"eval_trace_{idx}",
        help="Re-run this query and score with Ragas (LLM-as-Judge)",
    )

    result_key = f"eval_result_{idx}"
    if result_key in st.session_state and not clicked:
        _display_eval_metrics(st.session_state[result_key])

    if clicked:
        with st.spinner("Running Ragas evaluation…"):
            result = _evaluate_single_trace(query, meta)
        st.session_state[result_key] = result
        _display_eval_metrics(result)


def _evaluate_single_trace(
    query: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Re-run retrieval and evaluate a single query with Ragas."""
    try:
        from src.core.settings import load_settings
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        settings = load_settings()

        eval_settings = settings.evaluation
        override = type(eval_settings)(
            enabled=True,
            provider="ragas",
            metrics=eval_settings.metrics if hasattr(eval_settings, "metrics") else [],
        )
        evaluator = EvaluatorFactory.create(override)

        collection = meta.get("collection", "default")
        top_k = meta.get("top_k", 10)
        chunks = _retrieve_chunks(settings, query, top_k, collection)

        if not chunks:
            return {"error": "No chunks retrieved — is data indexed?"}

        texts = []
        for chunk in chunks:
            if hasattr(chunk, "text"):
                texts.append(chunk.text)
            elif isinstance(chunk, dict):
                texts.append(chunk.get("text", str(chunk)))
            else:
                texts.append(str(chunk))
        answer = " ".join(texts[:5])

        metrics = evaluator.evaluate(
            query=query,
            retrieved_chunks=chunks,
            generated_answer=answer,
        )
        return {"metrics": metrics}

    except ImportError as exc:
        return {"error": f"Ragas not installed: {exc}"}
    except Exception as exc:
        logger.exception("Ragas evaluation failed")
        return {"error": str(exc)}


def _retrieve_chunks(
    settings: Any,
    query: str,
    top_k: int,
    collection: str,
) -> list:
    """Re-run HybridSearch to retrieve chunks for evaluation."""
    try:
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        vector_store = VectorStoreFactory.create(settings, collection_name=collection)
        embedding_client = EmbeddingFactory.create(settings)
        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection
        query_processor = QueryProcessor()
        hybrid_search = create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )

        results = hybrid_search.search(query=query, top_k=top_k)
        return results if isinstance(results, list) else results.results
    except Exception as exc:
        logger.warning("Retrieval for evaluation failed: %s", exc)
        return []


def _display_eval_metrics(result: dict[str, Any]) -> None:
    """Display evaluation result (metrics or error)."""
    if "error" in result:
        st.error(f"❌ Evaluation failed: {result['error']}")
        return

    metrics = result.get("metrics", {})
    if not metrics:
        st.warning("No metrics returned.")
        return

    render_metric_cards(
        [
            MetricCard(
                label=name.replace("_", " ").title(),
                value=f"{value:.4f}",
                caption="Ragas score",
            )
            for name, value in sorted(metrics.items())
        ],
        columns=min(len(metrics), 4),
    )


def _find_stage(timings: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    """Find a stage dict by name, or None."""
    for timing in timings:
        if timing["stage_name"] == name:
            return timing
    return None
