"""Overview page – system configuration and data statistics.

Displays:
- Quick summary metrics
- Component configuration cards
- Collection statistics
- Trace activity overview
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.observability.dashboard.services.config_service import ConfigService
from src.observability.dashboard.services.trace_service import TraceService
from src.observability.dashboard.ui import (
    MetricCard,
    render_component_card,
    render_empty_state,
    render_metric_cards,
    render_page_header,
    render_section_header,
)


def _safe_collection_stats() -> dict[str, Any]:
    """Attempt to load collection statistics from ChromaDB.

    Returns empty dict on failure so the page still renders.
    """
    try:
        from src.core.settings import load_settings
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        settings = load_settings("config/settings.yaml")
        store = VectorStoreFactory.create(settings)
        collections = store.list_collections() if hasattr(store, "list_collections") else []
        stats: dict[str, Any] = {}
        for name in collections:
            count = store.count(collection_name=name) if hasattr(store, "count") else "?"
            stats[name] = {"chunk_count": count}
        return stats
    except Exception:
        return {}


def _safe_trace_summary() -> dict[str, Any]:
    """Load a lightweight trace summary for the overview page."""
    try:
        traces = TraceService().list_traces(limit=500)
    except Exception:
        return {}

    query_count = sum(1 for trace in traces if trace.get("trace_type") == "query")
    ingestion_count = sum(1 for trace in traces if trace.get("trace_type") == "ingestion")
    latest_started = traces[0].get("started_at", "—") if traces else "—"

    return {
        "total": len(traces),
        "query_count": query_count,
        "ingestion_count": ingestion_count,
        "latest_started": latest_started,
    }


def render() -> None:
    """Render the Overview page."""
    render_page_header(
        title="Modular RAG Dashboard",
        subtitle=(
            "集中查看当前 RAG 系统的核心配置、索引概况与追踪活跃度，"
            "让调试、评估和运维信息更聚合。"
        ),
        badge="Observability Overview",
    )

    try:
        config_service = ConfigService()
        cards = config_service.get_component_cards()
    except Exception as exc:
        st.error(f"Failed to load configuration: {exc}")
        return

    stats = _safe_collection_stats()
    trace_summary = _safe_trace_summary()
    traces_path = Path("logs/traces.jsonl")

    render_metric_cards(
        [
            MetricCard(
                label="Configured Components",
                value=len(cards),
                caption="LLM / Embedding / Vector Store / Retrieval / Evaluation",
            ),
            MetricCard(
                label="Collections",
                value=len(stats),
                caption="Detected from vector store",
            ),
            MetricCard(
                label="Indexed Chunks",
                value=sum(info.get("chunk_count", 0) for info in stats.values()),
                caption="Across available collections",
            ),
            MetricCard(
                label="Trace Events",
                value=trace_summary.get("total", 0),
                caption=f"Latest: {trace_summary.get('latest_started', '—')}",
            ),
        ],
        columns=4,
    )

    render_section_header(
        "Component Configuration",
        "使用统一信息卡展示当前启用的模型、向量库和检索链路参数。",
    )

    config_columns = st.columns(min(len(cards), 3))
    for idx, card in enumerate(cards):
        with config_columns[idx % len(config_columns)]:
            extra_lines = [f"{key}: {value}" for key, value in card.extra.items()]
            render_component_card(
                title=card.name,
                eyebrow=card.provider,
                summary_lines=[f"Model: {card.model}", *extra_lines[:3]],
                pills=[f"{key}={value}" for key, value in list(card.extra.items())[3:]],
            )

    collection_col, trace_col = st.columns([1.2, 1])

    with collection_col:
        render_section_header(
            "Collection Statistics",
            "每个 collection 的切片规模一目了然，方便快速判断索引状态。",
        )
        if not stats:
            render_empty_state(
                "No collection data",
                "当前还没有可展示的 collection，先进行一次文档导入即可。",
            )
        else:
            rows = [
                {
                    "Collection": name,
                    "Chunks": info.get("chunk_count", "?"),
                }
                for name, info in sorted(stats.items())
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

    with trace_col:
        render_section_header(
            "Trace Activity",
            "聚合最近的 query / ingestion 执行记录，便于观察系统使用热度。",
        )
        if not trace_summary:
            render_empty_state(
                "No trace activity",
                "日志文件尚未生成，执行一次查询或导入后会出现在这里。",
            )
        else:
            render_metric_cards(
                [
                    MetricCard(
                        label="Query Traces",
                        value=trace_summary.get("query_count", 0),
                        caption="User-facing retrieval requests",
                    ),
                    MetricCard(
                        label="Ingestion Traces",
                        value=trace_summary.get("ingestion_count", 0),
                        caption="Document pipeline executions",
                    ),
                ],
                columns=2,
            )
            if traces_path.exists():
                st.caption(
                    f"Trace log path: `{traces_path}` · last activity: "
                    f"{trace_summary.get('latest_started', '—')}"
                )
