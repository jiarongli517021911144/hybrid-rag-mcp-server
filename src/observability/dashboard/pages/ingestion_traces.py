"""Ingestion Traces page – browse ingestion trace history with stage waterfall."""

from __future__ import annotations

from statistics import mean

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


def render() -> None:
    """Render the Ingestion Traces page."""
    render_page_header(
        title="Ingestion Traces",
        subtitle=(
            "按时间回看导入流水线的阶段耗时，快速定位解析、切分、嵌入或写入瓶颈。"
        ),
        badge="Pipeline Performance",
    )

    svc = TraceService()
    traces = svc.list_traces(trace_type="ingestion")

    if not traces:
        render_empty_state(
            "No ingestion traces",
            "还没有记录到导入流水线执行，完成一次 ingestion 后这里会自动出现。",
        )
        return

    limit = st.selectbox(
        "Show recent traces",
        options=[5, 10, 20, 50, 100],
        index=1,
        key="ingestion_trace_limit",
    )
    visible_traces = traces[:limit]
    elapsed_values = [trace.get("elapsed_ms") for trace in visible_traces if trace.get("elapsed_ms") is not None]

    render_metric_cards(
        [
            MetricCard(
                label="Visible Traces",
                value=len(visible_traces),
                caption=f"Total recorded: {len(traces)}",
            ),
            MetricCard(
                label="Average Duration",
                value=format_ms(mean(elapsed_values)) if elapsed_values else "—",
                caption="Based on current view",
            ),
            MetricCard(
                label="Slowest Run",
                value=format_ms(max(elapsed_values)) if elapsed_values else "—",
                caption="Useful for spotting outliers",
            ),
        ],
        columns=3,
    )

    render_section_header(
        "Trace History",
        "展开单条 trace 可以同时看到阶段瀑布图、精确耗时表和原始 metadata。",
    )

    for idx, trace in enumerate(visible_traces):
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        total_label = format_ms(trace.get("elapsed_ms"))
        timings = svc.get_stage_timings(trace)

        with st.expander(
            f"**{trace_id[:12]}…** — {started} — total: {total_label}",
            expanded=(idx == 0),
        ):
            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.metric("Stages", len(timings))
            with summary_cols[1]:
                st.metric("Total", total_label)
            with summary_cols[2]:
                st.metric("Trace Type", trace.get("trace_type", "—"))

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
                    "No stage timings",
                    "该 trace 没有阶段耗时明细，可能是旧日志或中途失败的执行。",
                )

            meta = trace.get("metadata", {})
            if meta:
                with st.expander("Metadata", expanded=False):
                    st.json(meta)
