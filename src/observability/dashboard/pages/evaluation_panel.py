"""Evaluation Panel page – run evaluations and view metrics."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import streamlit as st

from src.observability.dashboard.ui import (
    MetricCard,
    render_empty_state,
    render_metric_cards,
    render_page_header,
    render_section_header,
)

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN_SET = Path("tests/fixtures/golden_test_set.json")
EVAL_HISTORY_PATH = Path("logs/eval_history.jsonl")


def _inspect_golden_set(path: Path) -> dict[str, Any]:
    """Return a lightweight summary of the configured golden dataset."""
    if not path.exists():
        return {"exists": False, "query_count": 0, "file_size_kb": 0.0}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"exists": True, "query_count": 0, "file_size_kb": path.stat().st_size / 1024}

    query_count = len(payload) if isinstance(payload, list) else 0
    return {
        "exists": True,
        "query_count": query_count,
        "file_size_kb": path.stat().st_size / 1024,
    }


def render() -> None:
    """Render the Evaluation Panel page."""
    render_page_header(
        title="Evaluation Panel",
        subtitle=(
            "运行 golden set 评测，统一查看聚合指标、逐条 query 结果和历史趋势，"
            "帮助你持续优化检索与生成效果。"
        ),
        badge="Quality Benchmarks",
    )

    history = _load_history()

    render_section_header(
        "Configuration",
        "选择评估后端、检索参数和 golden test set，界面会同步显示数据集可用性。",
    )

    config_col1, config_col2, config_col3 = st.columns(3)
    with config_col1:
        backend = st.selectbox(
            "Evaluator Backend",
            options=["custom", "ragas", "composite"],
            index=0,
            key="eval_backend",
            help="Select which evaluator backend to use.",
        )
    with config_col2:
        top_k = st.number_input(
            "Top-K",
            min_value=1,
            max_value=50,
            value=10,
            key="eval_top_k",
            help="Number of chunks to retrieve per query.",
        )
    with config_col3:
        collection = st.text_input(
            "Collection (optional)",
            value="",
            key="eval_collection",
            help="Limit retrieval to a specific collection.",
        )

    golden_path_str = st.text_input(
        "Golden Test Set Path",
        value=str(DEFAULT_GOLDEN_SET),
        key="eval_golden_path",
        help="Path to the golden_test_set.json file.",
    )
    golden_path = Path(golden_path_str)
    golden_info = _inspect_golden_set(golden_path)

    render_metric_cards(
        [
            MetricCard(
                label="Golden Set",
                value="Ready" if golden_info["exists"] else "Missing",
                caption=str(golden_path),
            ),
            MetricCard(
                label="Dataset Queries",
                value=golden_info["query_count"],
                caption="Entries parsed from the selected file",
            ),
            MetricCard(
                label="History Runs",
                value=len(history),
                caption="Stored in logs/eval_history.jsonl",
            ),
            MetricCard(
                label="Latest Evaluator",
                value=history[-1].get("evaluator_name", "—") if history else "—",
                caption=history[-1].get("timestamp", "No history yet") if history else "No history yet",
            ),
        ],
        columns=4,
    )

    if not golden_info["exists"]:
        st.warning(f"⚠️ Golden test set not found: `{golden_path}`")

    st.divider()

    run_clicked = st.button(
        "▶️ Run Evaluation",
        type="primary",
        key="eval_run_btn",
        disabled=not golden_info["exists"],
    )

    if run_clicked:
        _run_evaluation(
            backend=backend,
            golden_path=golden_path,
            top_k=int(top_k),
            collection=collection.strip() or None,
        )

    st.divider()
    _render_history(history)


def _run_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: str | None,
) -> None:
    """Execute an evaluation run and display results."""
    with st.spinner("Loading evaluator and running evaluation…"):
        try:
            report_dict = _execute_evaluation(
                backend=backend,
                golden_path=golden_path,
                top_k=top_k,
                collection=collection,
            )
        except Exception as exc:
            st.error(f"❌ Evaluation failed: {exc}")
            logger.exception("Evaluation failed")
            return

    st.success("✅ Evaluation complete!")
    _render_aggregate_metrics(report_dict)
    _render_query_details(report_dict)
    _save_to_history(report_dict)


def _execute_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: str | None,
) -> dict[str, Any]:
    """Run the evaluation pipeline and return the report dict."""
    from src.core.settings import load_settings
    from src.libs.evaluator.evaluator_factory import EvaluatorFactory
    from src.observability.evaluation.eval_runner import EvalRunner

    settings = load_settings()

    eval_settings = settings.evaluation
    override = type(eval_settings)(
        enabled=True,
        provider=backend,
        metrics=eval_settings.metrics if hasattr(eval_settings, "metrics") else [],
    )

    evaluator = EvaluatorFactory.create(override)
    hybrid_search = _try_create_hybrid_search(settings)

    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
    )

    report = runner.run(
        test_set_path=golden_path,
        top_k=top_k,
        collection=collection,
    )
    return report.to_dict()


def _try_create_hybrid_search(settings: Any) -> Any:
    """Attempt to create a HybridSearch instance."""
    try:
        from src.core.query_engine.hybrid_search import HybridSearch

        return HybridSearch(settings)
    except Exception as exc:
        logger.warning("Could not create HybridSearch: %s", exc)
        return None


def _render_aggregate_metrics(report: dict[str, Any]) -> None:
    """Display aggregate metrics as summary cards."""
    render_section_header(
        "Aggregate Metrics",
        "用统一 KPI 卡片展示本次评测的总体表现。",
    )

    agg = report.get("aggregate_metrics", {})
    if not agg:
        render_empty_state(
            "No aggregate metrics",
            "当前评估结果没有聚合指标输出。",
        )
        return

    render_metric_cards(
        [
            MetricCard(
                label=name.replace("_", " ").title(),
                value=f"{value:.4f}",
                caption="Aggregate score",
            )
            for name, value in sorted(agg.items())
        ],
        columns=min(len(agg), 4),
    )

    st.caption(
        f"Evaluator: **{report.get('evaluator_name', '—')}** · "
        f"Queries: **{report.get('query_count', 0)}** · "
        f"Total time: **{report.get('total_elapsed_ms', 0):.0f} ms**"
    )


def _render_query_details(report: dict[str, Any]) -> None:
    """Display per-query evaluation results in expandable cards."""
    render_section_header(
        "Per-Query Details",
        "逐条展开查看指标、召回 chunk ID 和生成答案。",
    )

    query_results = report.get("query_results", [])
    if not query_results:
        render_empty_state(
            "No query details",
            "当前评估没有逐条 query 的结果明细。",
        )
        return

    for idx, query_result in enumerate(query_results):
        query = query_result.get("query", "—")
        elapsed = query_result.get("elapsed_ms", 0)
        metrics = query_result.get("metrics", {})
        metric_summary = " · ".join(
            f"{name}: {value:.3f}" for name, value in sorted(metrics.items())
        ) or "no metrics"

        with st.expander(
            f"**Q{idx + 1}**: {query[:80]} — {elapsed:.0f} ms — {metric_summary}",
            expanded=False,
        ):
            if metrics:
                render_metric_cards(
                    [
                        MetricCard(label=name, value=f"{value:.4f}", caption="Per-query score")
                        for name, value in sorted(metrics.items())
                    ],
                    columns=min(len(metrics), 4),
                )

            detail_tab1, detail_tab2 = st.tabs(["Retrieved Chunks", "Generated Answer"])
            with detail_tab1:
                chunks = query_result.get("retrieved_chunk_ids", [])
                if chunks:
                    st.code(", ".join(chunks[:20]), language=None)
                else:
                    render_empty_state(
                        "No retrieved chunks",
                        "当前 query 结果没有记录 chunk ID。",
                    )
            with detail_tab2:
                answer = query_result.get("generated_answer")
                if answer:
                    st.text(answer[:1000])
                else:
                    render_empty_state(
                        "No generated answer",
                        "当前 query 结果没有返回生成答案。",
                    )


def _render_history(history: list[dict[str, Any]] | None = None) -> None:
    """Display historical evaluation results for comparison."""
    render_section_header(
        "Evaluation History",
        "保留最近多次评估结果，便于横向比较不同配置的表现。",
    )

    history_entries = history if history is not None else _load_history()
    if not history_entries:
        render_empty_state(
            "No history yet",
            "运行一次 evaluation 后，这里会开始记录历史趋势。",
        )
        return

    recent = history_entries[-10:]
    rows = []
    for entry in recent:
        rows.append(
            {
                "Timestamp": entry.get("timestamp", "—"),
                "Evaluator": entry.get("evaluator_name", "—"),
                "Queries": entry.get("query_count", 0),
                "Time (ms)": round(entry.get("total_elapsed_ms", 0)),
                **{
                    key: round(value, 4)
                    for key, value in entry.get("aggregate_metrics", {}).items()
                },
            }
        )

    render_metric_cards(
        [
            MetricCard(
                label="Recorded Runs",
                value=len(history_entries),
                caption="Persisted evaluation snapshots",
            ),
            MetricCard(
                label="Latest Run",
                value=history_entries[-1].get("timestamp", "—"),
                caption=history_entries[-1].get("evaluator_name", "—"),
            ),
        ],
        columns=2,
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _save_to_history(report: dict[str, Any]) -> None:
    """Append an evaluation report to the history file."""
    try:
        EVAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **report,
        }
        with EVAL_HISTORY_PATH.open("a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to save evaluation history: %s", exc)


def _load_history() -> list[dict[str, Any]]:
    """Load evaluation history from JSONL file."""
    if not EVAL_HISTORY_PATH.exists():
        return []

    entries: list[dict[str, Any]] = []
    try:
        with EVAL_HISTORY_PATH.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        logger.warning("Failed to load evaluation history: %s", exc)

    return entries
