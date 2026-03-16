"""Shared UI helpers for the Streamlit dashboard.

Provides lightweight theming and reusable page primitives so the
dashboard pages can share a more cohesive visual language without
duplicating markup.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

import streamlit as st


@dataclass(frozen=True)
class MetricCard:
    """UI-friendly metric card definition."""

    label: str
    value: str | int | float
    delta: str | int | float | None = None
    caption: str | None = None
    help_text: str | None = None


_THEME_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.14), transparent 26%),
            radial-gradient(circle at top right, rgba(16, 185, 129, 0.10), transparent 22%),
            linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(255, 255, 255, 1));
    }

    .block-container {
        max-width: 1480px;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
    }

    .dashboard-hero {
        position: relative;
        overflow: hidden;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.25rem;
        border-radius: 22px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.92));
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
    }

    .dashboard-hero::after {
        content: "";
        position: absolute;
        inset: auto -2rem -2rem auto;
        width: 180px;
        height: 180px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(96, 165, 250, 0.35), transparent 70%);
    }

    .dashboard-hero__badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        margin-bottom: 0.9rem;
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        background: rgba(96, 165, 250, 0.16);
        border: 1px solid rgba(147, 197, 253, 0.24);
        color: #dbeafe;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .dashboard-hero h1 {
        margin: 0;
        color: #f8fafc;
        font-size: 2.05rem;
        line-height: 1.15;
        letter-spacing: -0.03em;
    }

    .dashboard-hero p {
        margin: 0.65rem 0 0;
        max-width: 860px;
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.6;
    }

    .dashboard-section {
        margin: 1.2rem 0 0.6rem;
    }

    .dashboard-section h2 {
        margin: 0;
        color: #0f172a;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    .dashboard-section p {
        margin: 0.35rem 0 0;
        color: #64748b;
        font-size: 0.95rem;
    }

    .dashboard-card {
        padding: 1rem 1.05rem;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(8px);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        min-height: 188px;
    }

    .dashboard-card__eyebrow {
        color: #3b82f6;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .dashboard-card__title {
        margin-top: 0.45rem;
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 700;
    }

    .dashboard-card__meta {
        margin-top: 0.5rem;
        color: #475569;
        font-size: 0.94rem;
        line-height: 1.55;
    }

    .dashboard-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.85rem;
    }

    .dashboard-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.62rem;
        border-radius: 999px;
        background: rgba(241, 245, 249, 0.95);
        color: #334155;
        font-size: 0.78rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
    }

    .dashboard-empty {
        padding: 1rem 1.05rem;
        border-radius: 18px;
        border: 1px dashed rgba(148, 163, 184, 0.42);
        background: rgba(255, 255, 255, 0.56);
        color: #475569;
    }

    .dashboard-empty strong {
        display: block;
        margin-bottom: 0.35rem;
        color: #0f172a;
        font-size: 0.98rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 700;
    }

    div[data-testid="stExpander"] {
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.20);
        background: rgba(255, 255, 255, 0.72);
        overflow: hidden;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stExpander"] summary {
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
        margin-bottom: 0.6rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        background: rgba(248, 250, 252, 0.85);
        padding: 0.38rem 0.8rem;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(37, 99, 235, 0.18));
        color: #1d4ed8;
        border-color: rgba(59, 130, 246, 0.24);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        font-weight: 600;
    }

    .stButton > button[kind="primary"] {
        border: none;
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        box-shadow: 0 14px 32px rgba(37, 99, 235, 0.24);
    }

    div[data-testid="stFileUploader"] {
        border-radius: 18px;
        border: 1px dashed rgba(96, 165, 250, 0.38);
        background: rgba(239, 246, 255, 0.62);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
    }

    .stAlert {
        border-radius: 16px;
    }
</style>
"""


def apply_dashboard_theme() -> None:
    """Inject a consistent visual theme into the dashboard."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, badge: str | None = None) -> None:
    """Render a shared hero header for each dashboard page."""
    apply_dashboard_theme()
    badge_html = (
        f'<div class="dashboard-hero__badge">{escape(badge)}</div>'
        if badge
        else ""
    )
    st.markdown(
        (
            '<section class="dashboard-hero">'
            f"{badge_html}"
            f"<h1>{escape(title)}</h1>"
            f"<p>{escape(subtitle)}</p>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str | None = None) -> None:
    """Render a lightweight section heading with optional description."""
    desc_html = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(
        f'<div class="dashboard-section"><h2>{escape(title)}</h2>{desc_html}</div>',
        unsafe_allow_html=True,
    )


def render_metric_cards(cards: Sequence[MetricCard], columns: int | None = None) -> None:
    """Render a responsive grid of native Streamlit metric cards."""
    if not cards:
        return

    column_count = max(1, min(columns or len(cards), len(cards), 4))
    grid = st.columns(column_count)
    for index, card in enumerate(cards):
        with grid[index % column_count]:
            st.metric(
                label=card.label,
                value=card.value,
                delta=card.delta,
                help=card.help_text,
            )
            if card.caption:
                st.caption(card.caption)


def render_component_card(
    title: str,
    eyebrow: str,
    summary_lines: Sequence[str],
    pills: Sequence[str] | None = None,
) -> None:
    """Render a reusable translucent info card."""
    pill_html = ""
    if pills:
        rendered = "".join(
            f'<span class="dashboard-pill">{escape(pill)}</span>'
            for pill in pills
            if pill
        )
        if rendered:
            pill_html = f'<div class="dashboard-pill-row">{rendered}</div>'

    line_html = "<br/>".join(escape(line) for line in summary_lines if line)
    st.markdown(
        (
            '<section class="dashboard-card">'
            f'<div class="dashboard-card__eyebrow">{escape(eyebrow)}</div>'
            f'<div class="dashboard-card__title">{escape(title)}</div>'
            f'<div class="dashboard-card__meta">{line_html}</div>'
            f"{pill_html}"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, message: str) -> None:
    """Render a lightweight empty-state panel."""
    st.markdown(
        (
            '<section class="dashboard-empty">'
            f"<strong>{escape(title)}</strong>"
            f"<span>{escape(message)}</span>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def format_ms(value: Any) -> str:
    """Format a millisecond value for consistent display."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.0f} ms"
    except (TypeError, ValueError):
        return str(value)
