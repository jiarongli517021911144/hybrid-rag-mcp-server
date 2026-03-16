"""Data Browser page – browse ingested documents, chunks, and images.

Layout:
1. Collection selector + document filter
2. Summary metrics
3. Document detail cards with tabs for chunks and images
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.observability.dashboard.services.data_service import DataService
from src.observability.dashboard.ui import (
    MetricCard,
    render_empty_state,
    render_metric_cards,
    render_page_header,
    render_section_header,
)


def _matches_keyword(doc: dict[str, Any], keyword: str) -> bool:
    """Return whether a document matches the current keyword filter."""
    if not keyword:
        return True
    haystack = " ".join(
        [
            str(doc.get("source_path", "")),
            str(doc.get("collection", "")),
            str(doc.get("processed_at", "")),
        ]
    ).lower()
    return keyword.lower() in haystack


def _sort_documents(docs: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    """Return documents in the selected sort order."""
    if sort_by == "Most chunks":
        return sorted(docs, key=lambda doc: doc.get("chunk_count", 0), reverse=True)
    if sort_by == "Most images":
        return sorted(docs, key=lambda doc: doc.get("image_count", 0), reverse=True)
    if sort_by == "Name":
        return sorted(docs, key=lambda doc: Path(doc.get("source_path", "")).name.lower())
    return sorted(docs, key=lambda doc: str(doc.get("processed_at", "")), reverse=True)


def render() -> None:
    """Render the Data Browser page."""
    render_page_header(
        title="Data Browser",
        subtitle=(
            "按 collection、文件名和内容规模浏览已导入文档，快速查看 chunk、图片"
            "和文档元数据。"
        ),
        badge="Indexed Content Explorer",
    )

    try:
        svc = DataService()
    except Exception as exc:
        st.error(f"Failed to initialise DataService: {exc}")
        return

    control_col1, control_col2, control_col3 = st.columns([1.4, 1.4, 1])
    with control_col1:
        collection = st.text_input(
            "Collection name",
            value="default",
            key="db_collection_filter",
            help="Leave blank to use the default collection.",
        )
    with control_col2:
        keyword = st.text_input(
            "Filter documents",
            value="",
            key="db_keyword_filter",
            placeholder="Search by path or collection…",
        )
    with control_col3:
        sort_by = st.selectbox(
            "Sort by",
            options=["Newest", "Most chunks", "Most images", "Name"],
            key="db_sort_by",
        )

    coll_arg = collection.strip() if collection.strip() else None

    try:
        docs = svc.list_documents(coll_arg)
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    filtered_docs = [doc for doc in docs if _matches_keyword(doc, keyword.strip())]
    filtered_docs = _sort_documents(filtered_docs, sort_by)

    if docs:
        total_chunks = sum(doc.get("chunk_count", 0) for doc in docs)
        total_images = sum(doc.get("image_count", 0) for doc in docs)
        render_metric_cards(
            [
                MetricCard(
                    label="Documents",
                    value=len(filtered_docs),
                    delta=(
                        f"{len(filtered_docs) - len(docs):+d} filtered"
                        if len(filtered_docs) != len(docs)
                        else None
                    ),
                    caption=f"Total available: {len(docs)}",
                ),
                MetricCard(
                    label="Chunks",
                    value=total_chunks,
                    caption=f"Collection: {coll_arg or 'default'}",
                ),
                MetricCard(
                    label="Images",
                    value=total_images,
                    caption="Extracted during ingestion",
                ),
            ],
            columns=3,
        )

    render_section_header(
        "Document Library",
        "打开单个文档即可查看切片内容、图片预览和处理时间等细节。",
    )

    if not docs:
        render_empty_state(
            "No documents found",
            "当前 collection 中还没有可浏览的文档，请先完成一次导入。",
        )
        return

    if not filtered_docs:
        render_empty_state(
            "No matches",
            "当前筛选条件没有命中文档，试试清空关键词或切换排序方式。",
        )
        return

    for idx, doc in enumerate(filtered_docs):
        source_name = Path(doc["source_path"]).name
        label = (
            f"📄 {source_name} · {doc['chunk_count']} chunks · "
            f"{doc['image_count']} images"
        )
        with st.expander(label, expanded=(idx == 0 and len(filtered_docs) <= 3)):
            top_left, top_mid, top_right = st.columns(3)
            with top_left:
                st.metric("Chunks", doc["chunk_count"])
            with top_mid:
                st.metric("Images", doc["image_count"])
            with top_right:
                st.metric("Collection", doc.get("collection", "—"))

            st.caption(
                f"Source: `{doc['source_path']}` · "
                f"Hash: `{doc['source_hash'][:16]}…` · "
                f"Processed: {doc.get('processed_at', '—')}"
            )

            tab_chunks, tab_images, tab_meta = st.tabs(["Chunks", "Images", "Metadata"])

            with tab_chunks:
                chunks = svc.get_chunks(doc["source_hash"], coll_arg)
                if not chunks:
                    render_empty_state(
                        "No chunks available",
                        "向量库中暂未找到此文档对应的 chunk 数据。",
                    )
                else:
                    st.caption(f"Showing {len(chunks)} chunk(s) for this document")
                    for cidx, chunk in enumerate(chunks):
                        text = chunk.get("text", "")
                        meta = chunk.get("metadata", {})
                        with st.container(border=True):
                            st.markdown(
                                f"**Chunk {cidx + 1}** · `{chunk['id'][-16:]}` · {len(text)} chars"
                            )
                            st.text_area(
                                "Content",
                                value=text,
                                height=min(max(120, len(text) // 3), 320),
                                disabled=True,
                                key=f"chunk_text_{idx}_{cidx}",
                                label_visibility="collapsed",
                            )
                            if meta:
                                with st.expander("Chunk metadata", expanded=False):
                                    st.json(meta)

            with tab_images:
                images = svc.get_images(doc["source_hash"], coll_arg)
                if not images:
                    render_empty_state(
                        "No images extracted",
                        "这个文档没有关联图片，或图片文件尚未落盘。",
                    )
                else:
                    img_cols = st.columns(min(len(images), 3))
                    for iidx, img in enumerate(images):
                        with img_cols[iidx % len(img_cols)]:
                            img_path = Path(img.get("file_path", ""))
                            if img_path.exists():
                                st.image(str(img_path), caption=img["image_id"], width=220)
                            else:
                                st.caption(f"{img['image_id']} (file missing)")

            with tab_meta:
                st.json(
                    {
                        "source_path": doc.get("source_path"),
                        "collection": doc.get("collection"),
                        "source_hash": doc.get("source_hash"),
                        "chunk_count": doc.get("chunk_count"),
                        "image_count": doc.get("image_count"),
                        "processed_at": doc.get("processed_at"),
                    }
                )
