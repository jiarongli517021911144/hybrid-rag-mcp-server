"""Ingestion Manager page – upload files, trigger ingestion, delete documents.

Layout:
1. Upload panel with collection targeting
2. Progress feedback during ingestion
3. Searchable document management list
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
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

SUPPORTED_TYPES = ["pdf", "txt", "md", "docx"]


def _run_ingestion(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    collection: str,
    progress_bar: st.delta_generator.DeltaGenerator,
    status_text: st.delta_generator.DeltaGenerator,
) -> None:
    """Save the uploaded file to a temp location and run the pipeline."""
    from src.core.settings import load_settings
    from src.core.trace import TraceCollector, TraceContext
    from src.ingestion.pipeline import IngestionPipeline

    settings = load_settings("config/settings.yaml")

    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    def on_progress(stage: str, current: int, total: int) -> None:
        progress_bar.progress(current / total, text=f"Stage {current}/{total}: {stage}")
        status_text.caption(f"Processing: {stage} …")

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["source_path"] = uploaded_file.name
    trace.metadata["collection"] = collection
    trace.metadata["source"] = "dashboard"

    try:
        pipeline = IngestionPipeline(settings, collection=collection)
        result = pipeline.run(
            file_path=tmp_path,
            trace=trace,
            on_progress=on_progress,
        )
        if result.success:
            progress_bar.progress(1.0, text="✅ Complete")
            status_text.success(
                f"Successfully ingested **{uploaded_file.name}** into collection **{collection}**."
            )
        else:
            progress_bar.progress(1.0, text="❌ Failed")
            status_text.error(f"Ingestion failed: {result.error or 'unknown error'}")
    except Exception as exc:
        progress_bar.progress(1.0, text="❌ Failed")
        status_text.error(f"Ingestion failed: {exc}")
    finally:
        TraceCollector().collect(trace)
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _delete_document(doc: dict[str, Any]) -> None:
    """Delete a document from all backing stores."""
    from src.core.settings import load_settings
    from src.ingestion.document_manager import DocumentManager
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.ingestion.storage.image_storage import ImageStorage
    from src.libs.loader.file_integrity import SQLiteIntegrityChecker
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    settings = load_settings("config/settings.yaml")
    collection = doc.get("collection", "default")
    chroma = VectorStoreFactory.create(settings, collection_name=collection)
    bm25 = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
    images = ImageStorage(
        db_path="data/db/image_index.db",
        images_root="data/images",
    )
    integrity = SQLiteIntegrityChecker(db_path="data/db/ingestion_history.db")
    manager = DocumentManager(chroma, bm25, images, integrity)
    result = manager.delete_document(
        doc["source_path"],
        collection,
    )
    if result.success:
        st.success(
            f"Deleted: {result.chunks_deleted} chunks, {result.images_deleted} images removed."
        )
        st.rerun()
    else:
        st.warning(f"Partial delete. Errors: {result.errors}")


def _filter_documents(
    docs: list[dict[str, Any]], collection_filter: str, keyword: str
) -> list[dict[str, Any]]:
    """Apply collection and keyword filters to document rows."""
    collection_value = collection_filter.strip().lower()
    keyword_value = keyword.strip().lower()

    filtered = docs
    if collection_value:
        filtered = [
            doc for doc in filtered if collection_value in str(doc.get("collection", "")).lower()
        ]
    if keyword_value:
        filtered = [
            doc
            for doc in filtered
            if keyword_value in str(doc.get("source_path", "")).lower()
            or keyword_value in str(doc.get("processed_at", "")).lower()
        ]
    return filtered


def render() -> None:
    """Render the Ingestion Manager page."""
    render_page_header(
        title="Ingestion Manager",
        subtitle=(
            "上传新文档、观察处理进度，并在同一页面里清理历史导入内容。"
        ),
        badge="Document Operations",
    )

    try:
        svc = DataService()
        docs = svc.list_documents()
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    render_metric_cards(
        [
            MetricCard(
                label="Indexed Documents",
                value=len(docs),
                caption="Across currently visible stores",
            ),
            MetricCard(
                label="Indexed Chunks",
                value=sum(doc.get("chunk_count", 0) for doc in docs),
                caption="Total vectorized slices",
            ),
            MetricCard(
                label="Extracted Images",
                value=sum(doc.get("image_count", 0) for doc in docs),
                caption="Images saved during ingestion",
            ),
        ],
        columns=3,
    )

    render_section_header(
        "Upload & Ingest",
        "保持现有流程不变，但将上传信息、目标 collection 和状态反馈集中在一处。",
    )

    upload_col, target_col = st.columns([3, 1.2])
    with upload_col:
        uploaded = st.file_uploader(
            "Select a file to ingest",
            type=SUPPORTED_TYPES,
            key="ingest_uploader",
        )
        st.caption("Supported formats: PDF, TXT, Markdown and DOCX")
    with target_col:
        collection = st.text_input("Target collection", value="default", key="ingest_collection")
        st.caption("Use a stable collection name to keep related documents together.")

    if uploaded is not None:
        with st.container(border=True):
            file_size_kb = len(uploaded.getbuffer()) / 1024
            st.markdown(f"**Ready to ingest:** `{uploaded.name}`")
            info_a, info_b, info_c = st.columns(3)
            with info_a:
                st.metric("File type", Path(uploaded.name).suffix or "—")
            with info_b:
                st.metric("File size", f"{file_size_kb:.1f} KB")
            with info_c:
                st.metric("Collection", collection.strip() or "default")

        if st.button("🚀 Start Ingestion", key="btn_ingest", type="primary"):
            progress_bar = st.progress(0, text="Preparing…")
            status_text = st.empty()
            _run_ingestion(uploaded, collection.strip() or "default", progress_bar, status_text)

    st.divider()

    render_section_header(
        "Manage Documents",
        "支持按 collection 和关键词过滤，方便删除过期或错误导入的文件。",
    )

    filter_col1, filter_col2 = st.columns([1.2, 1.8])
    with filter_col1:
        collection_filter = st.text_input(
            "Filter by collection",
            value="",
            key="ingest_manage_collection_filter",
            placeholder="e.g. default",
        )
    with filter_col2:
        keyword = st.text_input(
            "Search documents",
            value="",
            key="ingest_manage_keyword",
            placeholder="Search by file path or processed time…",
        )

    filtered_docs = _filter_documents(docs, collection_filter, keyword)

    if not docs:
        render_empty_state(
            "No documents yet",
            "还没有可管理的导入记录，先上传一个文件试试看。",
        )
        return

    if not filtered_docs:
        render_empty_state(
            "No matching documents",
            "筛选条件下没有命中文档，试试清空关键词或 collection 过滤器。",
        )
        return

    for idx, doc in enumerate(filtered_docs):
        file_name = Path(doc.get("source_path", "")).name or doc.get("source_path", "—")
        row_left, row_right = st.columns([5, 1.2])
        with row_left:
            with st.container(border=True):
                st.markdown(f"**{file_name}**")
                st.caption(
                    f"Path: `{doc.get('source_path', '—')}` · "
                    f"Collection: `{doc.get('collection', '—')}` · "
                    f"Processed: {doc.get('processed_at', '—')}"
                )
                stat_a, stat_b = st.columns(2)
                with stat_a:
                    st.metric("Chunks", doc.get("chunk_count", 0))
                with stat_b:
                    st.metric("Images", doc.get("image_count", 0))
        with row_right:
            st.write("")
            st.write("")
            if st.button("🗑️ Delete", key=f"del_{idx}"):
                try:
                    _delete_document(doc)
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
