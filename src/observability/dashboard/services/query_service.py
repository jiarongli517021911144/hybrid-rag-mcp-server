"""Query service for the dashboard's interactive playground.

This module exposes a lightweight facade over the existing hybrid retrieval
stack so Streamlit pages can execute searches without duplicating the query
orchestration logic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.hybrid_search import HybridSearchResult, create_hybrid_search
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.reranker import CoreReranker, RerankResult, create_core_reranker
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.core.response.response_builder import MCPToolResponse, ResponseBuilder
from src.core.settings import Settings, load_settings
from src.core.trace import TraceCollector, TraceContext
from src.core.types import RetrievalResult
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)


@dataclass
class QueryExecutionResult:
    """Structured result returned to the dashboard UI."""

    query: str
    collection: str
    top_k: int
    response: MCPToolResponse
    final_results: list[RetrievalResult] = field(default_factory=list)
    dense_results: list[RetrievalResult] = field(default_factory=list)
    sparse_results: list[RetrievalResult] = field(default_factory=list)
    processed_keywords: list[str] = field(default_factory=list)
    dense_error: str | None = None
    sparse_error: str | None = None
    used_fallback: bool = False
    rerank_applied: bool = False
    rerank_used_fallback: bool = False
    rerank_fallback_reason: str | None = None
    reranker_type: str = "none"
    elapsed_ms: float = 0.0


class QueryService:
    """Lazy query facade for Streamlit pages."""

    def __init__(
        self,
        settings: Settings | None = None,
        response_builder: ResponseBuilder | None = None,
    ) -> None:
        self._settings = settings
        self._response_builder = response_builder or ResponseBuilder()
        self._hybrid_search: Any = None
        self._reranker: CoreReranker | None = None
        self._initialized = False
        self._current_collection: str | None = None

    @property
    def settings(self) -> Settings:
        """Return settings, loading them lazily if needed."""
        if self._settings is None:
            self._settings = load_settings("config/settings.yaml")
        return self._settings

    def _ensure_initialized(self, collection: str) -> None:
        """Build query components for the selected collection."""
        if self._initialized and self._current_collection == collection:
            return

        logger.info("Initializing QueryService components for collection=%s", collection)
        vector_store = VectorStoreFactory.create(self.settings, collection_name=collection)
        embedding_client = EmbeddingFactory.create(self.settings)
        dense_retriever = create_dense_retriever(
            settings=self.settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        sparse_retriever = create_sparse_retriever(
            settings=self.settings,
            bm25_indexer=BM25Indexer(index_dir=f"data/db/bm25/{collection}"),
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection

        self._hybrid_search = create_hybrid_search(
            settings=self.settings,
            query_processor=QueryProcessor(),
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )
        self._reranker = create_core_reranker(settings=self.settings)
        self._initialized = True
        self._current_collection = collection

    def run_query(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 10,
        use_rerank: bool = True,
    ) -> QueryExecutionResult:
        """Execute a query and return final plus debug retrieval details."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query cannot be empty")

        effective_collection = collection.strip() or "default"
        effective_top_k = max(1, int(top_k))
        self._ensure_initialized(effective_collection)

        if self._hybrid_search is None:
            raise RuntimeError("HybridSearch is not initialized")

        trace = TraceContext(trace_type="query")
        trace.metadata["query"] = cleaned_query[:200]
        trace.metadata["collection"] = effective_collection
        trace.metadata["top_k"] = effective_top_k
        trace.metadata["source"] = "dashboard_playground"

        started = time.monotonic()
        try:
            initial_top_k = effective_top_k * 2 if use_rerank else effective_top_k
            hybrid_result = self._hybrid_search.search(
                query=cleaned_query,
                top_k=initial_top_k,
                filters=None,
                trace=trace,
                return_details=True,
            )
            if not isinstance(hybrid_result, HybridSearchResult):
                hybrid_result = HybridSearchResult(results=hybrid_result)

            final_results = hybrid_result.results[:effective_top_k]
            rerank_result: RerankResult | None = None
            rerank_applied = False
            reranker_type = self._reranker.reranker_type if self._reranker else "none"

            if use_rerank and self._reranker is not None and self._reranker.is_enabled and final_results:
                rerank_result = self._reranker.rerank(
                    query=cleaned_query,
                    results=hybrid_result.results,
                    top_k=effective_top_k,
                    trace=trace,
                )
                final_results = rerank_result.results
                rerank_applied = True
                reranker_type = rerank_result.reranker_type

            response = self._response_builder.build(
                results=final_results,
                query=cleaned_query,
                collection=effective_collection,
                include_images=False,
            )
        except Exception:
            TraceCollector().collect(trace)
            raise

        elapsed_ms = (time.monotonic() - started) * 1000.0
        TraceCollector().collect(trace)

        return QueryExecutionResult(
            query=cleaned_query,
            collection=effective_collection,
            top_k=effective_top_k,
            response=response,
            final_results=final_results,
            dense_results=hybrid_result.dense_results or [],
            sparse_results=hybrid_result.sparse_results or [],
            processed_keywords=(hybrid_result.processed_query.keywords if hybrid_result.processed_query else []),
            dense_error=hybrid_result.dense_error,
            sparse_error=hybrid_result.sparse_error,
            used_fallback=hybrid_result.used_fallback,
            rerank_applied=rerank_applied,
            rerank_used_fallback=rerank_result.used_fallback if rerank_result else False,
            rerank_fallback_reason=rerank_result.fallback_reason if rerank_result else None,
            reranker_type=reranker_type,
            elapsed_ms=elapsed_ms,
        )
