"""Unit tests for the dashboard QueryService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.core.query_engine.hybrid_search import HybridSearchResult
from src.core.query_engine.reranker import RerankResult
from src.core.response.response_builder import MCPToolResponse
from src.core.types import ProcessedQuery, RetrievalResult
from src.observability.dashboard.services.query_service import QueryService


def _make_result(chunk_id: str, score: float) -> RetrievalResult:
    """Build a minimal retrieval result for tests."""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"text for {chunk_id}",
        metadata={"source_path": f"docs/{chunk_id}.pdf", "chunk_index": 0},
    )


def _make_service() -> QueryService:
    """Create a QueryService with injected mock components."""
    service = QueryService(settings=MagicMock(), response_builder=MagicMock())
    service._initialized = True
    service._current_collection = "default"
    service._hybrid_search = MagicMock()
    service._reranker = MagicMock()
    service._reranker.is_enabled = True
    service._reranker.reranker_type = "cross_encoder"
    return service


class TestQueryService:
    """Verify dashboard query orchestration."""

    @patch("src.observability.dashboard.services.query_service.TraceCollector")
    def test_run_query_with_rerank(self, mock_trace_collector: MagicMock) -> None:
        """Service should search, rerank, and build a response."""
        service = _make_service()
        dense_results = [_make_result("dense_1", 0.9), _make_result("dense_2", 0.8)]
        sparse_results = [_make_result("sparse_1", 0.7)]
        reranked_results = [dense_results[1], dense_results[0]]

        service._hybrid_search.search.return_value = HybridSearchResult(
            results=dense_results,
            dense_results=dense_results,
            sparse_results=sparse_results,
            processed_query=ProcessedQuery(
                original_query="naval",
                keywords=["naval", "wealth"],
                filters={},
            ),
        )
        service._reranker.rerank.return_value = RerankResult(
            results=reranked_results,
            reranker_type="cross_encoder",
        )
        service._response_builder.build.return_value = MCPToolResponse(content="ok")

        result = service.run_query(
            query="naval",
            collection="default",
            top_k=2,
            use_rerank=True,
        )

        assert result.rerank_applied is True
        assert result.reranker_type == "cross_encoder"
        assert result.final_results == reranked_results
        assert result.processed_keywords == ["naval", "wealth"]
        service._hybrid_search.search.assert_called_once_with(
            query="naval",
            top_k=4,
            filters=None,
            trace=service._hybrid_search.search.call_args.kwargs["trace"],
            return_details=True,
        )
        service._reranker.rerank.assert_called_once()
        service._response_builder.build.assert_called_once_with(
            results=reranked_results,
            query="naval",
            collection="default",
            include_images=False,
        )
        mock_trace_collector.return_value.collect.assert_called_once()

    @patch("src.observability.dashboard.services.query_service.TraceCollector")
    def test_run_query_without_rerank(self, mock_trace_collector: MagicMock) -> None:
        """Service should skip reranking when disabled from the UI."""
        service = _make_service()
        hybrid_results = [_make_result("dense_1", 0.9), _make_result("dense_2", 0.8)]
        service._hybrid_search.search.return_value = HybridSearchResult(
            results=hybrid_results,
            dense_results=hybrid_results,
            sparse_results=[],
            used_fallback=True,
            dense_error=None,
            sparse_error="bm25 missing",
            processed_query=ProcessedQuery(original_query="test", keywords=["test"], filters={}),
        )
        service._response_builder.build.return_value = MCPToolResponse(content="ok")

        result = service.run_query(
            query="test",
            collection="default",
            top_k=1,
            use_rerank=False,
        )

        assert result.rerank_applied is False
        assert result.used_fallback is True
        assert result.sparse_error == "bm25 missing"
        assert result.final_results == hybrid_results[:1]
        service._reranker.rerank.assert_not_called()
        service._response_builder.build.assert_called_once_with(
            results=hybrid_results[:1],
            query="test",
            collection="default",
            include_images=False,
        )
        mock_trace_collector.return_value.collect.assert_called_once()


def test_query_playground_page_importable() -> None:
    """The new dashboard page should be importable."""
    from src.observability.dashboard.pages import query_playground

    assert hasattr(query_playground, "render")
    assert callable(query_playground.render)
