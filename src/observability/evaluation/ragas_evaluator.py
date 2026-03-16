"""Ragas-based evaluator for RAG quality assessment.

This evaluator wraps the Ragas framework to compute LLM-as-Judge metrics:
- Faithfulness: Does the answer stick to the retrieved context?
- Answer Relevancy: Is the answer relevant to the query?
- Context Precision: Are the retrieved chunks relevant and well-ordered?

Design Principles:
- Pluggable: Implements BaseEvaluator interface, swappable via factory.
- Config-Driven: LLM/Embedding backend read from settings.yaml.
- Graceful Degradation: Clear ImportError if ragas not installed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Metric name constants
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"

SUPPORTED_METRICS = {FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION}


def _import_ragas() -> None:
    """Validate that ragas is importable, raising a clear error if not."""
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ragas' package is required for RagasEvaluator. "
            "Install it with: pip install ragas datasets"
        ) from exc


class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses the Ragas framework for LLM-as-Judge metrics.

    Ragas does NOT require ground-truth labels.  It uses an LLM to judge
    the quality of the generated answer against the retrieved context.

    Supported metrics:
        - faithfulness: Measures factual consistency with context.
        - answer_relevancy: Measures how relevant the answer is to the query.
        - context_precision: Measures relevance/ordering of retrieved chunks.

    Example::

        evaluator = RagasEvaluator(settings=settings)
        metrics = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"id": "c1", "text": "RAG is ..."}],
            generated_answer="RAG stands for ...",
        )
        # metrics == {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}
    """

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RagasEvaluator.

        Args:
            settings: Application settings (used to configure LLM backend).
            metrics: Metric names to compute. Defaults to all supported.
            **kwargs: Additional parameters (reserved).

        Raises:
            ImportError: If ragas is not installed.
            ValueError: If unsupported metric names are requested.
        """
        _import_ragas()

        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalised = [m.strip().lower() for m in (metrics or [])]
        if not normalised:
            normalised = sorted(SUPPORTED_METRICS)

        unsupported = [m for m in normalised if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )

        self._metric_names = normalised

    # ── public API ────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate RAG quality using Ragas LLM-as-Judge metrics.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks (dicts with 'text' key or strings).
            generated_answer: The generated answer text. Required for Ragas.
            ground_truth: Ignored by Ragas (not needed for LLM-as-Judge).
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters.

        Returns:
            Dictionary mapping metric names to float scores (0.0 – 1.0).

        Raises:
            ValueError: If query/chunks are invalid or generated_answer is missing.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        if not generated_answer or not generated_answer.strip():
            raise ValueError(
                "RagasEvaluator requires a non-empty 'generated_answer'. "
                "Ragas uses LLM-as-Judge and needs the answer text to evaluate."
            )

        contexts = self._extract_texts(retrieved_chunks)

        try:
            result = self._run_ragas(query, contexts, generated_answer)
        except Exception as exc:
            logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ragas evaluation failed: {exc}") from exc

        return result

    # ── private helpers ───────────────────────────────────────────

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        answer: str,
    ) -> Dict[str, float]:
        """Execute ragas evaluate() and return normalised metrics dict."""
        from ragas import evaluate as ragas_evaluate
        from ragas import EvaluationDataset, SingleTurnSample
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
        )

        # Build metric instances
        metric_map = {
            FAITHFULNESS: Faithfulness(),
            ANSWER_RELEVANCY: AnswerRelevancy(),
            CONTEXT_PRECISION: ContextPrecisionWithoutReference(),
        }
        selected_metrics = [metric_map[m] for m in self._metric_names]

        # Build single-turn sample
        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=contexts,
            response=answer,
        )
        dataset = EvaluationDataset(samples=[sample])

        # Build LLM / Embedding wrappers from settings
        llm_wrapper, embeddings_wrapper = self._build_wrappers()

        ragas_result = ragas_evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=llm_wrapper,
            embeddings=embeddings_wrapper,
        )

        # Extract scores from the result DataFrame
        scores: Dict[str, float] = {}
        df = ragas_result.to_pandas()
        for metric_name in self._metric_names:
            col = metric_name
            if col in df.columns:
                val = df[col].iloc[0]
                scores[metric_name] = float(val) if val is not None else 0.0
            else:
                scores[metric_name] = 0.0

        return scores

    def _build_wrappers(self) -> tuple:
        """Build Ragas LLM and Embedding wrappers from project settings.

        Returns:
            Tuple of (llm_wrapper, embeddings_wrapper).
        """
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        llm = self._create_langchain_llm()
        embeddings = self._create_langchain_embeddings()

        return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)

    def _create_langchain_llm(self) -> Any:
        """Create a LangChain chat model from settings."""
        if self.settings is None:
            raise ValueError("Settings required to create LLM for Ragas evaluation")

        llm_cfg = self.settings.llm
        provider = llm_cfg.provider.lower()

        if provider == "azure":
            from langchain_openai import AzureChatOpenAI
            import os

            return AzureChatOpenAI(
                azure_deployment=llm_cfg.deployment_name or llm_cfg.model,
                azure_endpoint=llm_cfg.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=llm_cfg.api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
                api_version=llm_cfg.api_version or "2024-02-15-preview",
                temperature=0.0,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            import os

            return ChatOpenAI(
                model=llm_cfg.model,
                api_key=llm_cfg.api_key or os.getenv("OPENAI_API_KEY", ""),
                temperature=0.0,
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider for Ragas: '{provider}'. "
                "Supported: azure, openai"
            )

    def _create_langchain_embeddings(self) -> Any:
        """Create a LangChain embedding model from settings."""
        if self.settings is None:
            raise ValueError("Settings required to create embeddings for Ragas evaluation")

        emb_cfg = self.settings.embedding
        provider = emb_cfg.provider.lower()

        if provider == "azure":
            from langchain_openai import AzureOpenAIEmbeddings
            import os

            return AzureOpenAIEmbeddings(
                azure_deployment=emb_cfg.deployment_name or emb_cfg.model,
                azure_endpoint=emb_cfg.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=emb_cfg.api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
                api_version=emb_cfg.api_version or "2024-02-15-preview",
            )
        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            import os

            return OpenAIEmbeddings(
                model=emb_cfg.model,
                api_key=emb_cfg.api_key or os.getenv("OPENAI_API_KEY", ""),
            )
        else:
            raise ValueError(
                f"Unsupported embedding provider for Ragas: '{provider}'. "
                "Supported: azure, openai"
            )

    def _extract_texts(self, chunks: List[Any]) -> List[str]:
        """Extract text strings from various chunk representations.

        Args:
            chunks: List of chunk dicts, strings, or objects with .text.

        Returns:
            List of text strings.
        """
        texts: List[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content", "")
                texts.append(str(text))
            elif hasattr(chunk, "text"):
                texts.append(str(getattr(chunk, "text")))
            else:
                texts.append(str(chunk))
        return texts

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []
        raw_metrics = getattr(evaluation, "metrics", None)
        if raw_metrics is None:
            return []
        # Filter to only ragas-supported metrics
        return [m for m in raw_metrics if m.lower() in SUPPORTED_METRICS]
