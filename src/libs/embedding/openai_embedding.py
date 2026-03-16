"""OpenAI Embedding implementation.

This module provides the OpenAI Embedding implementation that works with
standard and OpenAI-compatible Embeddings APIs.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.libs.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger(__name__)


class OpenAIEmbeddingError(RuntimeError):
    """Raised when OpenAI Embeddings API call fails."""


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding provider implementation."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        settings: Any,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI Embedding provider."""
        self.model = settings.embedding.model
        self.dimensions = getattr(settings.embedding, "dimensions", None)

        settings_api_key = self._as_non_empty_str(getattr(settings.embedding, "api_key", None))
        self.api_key = api_key or settings_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        settings_base_url = self._as_non_empty_str(getattr(settings.embedding, "base_url", None))
        self.base_url = base_url or settings_base_url or self.DEFAULT_BASE_URL
        self._extra_config = kwargs

    def embed(
        self,
        texts: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts using OpenAI API."""
        del trace
        self.validate_texts(texts)
        client = self._create_client()
        return self._embed_with_fallback(client, texts, **kwargs)

    def _create_client(self) -> Any:
        """Create an OpenAI-compatible client lazily."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI Python package not installed. Install with: pip install openai"
            ) from exc

        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _embed_with_fallback(
        self,
        client: Any,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """Embed texts and retry with smaller batches or smaller text slices."""
        api_params = {
            "input": texts,
            "model": self.model,
        }

        dimensions = kwargs.get("dimensions", self.dimensions)
        if dimensions is not None:
            api_params["dimensions"] = dimensions

        try:
            response = client.embeddings.create(**api_params)
        except Exception as exc:
            if len(texts) > 1 and (
                self._is_batch_limit_error(exc) or self._is_input_too_long_error(exc)
            ):
                split_index = max(1, len(texts) // 2)
                logger.warning(
                    "Embedding batch of %s hit provider limits; retrying with smaller batches.",
                    len(texts),
                )
                left = self._embed_with_fallback(client, texts[:split_index], **kwargs)
                right = self._embed_with_fallback(client, texts[split_index:], **kwargs)
                return left + right

            if len(texts) == 1 and self._is_input_too_long_error(exc):
                logger.warning(
                    "Embedding input length exceeded provider limit; retrying with smaller text slices."
                )
                return [self._embed_single_text_with_fallback(client, texts[0], **kwargs)]

            raise OpenAIEmbeddingError(
                f"OpenAI Embeddings API call failed: {exc}"
            ) from exc

        try:
            embeddings = [item.embedding for item in response.data]
        except (AttributeError, KeyError) as exc:
            raise OpenAIEmbeddingError(
                f"Failed to parse OpenAI Embeddings API response: {exc}"
            ) from exc

        if len(embeddings) != len(texts):
            raise OpenAIEmbeddingError(
                f"Output length mismatch: expected {len(texts)}, got {len(embeddings)}"
            )

        return embeddings

    def _embed_single_text_with_fallback(
        self,
        client: Any,
        text: str,
        **kwargs: Any,
    ) -> list[float]:
        """Embed a single long text by splitting it recursively and averaging sub-vectors."""
        if len(text) <= 1:
            raise OpenAIEmbeddingError(
                "OpenAI Embeddings API call failed: input is too long and cannot be reduced further"
            )

        left_text, right_text = self._split_text_for_embedding(text)
        if not left_text or not right_text or left_text == text or right_text == text:
            midpoint = max(1, len(text) // 2)
            left_text = text[:midpoint]
            right_text = text[midpoint:]

        left_vector = self._embed_with_fallback(client, [left_text], **kwargs)[0]
        right_vector = self._embed_with_fallback(client, [right_text], **kwargs)[0]
        return self._weighted_average_vectors(
            left_vector,
            right_vector,
            len(left_text),
            len(right_text),
        )

    @staticmethod
    def _is_batch_limit_error(exc: Exception) -> bool:
        """Return whether an exception looks like a provider batch-size error."""
        message = str(exc).lower()
        patterns = (
            "maximum allowed batch size",
            "input batch size",
            "batch size",
            "too many inputs",
            "too many items",
        )
        return any(pattern in message for pattern in patterns)

    @staticmethod
    def _is_input_too_long_error(exc: Exception) -> bool:
        """Return whether an exception looks like a per-input token-length error."""
        message = str(exc).lower()
        patterns = (
            "less than 512 tokens",
            "maximum context length",
            "too many tokens",
            "input is too long",
            "input must have less than",
            "token limit",
        )
        return any(pattern in message for pattern in patterns)

    @staticmethod
    def _split_text_for_embedding(text: str) -> tuple[str, str]:
        """Split text near the midpoint, preferring natural boundaries when possible."""
        midpoint = len(text) // 2
        separators = "\n。！？；;.!?，, "
        window = min(120, max(20, len(text) // 10))
        start = max(1, midpoint - window)
        end = min(len(text) - 1, midpoint + window)

        for index in range(midpoint, start - 1, -1):
            if text[index] in separators:
                return text[:index].strip(), text[index + 1 :].strip()

        for index in range(midpoint + 1, end):
            if text[index] in separators:
                return text[:index].strip(), text[index + 1 :].strip()

        return text[:midpoint].strip(), text[midpoint:].strip()

    @staticmethod
    def _weighted_average_vectors(
        left_vector: list[float],
        right_vector: list[float],
        left_weight: int,
        right_weight: int,
    ) -> list[float]:
        """Average two embedding vectors using text length as weight."""
        total_weight = max(1, left_weight + right_weight)
        return [
            ((left_value * left_weight) + (right_value * right_weight)) / total_weight
            for left_value, right_value in zip(left_vector, right_vector)
        ]

    @staticmethod
    def _as_non_empty_str(value: Any) -> str | None:
        """Return a stripped string value, or ``None`` for mocks/empty values."""
        return value.strip() if isinstance(value, str) and value.strip() else None

    def get_dimension(self) -> int | None:
        """Get the embedding dimension for the configured model."""
        if self.dimensions is not None:
            return self.dimensions

        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dimensions.get(self.model)
