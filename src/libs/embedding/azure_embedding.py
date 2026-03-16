"""Azure OpenAI Embedding implementation.

This module provides the Azure OpenAI Embedding implementation, which handles
Azure-specific configuration while reusing the core embedding contract.
"""

from __future__ import annotations

import os
from typing import Any

from src.libs.embedding.base_embedding import BaseEmbedding


class AzureEmbeddingError(RuntimeError):
    """Raised when Azure OpenAI Embeddings API call fails."""


class AzureEmbedding(BaseEmbedding):
    """Azure OpenAI Embedding provider implementation."""

    DEFAULT_API_VERSION = "2024-02-01"

    def __init__(
        self,
        settings: Any,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Azure OpenAI Embedding provider."""
        configured_deployment = self._as_non_empty_str(
            getattr(settings.embedding, "deployment_name", None)
        )
        self.deployment_name = configured_deployment or settings.embedding.model
        self.dimensions = getattr(settings.embedding, "dimensions", None)

        configured_api_key = self._as_non_empty_str(getattr(settings.embedding, "api_key", None))
        self.api_key = (
            api_key
            or os.environ.get("AZURE_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or configured_api_key
        )
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not provided. Configure 'api_key' in settings.yaml, "
                "set AZURE_OPENAI_API_KEY environment variable, or pass api_key parameter."
            )

        configured_endpoint = self._as_non_empty_str(
            getattr(settings.embedding, "azure_endpoint", None)
        )
        self.azure_endpoint = (
            azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") or configured_endpoint
        )
        if not self.azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not provided. Configure 'azure_endpoint' in settings.yaml, "
                "set AZURE_OPENAI_ENDPOINT environment variable, or pass azure_endpoint parameter."
            )

        configured_api_version = self._as_non_empty_str(
            getattr(settings.embedding, "api_version", None)
        )
        self.api_version = api_version or configured_api_version or self.DEFAULT_API_VERSION
        self._extra_config = kwargs

    def embed(
        self,
        texts: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts using Azure OpenAI API."""
        del trace
        self.validate_texts(texts)

        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI Python package not installed. Install with: pip install openai"
            ) from exc

        client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )

        api_params = {
            "input": texts,
            "model": self.deployment_name,
        }

        dimensions = kwargs.get("dimensions", self.dimensions)
        if dimensions is not None and "text-embedding-3" in self.deployment_name.lower():
            api_params["dimensions"] = dimensions

        try:
            response = client.embeddings.create(**api_params)
        except Exception as exc:
            raise AzureEmbeddingError(
                f"Azure OpenAI Embeddings API call failed: {exc}"
            ) from exc

        try:
            embeddings = [item.embedding for item in response.data]
        except (AttributeError, KeyError) as exc:
            raise AzureEmbeddingError(
                f"Failed to parse Azure OpenAI Embeddings API response: {exc}"
            ) from exc

        if len(embeddings) != len(texts):
            raise AzureEmbeddingError(
                f"Output length mismatch: expected {len(texts)}, got {len(embeddings)}"
            )

        return embeddings

    @staticmethod
    def _as_non_empty_str(value: Any) -> str | None:
        """Return a stripped string value, or ``None`` for mocks/empty values."""
        return value.strip() if isinstance(value, str) and value.strip() else None

    def get_dimension(self) -> int | None:
        """Get the embedding dimension for the configured deployment."""
        if self.dimensions is not None:
            return self.dimensions

        deployment_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        if self.deployment_name in deployment_dimensions:
            return deployment_dimensions[self.deployment_name]

        for model_key in sorted(deployment_dimensions.keys(), key=len, reverse=True):
            if model_key in self.deployment_name:
                return deployment_dimensions[model_key]

        return None
