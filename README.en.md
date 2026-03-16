# Hybrid RAG MCP Server

<p align="center">
  <a href="./README.md">Home</a> |
  <a href="./README.zh-CN.md">简体中文</a>
</p>

> A modular, observable RAG (Retrieval-Augmented Generation) framework with MCP integration.

---

## Overview

`Hybrid RAG MCP Server` is an engineering-oriented RAG project that separates document ingestion, chunking, embedding, hybrid retrieval, reranking, evaluation, and observability into composable modules.

It is designed for scenarios such as:

- enterprise knowledge retrieval
- document question answering and internal search
- MCP-based retrieval tools
- RAG experimentation and evaluation
- observable retrieval system prototyping

## Key Features

- **Modular architecture**: swap LLMs, embedding models, vector stores, rerankers, loaders, and splitters
- **Hybrid retrieval**: Dense Retrieval + Sparse Retrieval + RRF Fusion
- **Optional reranking**: enable or disable second-stage rerank
- **Document ingestion**: supports PDF, TXT, Markdown, DOCX, and similar formats
- **Observability**: built-in ingestion/query traces, evaluation history, and dashboard views
- **MCP integration**: expose retrieval capability as MCP tools
- **Interactive debugging**: `Query Playground` shows dense / sparse / fusion / rerank results side by side

## UI Preview

### Query Playground

![Query Playground](image/query_playground.png)

An interactive page for running queries and inspecting dense / sparse / fusion / rerank retrieval behavior.

### Data Browser

![Data Browser](image/data_browser.png)

A page for browsing ingested documents, chunk content, images, and metadata.

## Project Layout

- `src/core/`: query orchestration, response building, trace context
- `src/ingestion/`: ingestion, chunking, encoding, indexing, storage
- `src/libs/`: adapters for models, vector stores, loaders, and rerankers
- `src/mcp_server/`: MCP server implementation and tools
- `src/observability/`: logging, evaluation, dashboard
- `scripts/`: command-line entry points
- `tests/`: unit and integration tests

## Quick Start

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2. Configure models and services

Update `config/settings.yaml` with your own values:

- `llm.api_key`
- `embedding.api_key`
- optional `vision_llm.api_key`
- a valid `base_url` if you are using an OpenAI-compatible provider

You can also prefer environment variables, for example:

```bash
export OPENAI_API_KEY="your_api_key"
```

> Do not commit real API keys to a public repository.

### 3. Ingest documents

```bash
./.venv/bin/python scripts/ingest.py --file path/to/your.pdf --collection default
```

### 4. Run a query

```bash
./.venv/bin/python scripts/query.py \
  --query "Naval wealth happiness" \
  --collection default \
  --verbose
```

### 5. Launch the dashboard

```bash
./.venv/bin/python scripts/start_dashboard.py --host 127.0.0.1 --port 8501
```

Open: `http://127.0.0.1:8501`

Available pages include:

- `Overview`
- `Data Browser`
- `Ingestion Manager`
- `Ingestion Traces`
- `Query Playground`
- `Query Traces`
- `Evaluation Panel`

## Configuration

Main configuration lives in `config/settings.yaml`:

- `llm`: LLM backend settings
- `embedding`: embedding provider settings
- `vector_store`: vector database settings
- `retrieval`: dense / sparse / fusion parameters
- `rerank`: reranking settings
- `evaluation`: evaluation settings
- `observability`: logging and trace settings
- `ingestion`: chunking and batch-processing settings

## Notes

- Chinese sparse retrieval is currently based on simple rule-based tokenization
- For stronger Chinese sparse recall, integrating a dedicated tokenizer is recommended
- Runtime artifacts such as `data/`, `logs/`, virtual environments, and local editor metadata should remain untracked

## Testing

Run the full test suite:

```bash
./.venv/bin/python -m pytest
```

Run unit tests only:

```bash
./.venv/bin/python -m pytest tests/unit -q
```
