# Modular RAG MCP Server

> 中文 | English
>
> 一个可插拔、可观测、支持 MCP 集成的模块化 RAG（Retrieval-Augmented Generation）服务框架。  
> A modular, observable RAG (Retrieval-Augmented Generation) framework with MCP integration.

---

## 中文说明

### 项目简介

`Modular RAG MCP Server` 是一个面向工程实践的 RAG 项目，目标是将文档摄取、切分、向量化、混合检索、重排、评测和可观测能力解耦成可组合模块，方便你按需替换模型、向量库和检索策略。

项目适合以下场景：

- 企业知识库检索
- 文档问答与内部搜索
- MCP 工具化知识增强
- RAG 流程实验与评测
- 可观测的检索系统原型开发

### 核心特性

- **模块化架构**：LLM、Embedding、Vector Store、Reranker、Loader、Splitter 都可替换
- **混合检索**：支持 Dense Retrieval + Sparse Retrieval + RRF Fusion
- **可选重排**：支持关闭或启用 rerank 进行二阶段精排
- **文档摄取**：支持 PDF、TXT、Markdown、DOCX 等常见文档格式
- **可观测性**：内置 ingestion/query trace、评测历史与可视化 Dashboard
- **MCP 集成**：可以通过 MCP Tool 形式暴露知识检索能力
- **交互式调试**：提供 `Query Playground` 页面查看 dense / sparse / fusion / rerank 结果

## 界面预览 | UI Preview

### Query Playground

![Query Playground](image/query_playground.png)

- 中文：用于直接发起查询，查看 dense / sparse / fusion / rerank 的召回效果。
- English: An interactive page for running queries and inspecting dense / sparse / fusion / rerank retrieval behavior.

### Data Browser

![Data Browser](image/data_browser.png)

- 中文：用于浏览已导入文档、chunk 内容、图片与元数据。
- English: A page for browsing ingested documents, chunk content, images, and metadata.

### 系统组成

项目主要由以下部分组成：

- `src/core/`：查询编排、响应构建、追踪上下文等核心逻辑
- `src/ingestion/`：文档摄取、切分、编码、索引与存储
- `src/libs/`：模型、向量库、加载器、重排器等适配层
- `src/mcp_server/`：MCP Server 与工具定义
- `src/observability/`：日志、评测、Dashboard
- `scripts/`：常用命令行入口
- `tests/`：单元测试与集成测试

### 快速开始

#### 1. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

#### 2. 配置模型与服务

编辑 `config/settings.yaml`，填入你自己的配置：

- `llm.api_key`
- `embedding.api_key`
- 可选的 `vision_llm.api_key`
- 如果使用 OpenAI 兼容服务，请确认 `base_url` 正确

也可以优先使用环境变量，例如：

```bash
export OPENAI_API_KEY="your_api_key"
```

#### 3. 导入文档

```bash
./.venv/bin/python scripts/ingest.py --file path/to/your.pdf --collection default
```

#### 4. 运行查询

```bash
./.venv/bin/python scripts/query.py \
  --query "纳瓦尔 财富 幸福" \
  --collection default \
  --verbose
```

#### 5. 启动可视化 Dashboard

```bash
./.venv/bin/python scripts/start_dashboard.py --host 127.0.0.1 --port 8501
```

浏览器打开：`http://127.0.0.1:8501`

可使用页面包括：

- `Overview`
- `Data Browser`
- `Ingestion Manager`
- `Ingestion Traces`
- `Query Playground`
- `Query Traces`
- `Evaluation Panel`

### 配置说明

关键配置位于 `config/settings.yaml`：

- `llm`：大语言模型配置
- `embedding`：向量模型配置
- `vector_store`：向量数据库配置
- `retrieval`：Dense / Sparse / Fusion 参数
- `rerank`：重排配置
- `evaluation`：评测配置
- `observability`：日志与 trace 配置
- `ingestion`：切分、批处理等摄取参数

### 常见说明

- 当前中文查询的规则分词较简单，`Sparse Retrieval` 对“空格分隔关键词”更友好
- 如果你更重视中文稀疏召回，建议后续接入更好的中文分词方案
- 默认不提交 `data/`、`logs/`、虚拟环境与本地 IDE 配置

### 测试

运行全部测试：

```bash
./.venv/bin/python -m pytest
```

运行部分测试：

```bash
./.venv/bin/python -m pytest tests/unit -q
```

---

## English

### Overview

`Modular RAG MCP Server` is an engineering-oriented RAG project that separates document ingestion, chunking, embedding, hybrid retrieval, reranking, evaluation, and observability into composable modules.

It is designed for scenarios such as:

- enterprise knowledge retrieval
- document question answering and internal search
- MCP-based retrieval tools
- RAG experimentation and evaluation
- observable retrieval system prototyping

### Key Features

- **Modular architecture**: swap LLMs, embedding models, vector stores, rerankers, loaders, and splitters
- **Hybrid retrieval**: Dense Retrieval + Sparse Retrieval + RRF Fusion
- **Optional reranking**: enable or disable second-stage rerank
- **Document ingestion**: supports PDF, TXT, Markdown, DOCX, and similar formats
- **Observability**: built-in ingestion/query traces, evaluation history, and dashboard views
- **MCP integration**: expose retrieval capability as MCP tools
- **Interactive debugging**: `Query Playground` shows dense / sparse / fusion / rerank results side by side

### Project Layout

- `src/core/`: query orchestration, response building, trace context
- `src/ingestion/`: ingestion, chunking, encoding, indexing, storage
- `src/libs/`: adapters for models, vector stores, loaders, and rerankers
- `src/mcp_server/`: MCP server implementation and tools
- `src/observability/`: logging, evaluation, dashboard
- `scripts/`: command-line entry points
- `tests/`: unit and integration tests

### Quick Start

#### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

#### 2. Configure models and services

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

#### 3. Ingest documents

```bash
./.venv/bin/python scripts/ingest.py --file path/to/your.pdf --collection default
```

#### 4. Run a query

```bash
./.venv/bin/python scripts/query.py \
  --query "Naval wealth happiness" \
  --collection default \
  --verbose
```

#### 5. Launch the dashboard

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

### Configuration

Main configuration lives in `config/settings.yaml`:

- `llm`: LLM backend settings
- `embedding`: embedding provider settings
- `vector_store`: vector database settings
- `retrieval`: dense / sparse / fusion parameters
- `rerank`: reranking settings
- `evaluation`: evaluation settings
- `observability`: logging and trace settings
- `ingestion`: chunking and batch-processing settings

### Notes

- Chinese sparse retrieval is currently based on simple rule-based tokenization
- For stronger Chinese sparse recall, integrating a dedicated tokenizer is recommended
- Runtime artifacts such as `data/`, `logs/`, virtual environments, and local editor metadata should remain untracked

### Testing

Run the full test suite:

```bash
./.venv/bin/python -m pytest
```

Run unit tests only:

```bash
./.venv/bin/python -m pytest tests/unit -q
```
