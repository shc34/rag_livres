# 📰 Project News — AI News Intelligence System

An end-to-end AI-powered system that ingests daily news articles, processes them through an NLP pipeline, and exposes insights via a RAG-based API.

## 🎯 Overview

This system automatically:
- Fetches and processes news articles daily
- Generates embeddings and extracts keywords
- Provides a conversational Q&A interface with source citations
- Exposes topic trends and article search via API

## 🏗️ Architecture

News APIs → Batch Pipeline (Prefect) → PostgreSQL + ChromaDB
                                              ↓
                                    FastAPI ← LangGraph RAG


### Components

| Component | Tech | Role |
|-----------|------|------|
| **Pipeline** | Prefect | Daily ingestion, cleaning, embedding, keyword extraction |
| **Storage** | PostgreSQL + ChromaDB | Structured data + vector search |
| **API** | FastAPI | REST endpoints for chat, topics, articles |
| **RAG** | LangGraph | Query understanding, retrieval, answer generation |
| **Observability** | LangSmith / Langfuse | Tracing and evaluation |

## 🚀 Quickstart

### Prerequisites
- Python 3.12+
- PostgreSQL
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
git clone git@github.com:YOUR-USERNAME/project_news.git
cd project_news
uv sync --extra dev

Configuration

cp .env.example .env
# Edit .env with your API keys and database credentials

Run

# Start the API
uv run uvicorn src.api.main:app --reload

# Run the pipeline
uv run prefect deployment run ...

📁 Project Structure

project_news/
├── src/
│   ├── pipeline/       # Prefect flows: ingestion, embeddings, keywords
│   ├── api/            # FastAPI endpoints
│   ├── rag/            # LangGraph orchestration
│   ├── db/             # PostgreSQL & ChromaDB clients
│   └── config/         # Settings, env loading
├── tests/
├── pyproject.toml
└── README.md

📡 API Endpoints
Method 	Endpoint 	Description
POST 	/chat 	RAG-based Q&A with sources
GET 	/topics 	Keyword/topic aggregation
GET 	/articles 	Filter articles by date/topic
🛠️ Tech Stack

Python · FastAPI · PostgreSQL · ChromaDB · LangGraph · LangChain · Prefect · TF-IDF
📄 License

MIT


Remplace `YOUR-USERNAME` par ton pseudo GitHub. C'est concis, informatif, et facile à mettre à jour au fil du projet.
