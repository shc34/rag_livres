# -*- coding: utf-8 -*-
# src/api/endpoints.py
"""FastAPI application exposing the RAG graph."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from src.api.schemas import ChatRequest, ChatResponse, SourceDetail
from src.core.logger import get_logger

logger = get_logger(__name__)

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from src.rag.graph import build_rag_graph
        logger.info("Building RAG graph...")
        _graph = build_rag_graph()
        logger.info("RAG graph ready.")
    return _graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-build the graph at startup."""
    _get_graph()
    yield


app = FastAPI(title="RAG Livre API", version="0.1.0", lifespan=lifespan)


@app.get("/api/rag/health")
async def health():
    return {"status": "ok"}


@app.post("/api/rag/chat", response_model=ChatResponse)
async def rag_chat(req: ChatRequest):
    try:
        graph = _get_graph()
        result = graph.invoke({"query": req.message, "corpus": req.corpus})

        raw_sources = result.get("sources", [])
        sources = [
            SourceDetail(
                filename=s.get("filename", "inconnu"),
                page=s.get("page", "?"),
                score=s.get("score"),
            )
            for s in raw_sources
        ]

        return ChatResponse(
            answer=result.get("answer", "Aucune réponse générée."),
            sources=sources,
        )

    except Exception as e:
        logger.error(f"RAG invocation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
