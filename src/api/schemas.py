# -*- coding: utf-8 -*-
# src/api/schemas.py
"""Pydantic schemas for the RAG API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message."""
    message: str = Field(..., min_length=1, max_length=2000)
    corpus: str = Field(default="zola", pattern="^(zola|balzac)$")


class SourceDetail(BaseModel):
    """A single source reference."""
    filename: str = "inconnu"
    page: str | int = "?"
    score: float | None = None


class ChatResponse(BaseModel):
    """Response from the RAG graph."""
    answer: str
    sources: list[SourceDetail] = []
