# -*- coding: utf-8 -*-
# tests/test_schemas.py
"""Tests for the RAG API Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.api.schemas import ChatRequest, ChatResponse, SourceDetail


# ─── ChatRequest ──────────────────────────────────────────────────

class TestChatRequest:
    def test_valid_zola(self):
        req = ChatRequest(message="Qui est Gervaise ?", corpus="zola")
        assert req.message == "Qui est Gervaise ?"
        assert req.corpus == "zola"

    def test_valid_balzac(self):
        req = ChatRequest(message="Résumé", corpus="balzac")
        assert req.corpus == "balzac"

    def test_default_corpus(self):
        req = ChatRequest(message="Hello")
        assert req.corpus == "zola"

    def test_empty_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="", corpus="zola")

    def test_missing_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(corpus="zola")

    def test_invalid_corpus_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello", corpus="hugo")

    def test_message_max_length(self):
        req = ChatRequest(message="x" * 2000)
        assert len(req.message) == 2000

    def test_message_too_long_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="x" * 2001)


# ─── SourceDetail ─────────────────────────────────────────────────

class TestSourceDetail:
    def test_full_source(self):
        s = SourceDetail(filename="livre.pdf", page=42, score=0.95)
        assert s.filename == "livre.pdf"
        assert s.page == 42
        assert s.score == 0.95

    def test_defaults(self):
        s = SourceDetail()
        assert s.filename == "inconnu"
        assert s.page == "?"
        assert s.score is None

    def test_page_as_string(self):
        s = SourceDetail(page="12-13")
        assert s.page == "12-13"


# ─── ChatResponse ────────────────────────────────────────────────

class TestChatResponse:
    def test_full_response(self):
        resp = ChatResponse(
            answer="Réponse",
            sources=[SourceDetail(filename="a.pdf", page=1, score=0.9)],
        )
        assert resp.answer == "Réponse"
        assert len(resp.sources) == 1

    def test_empty_sources_default(self):
        resp = ChatResponse(answer="Réponse")
        assert resp.sources == []

    def test_missing_answer_raises(self):
        with pytest.raises(ValidationError):
            ChatResponse()
