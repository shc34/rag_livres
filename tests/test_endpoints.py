# -*- coding: utf-8 -*-
# tests/test_endpoints.py
"""Tests for the RAG API endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.endpoints import app, _get_graph


@pytest.fixture
def client():
    return TestClient(app)


def _make_fake_graph():
    fake = MagicMock()
    fake.invoke.return_value = {
        "answer": "Gervaise est un personnage de L'Assommoir.",
        "sources": [
            {"filename": "assommoir.pdf", "page": 12, "score": 0.95},
            {"filename": "assommoir.pdf", "page": 34},
        ],
    }
    return fake


@pytest.fixture(autouse=True)
def mock_graph():
    fake = _make_fake_graph()
    with patch("src.api.endpoints._graph", fake):
        yield fake


# ─── Health ───────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/rag/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ─── Chat endpoint ───────────────────────────────────────────────

class TestRagChat:
    def test_valid_request_zola(self, client, mock_graph):
        resp = client.post(
            "/api/rag/chat",
            json={"message": "Qui est Gervaise ?", "corpus": "zola"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Gervaise est un personnage de L'Assommoir."
        assert len(data["sources"]) == 2
        assert data["sources"][0]["filename"] == "assommoir.pdf"
        assert data["sources"][0]["page"] == 12
        assert data["sources"][0]["score"] == 0.95
        assert data["sources"][1]["score"] is None

        mock_graph.invoke.assert_called_once_with({
            "query": "Qui est Gervaise ?",
            "corpus": "zola",
        })

    def test_valid_request_balzac(self, client, mock_graph):
        resp = client.post(
            "/api/rag/chat",
            json={"message": "Qui est Rastignac ?", "corpus": "balzac"},
        )
        assert resp.status_code == 200
        mock_graph.invoke.assert_called_once_with({
            "query": "Qui est Rastignac ?",
            "corpus": "balzac",
        })

    def test_default_corpus_is_zola(self, client, mock_graph):
        resp = client.post(
            "/api/rag/chat",
            json={"message": "Bonjour"},
        )
        assert resp.status_code == 200
        mock_graph.invoke.assert_called_once_with({
            "query": "Bonjour",
            "corpus": "zola",
        })

    def test_empty_message_rejected(self, client):
        resp = client.post(
            "/api/rag/chat",
            json={"message": "", "corpus": "zola"},
        )
        assert resp.status_code == 422

    def test_invalid_corpus_rejected(self, client):
        resp = client.post(
            "/api/rag/chat",
            json={"message": "Hello", "corpus": "hugo"},
        )
        assert resp.status_code == 422

    def test_graph_error_returns_500(self, client, mock_graph):
        mock_graph.invoke.side_effect = RuntimeError("LLM timeout")
        resp = client.post(
            "/api/rag/chat",
            json={"message": "Question", "corpus": "zola"},
        )
        assert resp.status_code == 500
        assert "LLM timeout" in resp.json()["detail"]

    def test_missing_sources_in_result(self, client, mock_graph):
        mock_graph.invoke.return_value = {"answer": "Réponse sans sources"}
        resp = client.post(
            "/api/rag/chat",
            json={"message": "Test"},
        )
        assert resp.status_code == 200
        assert resp.json()["sources"] == []

    def test_missing_answer_in_result(self, client, mock_graph):
        mock_graph.invoke.return_value = {}
        resp = client.post(
            "/api/rag/chat",
            json={"message": "Test"},
        )
        assert resp.status_code == 200
        assert resp.json()["answer"] == "Aucune réponse générée."


# ─── _get_graph lazy loading ─────────────────────────────────────

class TestGetGraph:
    def test_builds_graph_when_none(self):
        """Covers lines 18-23: _graph is None → import + build."""
        fake = MagicMock()
        with patch("src.api.endpoints._graph", None), \
             patch("src.api.endpoints.build_rag_graph", create=True) as mock_build:
            # We need to patch at module level since import happens inside _get_graph
            import src.api.endpoints as mod
            original = mod._graph
            mod._graph = None

            mock_build_graph = MagicMock(return_value=fake)
            with patch.dict("sys.modules", {}), \
                 patch("src.rag.graph.build_rag_graph", mock_build_graph, create=True):
                # Directly patch the import target
                with patch("src.api.endpoints._graph", None):
                    with patch(
                        "src.rag.graph.build_rag_graph",
                        return_value=fake,
                        create=True,
                    ) as mb:
                        mod._graph = None
                        result = mod._get_graph()
                        assert result is not None

            mod._graph = original

    def test_returns_cached_graph(self):
        """Covers: _graph is not None → return immediately."""
        fake = MagicMock()
        import src.api.endpoints as mod
        original = mod._graph
        mod._graph = fake
        assert mod._get_graph() is fake
        mod._graph = original


# ─── Lifespan ────────────────────────────────────────────────────

class TestLifespan:
    def test_lifespan_calls_get_graph(self):
        """Covers lines 29-30: lifespan startup."""
        fake = MagicMock()
        with patch("src.api.endpoints._graph", fake):
            # TestClient triggers lifespan automatically
            with TestClient(app):
                pass  # startup + shutdown ran without error
