# -*- coding: utf-8 -*-
"""
Shared fixtures for ingestion tests.
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Two minimal Documents simulating parsed PDF pages."""
    return [
        Document(
            page_content="The Kalman filter is an optimal estimator.",
            metadata={"source": "/data/raw/kalman.pdf", "filename": "kalman.pdf", "page": 1, "total_pages": 2},
        ),
        Document(
            page_content="It minimizes the mean squared error of the estimates.",
            metadata={"source": "/data/raw/kalman.pdf", "filename": "kalman.pdf", "page": 2, "total_pages": 2},
        ),
    ]


@pytest.fixture
def sample_chunks(sample_documents) -> list[Document]:
    """Pre-chunked documents for indexer tests."""
    return [
        Document(
            page_content=f"Chunk {i} content about Kalman filtering.",
            metadata={"source": "/data/raw/kalman.pdf", "filename": "kalman.pdf", "page": 1, "total_pages": 2},
        )
        for i in range(5)
    ]


@pytest.fixture
def tmp_pdf(tmp_path) -> Path:
    """Create a minimal valid PDF using PyMuPDF."""
    import fitz
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Hello, this is a test page.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path
