"""
Configuración compartida de tests — conftest.py

Propósito:
    Archivo especial de pytest que se ejecuta automáticamente antes de
    cualquier test. Define fixtures reutilizables que se inyectan por
    nombre en los tests que las necesiten.

Fixtures disponibles:
    - sample_documents: Lista de Document de ejemplo (AAPL, GOOGL)
    - sample_chunks: Lista de RetrievedChunk con scores de relevancia
    - mock_groq_response: Respuesta simulada de Groq API (sin tool calls)
    - mock_groq_tool_call_response: Respuesta simulada que pide usar una tool

Convención:
    Todas las fixtures usan datos realistas pero controlados.
    Ninguna fixture accede a recursos externos (API, BD, disco).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import json

import pytest

# Agregar raiz del proyecto al path (una sola vez)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.core.schema import Document, Chunk, RetrievedChunk


# =======================================================
# Fixtures: Datos de ejemplo
# =======================================================

@pytest.fixture
def sample_documents():
    """Documentos de ejemplo para tests (sin leer corpus real)."""
    return [
        Document(
            doc_id="AAPL_2020_Q1",
            text="Apple reported record revenue of $91.8 billion for Q1 2020. "
                 "iPhone revenue was $55.96 billion. Services revenue reached "
                 "$12.7 billion, an all-time high. Tim Cook stated the company "
                 "is optimistic about future growth in wearables and services.",
            metadata={"company": "AAPL", "year": 2020, "quarter": "Q1"}
        ),
        Document(
            doc_id="GOOGL_2019_Q3",
            text="Google parent Alphabet reported revenue of $40.5 billion "
                 "in Q3 2019. Advertising revenue grew 17% year over year. "
                 "Cloud revenue showed strong momentum reaching $2.4 billion. "
                 "Sundar Pichai emphasized AI investments across all products.",
            metadata={"company": "GOOGL", "year": 2019, "quarter": "Q3"}
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Chunks de ejemplo para tests de búsqueda y generación."""
    return [
        RetrievedChunk(
            chunk_id="chunk_001",
            text="Apple reported record revenue of $91.8 billion for Q1 2020.",
            score=0.92,
            doc_id="AAPL_2020_Q1",
            metadata={"company": "AAPL", "year": 2020, "quarter": "Q1"},
        ),
        RetrievedChunk(
            chunk_id="chunk_002",
            text="iPhone revenue was $55.96 billion. Services revenue reached $12.7 billion.",
            score=0.85,
            doc_id="AAPL_2020_Q1",
            metadata={"company": "AAPL", "year": 2020, "quarter": "Q1"},
        ),
    ]


@pytest.fixture
def mock_groq_response():
    """Respuesta simulada de Groq API (sin tool calls)."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Apple reportó ingresos récord de $91.8 mil millones en Q1 2020."
            }
        }]
    }


@pytest.fixture
def mock_groq_tool_call_response():
    """Respuesta simulada de Groq API que pide usar una herramienta."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "search_earnings_calls",
                        "arguments": json.dumps({
                            "query": "Apple revenue Q1 2020",
                            "company_id": "AAPL",
                            "num_results": 5
                        })
                    }
                }]
            }
        }]
    }
