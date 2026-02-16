"""
Tests unitarios — Chunking (step2_chunking.py)

Módulo bajo test:
    src.rag.pipeline.step2_chunking.chunk_documents()

Qué se valida:
    - Que un texto largo se divide en chunks del tamaño configurado (CHUNK_SIZE).
    - Que existe solapamiento (overlap) entre chunks consecutivos para no perder
      contexto semántico en los bordes.
    - Que la metadata del Document padre se hereda correctamente a cada Chunk.
    - Comportamiento con documentos de texto vacío (edge case).

Estrategia:
    Tests aislados con datos inline (sin leer corpus real).
    No requiere dependencias externas.
"""
import pytest
from src.rag.pipeline.step2_chunking import chunk_documents
from src.rag.core.schema import Document


class TestChunking:

    def test_chunk_documents_splits_correctly(self):
        """Un texto largo se divide en chunks del tamaño configurado."""
        # Crear un documento con texto largo (> CHUNK_SIZE = 800 chars)
        long_text = "palabra " * 200  # ~1600 chars
        doc = Document(doc_id="TEST_001", text=long_text, metadata={"company": "TEST"})

        chunks = chunk_documents([doc])

        assert len(chunks) > 1, "Debería generar más de 1 chunk para texto largo"
        for chunk in chunks:
            # Cada chunk no debería superar CHUNK_SIZE + margen
            assert len(chunk.text) <= 900, f"Chunk demasiado largo: {len(chunk.text)}"

    def test_chunk_overlap_works(self):
        """Los chunks consecutivos tienen texto en común (overlap)."""
        long_text = "palabra " * 200
        doc = Document(doc_id="TEST_001", text=long_text, metadata={"company": "TEST"})

        chunks = chunk_documents([doc])

        if len(chunks) >= 2:
            # El final del chunk 0 debería aparecer al inicio del chunk 1
            tail_of_first = chunks[0].text[-50:]
            assert tail_of_first in chunks[1].text, \
                "No se detectó overlap entre chunks consecutivos"

    def test_chunk_metadata_preserved(self):
        """Los chunks heredan la metadata del documento padre."""
        metadata = {"company": "AAPL", "year": 2020, "quarter": "Q1"}
        doc = Document(doc_id="AAPL_2020", text="Texto de prueba " * 200, metadata=metadata)

        chunks = chunk_documents([doc])

        for chunk in chunks:
            assert chunk.metadata == metadata, "Los chunks deben heredar la metadata"
            assert chunk.doc_id == "AAPL_2020", "Los chunks deben heredar el doc_id"

    def test_empty_document_produces_single_chunk(self):
        """Un documento con texto vacío produce un solo chunk (texto corto <= chunk_size)."""
        doc = Document(doc_id="EMPTY", text="", metadata={})

        chunks = chunk_documents([doc])

        # Texto vacío es <= chunk_size, así que se crea 1 chunk con texto vacío
        assert len(chunks) == 1
        assert chunks[0].text == ""
