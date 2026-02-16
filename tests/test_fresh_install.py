
import pytest
import sqlite3
import numpy as np
from src.rag.storage.unified_store import UnifiedDocumentStore
from src.rag.core.schema import Chunk

def test_fresh_install_schema(tmp_path):
    """Verifica que una instalación fresca crea la tabla con las columnas nuevas."""
    db_path = tmp_path / "test_schema.db"
    
    # 1. Instanciar Store (crea DB y tablas)
    store = UnifiedDocumentStore(db_path, dimension=384)
    
    # 2. Verificar Columnas
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in cursor.fetchall()}
        
    expected_cols = {"chunk_id", "doc_id", "text", "chunk_index", "metadata", "embedding", "company", "year", "quarter"}
    assert expected_cols.issubset(columns), f"Faltan columnas. Encontradas: {columns}"
    
    # 3. Verificar Índices
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}
        
    expected_indexes = {"idx_doc_id", "idx_company", "idx_year", "idx_quarter"}
    found_indexes = {idx for idx in indexes if idx in expected_indexes}
    assert expected_indexes.issubset(found_indexes), f"Faltan índices. Encontrados: {indexes}"

def test_fresh_ingest_populates_columns(tmp_path):
    """Verifica que add_documents puebla las columnas SQL."""
    db_path = tmp_path / "test_ingest.db"
    store = UnifiedDocumentStore(db_path, dimension=2)
    
    chunk = Chunk(
        chunk_id="test_1",
        doc_id="doc_1",
        text="Sample text",
        chunk_index=0,
        metadata={"company": "TEST", "year": 2024, "quarter": "Q1"}
    )
    fake_emb = np.array([[0.1, 0.2]], dtype=np.float32)
    
    store.add_documents([chunk], fake_emb)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT company, year, quarter FROM chunks WHERE chunk_id='test_1'")
        row = cursor.fetchone()
        
    assert row == ("TEST", 2024, 1) # Note: ingestion logic extracts digits from "Q1" -> 1
