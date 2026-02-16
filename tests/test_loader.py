"""
Tests unitarios — Loader / Ingesta (step1_loader.py)

Módulo bajo test:
    src.rag.pipeline.step1_loader.load_processed_corpus()

Qué se valida:
    - Carga correcta de un archivo JSONL válido a objetos Document.
    - Manejo graceful de archivo inexistente (devuelve lista vacía, no crashea).
    - Tolerancia a líneas JSON malformadas: se saltan sin afectar al resto.

Estrategia:
    Usa la fixture `tmp_path` de pytest para crear archivos temporales en disco.
    No lee el corpus real del proyecto — datos controlados por el test.
"""
import json
import pytest
from src.rag.pipeline.step1_loader import load_processed_corpus


class TestLoader:

    def test_load_valid_corpus(self, tmp_path):
        """Cargar un corpus JSONL válido devuelve Documents correctos."""
        corpus_file = tmp_path / "corpus.jsonl"
        data = [
            {"doc_id": "AAPL_2020_Q1", "text": "Apple revenue was great.", "metadata": {"company": "AAPL"}},
            {"doc_id": "GOOGL_2019_Q3", "text": "Google ads grew.", "metadata": {"company": "GOOGL"}},
        ]
        corpus_file.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")

        documents = load_processed_corpus(corpus_file)

        assert len(documents) == 2
        assert documents[0].doc_id == "AAPL_2020_Q1"
        assert documents[1].text == "Google ads grew."
        assert documents[0].metadata["company"] == "AAPL"

    def test_load_missing_file_returns_empty(self, tmp_path):
        """Si el archivo no existe, devuelve lista vacía sin crashear."""
        fake_path = tmp_path / "no_existe.jsonl"

        documents = load_processed_corpus(fake_path)

        assert documents == []

    def test_load_malformed_line_skips(self, tmp_path):
        """Líneas JSON inválidas se saltan sin afectar al resto."""
        corpus_file = tmp_path / "corpus.jsonl"
        lines = [
            json.dumps({"doc_id": "OK_001", "text": "Valid doc.", "metadata": {}}),
            "esto no es json {{{",
            json.dumps({"doc_id": "OK_002", "text": "Also valid.", "metadata": {}}),
        ]
        corpus_file.write_text("\n".join(lines), encoding="utf-8")

        documents = load_processed_corpus(corpus_file)

        assert len(documents) == 2, "Debería cargar 2 docs y saltar la línea rota"
