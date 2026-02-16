
import pytest
from unittest.mock import MagicMock, patch
from src.rag.retrieval.reranker import Reranker
from src.rag.core.schema import RetrievedChunk

class TestReranker:
    @patch("src.rag.retrieval.reranker.CrossEncoder")
    def test_rerank_logic(self, mock_cross_encoder_cls):
        """Probar que rerank ordena correctamente basado en scores del modelo."""
        # 1. Configurar Mock
        mock_model = MagicMock()
        mock_cross_encoder_cls.return_value = mock_model
        
        # Simulamos que el modelo devuelve scores para 2 chunks
        # Chunk 1 -> 0.1 (malo)
        # Chunk 2 -> 0.9 (bueno)
        mock_model.predict.return_value = [0.1, 0.9]
        
        # 2. Instanciar
        reranker = Reranker()
        
        # 3. Datos de prueba
        chunk1 = RetrievedChunk(chunk_id="c1", text="text1", score=0.5, doc_id="d1")
        chunk2 = RetrievedChunk(chunk_id="c2", text="text2", score=0.5, doc_id="d2")
        
        chunks = [chunk1, chunk2]
        
        # 4. Ejecutar rerank
        results = reranker.rerank("query", chunks, top_k=2)
        
        # 5. Aserciones
        assert len(results) == 2
        # El primero debe ser chunk2 (score 0.9)
        assert results[0].chunk_id == "c2"
        assert results[0].score == 0.9
        
        # El segundo debe ser chunk1 (score 0.1)
        assert results[1].chunk_id == "c1"
        assert results[1].score == 0.1
        
        # Verificar que se llamó a predict con los pares correctos
        expected_pairs = [["query", "text1"], ["query", "text2"]]
        mock_model.predict.assert_called_with(expected_pairs)

    @patch("src.rag.retrieval.reranker.CrossEncoder")
    def test_rerank_top_k(self, mock_cross_encoder_cls):
        """Probar que respeta el top_k."""
        mock_model = MagicMock()
        mock_cross_encoder_cls.return_value = mock_model
        
        # 3 chunks, scores [0.1, 0.8, 0.2]
        mock_model.predict.return_value = [0.1, 0.8, 0.2]
        
        reranker = Reranker()
        
        c1 = RetrievedChunk(chunk_id="c1", text="t1", score=0, doc_id="d")
        c2 = RetrievedChunk(chunk_id="c2", text="t2", score=0, doc_id="d")
        c3 = RetrievedChunk(chunk_id="c3", text="t3", score=0, doc_id="d")
        
        results = reranker.rerank("q", [c1, c2, c3], top_k=1)
        
        assert len(results) == 1
        assert results[0].chunk_id == "c2" # El de score 0.8

    def test_rerank_empty(self):
        """Probar lista vacía."""
        # No necesitamos mockear init si la clase carga el modelo en init...
        # Espera, sí, porque Reranker.__init__ carga el modelo real.
        with patch("src.rag.retrieval.reranker.CrossEncoder") as mock_cls:
            reranker = Reranker()
            results = reranker.rerank("q", [], top_k=5)
            assert results == []
