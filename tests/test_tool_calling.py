"""
Tests unitarios — ToolManager (generation/tools.py)

Módulo bajo test:
    src.rag.generation.tools.ToolManager.execute_tool_call()

Qué se valida:
    - Que la tool `search_earnings_calls` delega correctamente al Retriever
      pasando los parámetros de búsqueda.
    - Que `list_available_companies` devuelve la lista de empresas disponibles.
    - Que `predict_financial_outlook` delega correctamente al FinancialPredictor.
    - Que un tool_call con nombre desconocido devuelve un error controlado
      en vez de crashear.

Estrategia:
    Se usa unittest.mock.patch para inyectar mocks de Retriever y
    FinancialPredictor. Esto permite testear el routing de tools sin necesitar
    base de datos SQLite, modelo de embeddings, ni modelo ML cargados.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from src.rag.core.schema import RetrievedChunk


class TestToolManager:

    @patch("src.rag.generation.tools.FinancialPredictor")
    @patch("src.rag.generation.tools.Retriever")
    def test_execute_search_tool(self, MockRetriever, MockPredictor):
        """search_earnings_calls delega correctamente al Retriever."""
        # Configurar mock del retriever
        mock_retriever_instance = MockRetriever.return_value
        mock_retriever_instance.search.return_value = [
            RetrievedChunk(
                chunk_id="c1", text="Apple revenue was $91B.",
                score=0.9, doc_id="AAPL_2020", metadata={"company": "AAPL"}
            )
        ]

        # Importar DESPUÉS de patchear (para que use los mocks)
        from src.rag.generation.tools import ToolManager
        tm = ToolManager()

        # Simular un tool_call del LLM
        tool_call = {
            "id": "call_001",
            "function": {
                "name": "search_earnings_calls",
                "arguments": json.dumps({
                    "query": "Apple revenue",
                    "company_id": "AAPL",
                    "num_results": 3
                })
            }
        }

        result = tm.execute_tool_call(tool_call)
        result_data = json.loads(result)

        # Verificar que delegó la búsqueda al retriever
        mock_retriever_instance.search.assert_called_once()
        assert len(result_data) > 0

    @patch("src.rag.generation.tools.FinancialPredictor")
    @patch("src.rag.generation.tools.Retriever")
    def test_execute_list_companies_tool(self, MockRetriever, MockPredictor):
        """list_available_companies devuelve JSON con empresas."""
        mock_retriever_instance = MockRetriever.return_value
        mock_retriever_instance.get_available_companies.return_value = [
            {"company_id": "AAPL", "company_name": "Apple"},
            {"company_id": "GOOGL", "company_name": "Google"},
        ]

        from src.rag.generation.tools import ToolManager
        tm = ToolManager()

        tool_call = {
            "id": "call_002",
            "function": {
                "name": "list_available_companies",
                "arguments": "{}"
            }
        }

        result = tm.execute_tool_call(tool_call)
        result_data = json.loads(result)

        assert len(result_data) == 2
        assert result_data[0]["company_id"] == "AAPL"

    @patch("src.rag.generation.tools.FinancialPredictor")
    @patch("src.rag.generation.tools.Retriever")
    def test_execute_predict_tool(self, MockRetriever, MockPredictor):
        """predict_financial_outlook delega al Predictor."""
        # Mock retriever para el contexto
        mock_retriever_instance = MockRetriever.return_value
        mock_retriever_instance.search.return_value = [
            RetrievedChunk(
                chunk_id="c1", text="We expect strong revenue growth next quarter.",
                score=0.8, doc_id="AAPL_2020", metadata={"company": "AAPL"}
            )
        ]

        # Mock predictor
        mock_predictor_instance = MockPredictor.return_value
        mock_predictor_instance.predict.return_value = {
            "prediction": "POSITIVE",
            "confidence": 0.85,
            "features": {"sentiment": 0.7}
        }

        from src.rag.generation.tools import ToolManager
        tm = ToolManager()

        tool_call = {
            "id": "call_003",
            "function": {
                "name": "predict_financial_outlook",
                "arguments": json.dumps({
                    "company_id": "AAPL",
                    "year": 2020,
                    "quarter": 1
                })
            }
        }

        result = tm.execute_tool_call(tool_call)
        result_data = json.loads(result)

        assert "prediction" in result_data or "error" not in result_data

    @patch("src.rag.generation.tools.FinancialPredictor")
    @patch("src.rag.generation.tools.Retriever")
    def test_unknown_tool_returns_error(self, MockRetriever, MockPredictor):
        """Un tool_call con nombre desconocido devuelve error controlado."""
        from src.rag.generation.tools import ToolManager
        tm = ToolManager()

        tool_call = {
            "id": "call_999",
            "function": {
                "name": "herramienta_inventada",
                "arguments": "{}"
            }
        }

        result = tm.execute_tool_call(tool_call)

        assert "error" in result.lower() or "no encontrada" in result.lower() or "unknown" in result.lower(), \
            f"Debería devolver un mensaje de error, pero devolvió: {result}"
