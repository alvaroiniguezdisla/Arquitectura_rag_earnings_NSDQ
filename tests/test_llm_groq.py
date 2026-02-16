"""
Tests unitarios — GroqLLM (generation/llm_groq.py)

Módulo bajo test:
    src.rag.generation.llm_groq.GroqLLM

Qué se valida:
    - Respuesta directa: cuando el LLM responde sin usar herramientas,
      se devuelve el contenido correctamente.
    - Flujo completo de Tool Calling: simula el ciclo de 2 pasos
      (1ª llamada → tool call → ejecución → 2ª llamada → respuesta final).
    - Manejo de errores HTTP: un fallo de conexión devuelve un mensaje
      de error legible en vez de una excepción no controlada.
    - Validación de API key: sin GROQ_API_KEY se lanza ValueError.

Estrategia:
    Se mockea `requests.post` para simular respuestas de la API de Groq
    y `tool_manager` para simular la ejecución de herramientas.
    No se realizan llamadas HTTP reales — no necesita API key.
"""
import json
import pytest
from unittest.mock import patch, MagicMock


class TestGroqLLM:

    @patch("src.rag.generation.llm_groq.requests.post")
    @patch("src.rag.generation.llm_groq.tool_manager")
    def test_chat_direct_response(self, mock_tool_manager, mock_post, mock_groq_response):
        """Cuando Groq responde directamente (sin tools), devuelve el contenido."""
        # Simular respuesta HTTP exitosa
        mock_response = MagicMock()
        mock_response.json.return_value = mock_groq_response
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        from src.rag.generation.llm_groq import GroqLLM
        llm = GroqLLM(api_key="fake-key-for-testing")

        result = llm.chat_with_tools("¿Cuáles fueron los ingresos de Apple?")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Apple" in result or "91.8" in result

    @patch("src.rag.generation.llm_groq.requests.post")
    @patch("src.rag.generation.llm_groq.tool_manager")
    def test_chat_with_tool_call(self, mock_tool_manager, mock_post,
                                  mock_groq_tool_call_response, mock_groq_response):
        """Simula el flujo de 2 pasos: tool call → ejecución → respuesta final."""
        # Primera llamada: el LLM pide usar una tool
        mock_response_1 = MagicMock()
        mock_response_1.json.return_value = mock_groq_tool_call_response
        mock_response_1.raise_for_status = MagicMock()

        # Segunda llamada: el LLM da la respuesta final
        mock_response_2 = MagicMock()
        mock_response_2.json.return_value = mock_groq_response
        mock_response_2.raise_for_status = MagicMock()

        mock_post.side_effect = [mock_response_1, mock_response_2]

        # Mock del resultado de la tool
        mock_tool_manager.execute_tool_call.return_value = json.dumps([
            {"text": "Apple revenue was $91.8B", "score": 0.95}
        ])

        from src.rag.generation.llm_groq import GroqLLM
        llm = GroqLLM(api_key="fake-key-for-testing")

        result = llm.chat_with_tools("¿Ingresos de Apple en Q1 2020?")

        # Verificar que se hicieron 2 llamadas HTTP (1ª decisión + 2ª respuesta)
        assert mock_post.call_count == 2
        # Verificar que se ejecutó la tool
        mock_tool_manager.execute_tool_call.assert_called_once()
        assert isinstance(result, str)

    @patch("src.rag.generation.llm_groq.requests.post")
    @patch("src.rag.generation.llm_groq.tool_manager")
    def test_api_error_returns_message(self, mock_tool_manager, mock_post):
        """Un error HTTP devuelve mensaje de error en vez de crashear."""
        import requests as real_requests
        mock_post.side_effect = real_requests.exceptions.ConnectionError("timeout")

        from src.rag.generation.llm_groq import GroqLLM
        llm = GroqLLM(api_key="fake-key-for-testing")

        result = llm.chat_with_tools("¿Qué tal?")

        assert "Error" in result
        assert isinstance(result, str)

    @patch("src.rag.generation.llm_groq.requests.post")
    @patch("src.rag.generation.llm_groq.tool_manager")
    @patch("src.rag.generation.llm_groq.os.getenv", return_value=None)
    def test_missing_api_key_raises(self, mock_getenv, mock_tool_manager, mock_post):
        """Sin API key, el constructor lanza ValueError."""
        from src.rag.generation.llm_groq import GroqLLM

        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            GroqLLM(api_key=None)
