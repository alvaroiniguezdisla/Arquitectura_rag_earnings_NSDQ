"""
Tests unitarios — MemoryManager (core/memory.py)

Módulo bajo test:
    src.rag.core.memory.MemoryManager

Qué se valida:
    - Que los mensajes se añaden y se recuperan respetando el orden (FIFO).
    - Que la ventana deslizante (sliding window) descarta los mensajes más
      antiguos cuando se supera el límite configurado.
    - Que un MemoryManager recién creado tiene historial vacío.

Estrategia:
    Tests de caja blanca sobre la estructura de datos interna.
    No requiere dependencias externas.
"""
import pytest
from src.rag.core.memory import MemoryManager


class TestMemoryManager:

    def test_add_and_get_messages(self):
        """Añadir mensajes y recuperarlos en orden."""
        memory = MemoryManager(limit=10)

        memory.add_user_message("Hola")
        memory.add_assistant_message("¡Hola! ¿En qué te ayudo?")

        history = memory.get_messages()

        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hola"}
        assert history[1] == {"role": "assistant", "content": "¡Hola! ¿En qué te ayudo?"}

    def test_sliding_window_limit(self):
        """Al superar el límite, se eliminan los mensajes más antiguos."""
        memory = MemoryManager(limit=4)

        # Añadir 6 mensajes (3 pares user/assistant)
        for i in range(3):
            memory.add_user_message(f"Pregunta {i}")
            memory.add_assistant_message(f"Respuesta {i}")

        history = memory.get_messages()

        # Solo deben quedar los últimos 4
        assert len(history) == 4
        # Los primeros 2 (Pregunta 0, Respuesta 0) se eliminaron
        assert history[0]["content"] == "Pregunta 1"

    def test_empty_history(self):
        """Un MemoryManager nuevo tiene historial vacío."""
        memory = MemoryManager()

        assert len(memory.get_messages()) == 0
