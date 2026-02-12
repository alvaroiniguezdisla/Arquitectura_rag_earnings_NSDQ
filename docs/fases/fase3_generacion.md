# Fase 3: Generación y Respuesta

## Descripción General
Una vez que tenemos los "ingredientes" (los fragmentos de texto recuperados en la Fase 2), el "chef" (el LLM) cocina la respuesta final.

## Flujo Detallado

1.  **Construcción del Contexto**:
    *   El sistema toma la lista de textos recuperados (Chunks).
    *   Los formatea como un bloque de texto JSON o texto plano para que el LLM los pueda leer.
    *   Este bloque se inyecta en la conversación como un mensaje con rol `tool` o `system` (dependiendo de la implementación exacta de la API).
2.  **Llamada Final al LLM**:
    *   Enviamos al LLM:
        1.  El Prompt del Sistema ("Eres un experto...").
        2.  El Historial de Chat.
        3.  La Pregunta del Usuario.
        4.  **El Resultado de la Búsqueda** (Contexto).
3.  **Síntesis**:
    *   El LLM lee la pregunta y el contexto proporcionado.
    *   Genera una respuesta en lenguaje natural que integra los datos encontrados.
    *   Si los datos no son suficientes, el LLM debería indicarlo (según su System Prompt).
4.  **Entrega**: La respuesta de texto se muestra al usuario en la terminal.

## Archivos Clave
*   `src/rag/generation/llm_groq.py`: Método `chat_with_tools` (específicamente la parte después de ejecutar las tools).
*   `src/rag/core/prompts.py`: Contiene las instrucciones que guían al LLM sobre cómo usar el contexto.
