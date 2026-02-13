# Fase 3: Generacion y Respuesta

## Descripcion General
Una vez que tenemos los "ingredientes" (los fragmentos de texto recuperados en la Fase 2), el LLM genera la respuesta final para el usuario.

## Flujo Detallado

1.  **Construccion del Contexto**:
    *   El sistema toma la lista de textos recuperados (Chunks).
    *   Los formatea como un bloque JSON para que el LLM los pueda leer.
    *   Este bloque se inyecta en la conversacion como un mensaje con rol `tool`.
2.  **Llamada Final al LLM (Segunda llamada)**:
    *   Enviamos al LLM:
        1.  El Prompt del Sistema ("Eres un experto...").
        2.  El Historial de Chat (memoria, ultimos 10 msgs).
        3.  La Pregunta del Usuario.
        4.  La respuesta del asistente pidiendo tools (primer turno).
        5.  **El Resultado de la Tool** (mensaje con rol `tool`).
            *   Puede ser una lista de textos (si usó search).
            *   Puede ser una predicción JSON (si usó predict).
3.  **Sintesis**:
    *   El LLM lee la pregunta y el contexto proporcionado.
    *   Genera una respuesta en lenguaje natural que integra los datos encontrados.
    *   Si los datos no son suficientes, el LLM indica que no encontro informacion en la BD 2019-2020.
4.  **Entrega**: La respuesta de texto se muestra al usuario en la terminal.

## Archivos Clave
*   `src/rag/generation/llm_groq.py`: Metodo `chat_with_tools` (la parte despues de ejecutar las tools, la "segunda llamada").
*   `src/rag/core/prompts.py`: Contiene las instrucciones que guian al LLM sobre como usar el contexto.
