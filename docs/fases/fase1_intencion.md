# Fase 1: Intención y Decisión (Tool Calling)

## 📂 Archivos y Componentes Clave

Estos son los "actores" que intervienen en esta fase. No pierdas de vista estos nombres:

1.  **`scripts/chat_cli.py` (La Ventanilla)**:
    *   **Rol**: Interfaz de usuario. Recibe el texto y muestra la respuesta.
    *   **Función Clave**: `main()` (el bucle while) llama a `llm.chat_with_tools(...)`.
2.  **`src/rag/generation/llm_groq.py` (El Cerebro/Mensajero)**:
    *   **Rol**: Prepara el paquete para la API, lo envía y procesa la respuesta.
    *   **Función Clave**: `chat_with_tools(...)`. Aquí ocurre toda la lógica de decisión.
3.  **`src/rag/generation/tools.py` (El Menú de Herramientas)**:
    *   **Rol**: Define QUÉ puede hacer el modelo (JSON Schemas) y CÓMO hacerlo (Python code).
    *   **Constante Clave**: `AVAILABLE_TOOLS_SCHEMAS` (el menú que lee el LLM).
    *   **Clase Clave**: `ToolManager` (el ejecutor real).
4.  **`src/rag/core/prompts.py` (La Personalidad)**:
    *   **Rol**: Instrucciones base ("Eres un experto financiero...").
5.  **`src/rag/core/memory.py` (La Libreta)**:
    *   **Rol**: Historial de chat para tener contexto.

---

## 🔄 El Flujo Paso a Paso (La "Magia" Explicada)

Vamos a seguir el viaje de una pregunta: **"¿Cuáles fueron los ingresos de Apple?"**.

### Paso 1: El Envío (Python -> Groq)
El usuario escribe la pregunta en `chat_cli.py`. El archivo `llm_groq.py` (línea 77) empaqueta todo en un JSON para enviarlo a la nube.

**Lo que enviamos a la API:**
```json
{
  "messages": [
    {"role": "system", "content": "Eres un experto financiero..."},
    // ... [Aquí van los últimos 10 mensajes de la memoria para dar contexto] ...
    {"role": "user", "content": "Hola, soy Juan"},
    {"role": "assistant", "content": "Hola Juan"},
    // ...
    {"role": "user", "content": "¿Cuáles fueron los ingresos de Apple?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_earnings_calls",
        "description": "Busca datos financieros...",
        "parameters": { ... }
      }
    }
  ],
  "tool_choice": "auto"
}
```
*Le decimos al LLM: "Mira, aquí tienes la pregunta **y lo que hemos hablado antes**, y aquí tienes una herramienta de búsqueda. Tú decides si usarla."*

### Paso 2: La Decisión (En la Nube ☁️)
El modelo Llama 3 recibe el paquete. Razona que no puede inventarse los ingresos de Apple. Decide que necesita usar la herramienta `search_earnings_calls`.

**Lo que Groq nos devuelve (`response.json()`):**
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,  // ¡OJO! Viene vacío porque no te está hablando, está pidiendo ayuda.
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "search_earnings_calls",
              "arguments": "{\"query\": \"ingresos Apple\", \"company_id\": \"AAPL\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

### Paso 3: La Recepción e Interpretación (Python)
De vuelta en `llm_groq.py` (línea 91), nuestro código abre este paquete.

1.  **Chequeo**: Mira si `message.get("tool_calls")` tiene algo.
2.  **Caso SI (Tiene tool_calls)**:
    *   ¡Bingo! El LLM quiere buscar.
    *   El código pausa la conversación con el usuario ("🤖 Pensando...").
    *   Entra en el bucle `for tool_call in tool_calls:`.
    *   Llama al `ToolManager` para ejecutar la función REAL en Python.
    *   **AQUÍ TERMINA LA FASE 1 Y EMPIEZA LA FASE 2 (Recuperación).**

3.  **Caso NO (El usuario dijo "Hola")**:
    *   `tool_calls` vendría vacío y `content` tendría "¡Hola! ¿En qué puedo ayudarte?".
    *   El código simplemente devuelve ese `content` y termina.

---

## 🧠 ¿Por qué hacemos esto?
Este diseño desacopla la **INTENCIÓN** (Fase 1) de la **EJECUCIÓN** (Fase 2).
*   El LLM es el "cerebro" que decide QUÉ necesitamos.
*   Nuestro código es el "músculo" que busca el dato.
*   Si mañana queremos añadir una herramienta de "Calculadora", solo la añadimos a `tools.py` y el LLM sabrá cuándo usarla, sin cambiar nada más.

---

## 🚀 Futuras Mejoras y Roadmap

En base a lo analizado, aquí es donde podemos pulir este diamante:

1.  **Validación de Argumentos Robusta**:
    *   El LLM a veces puede alucinar argumentos (ej: `year="last year"` en vez de `2023`). Podríamos añadir una capa de validación usando Pydantic antes de ejecutar la tool.
2.  **Multiturn Tool Use (Agente Real)**:
    *   Ahora mismo es: Pregunta -> Tool -> Respuesta.
    *   Futuro: Pregunta -> Tool 1 -> Tool 2 (basada en el resultado de la 1) -> Respuesta.
3.  **Feedback Loop en Error**:
    *   Si la tool falla (ej: error de red), ahora mismo devolvemos el error en JSON. Podríamos hacer que el LLM lo lea e intente corregir su petición (ej: reformular la query) automáticamente.
4.  **Desambiguación con el Usuario**:
    *   Si la intención no es clara (ej: "¿Cuánto ganó?"), el LLM podría preguntar "¿Te refieres a Apple o Microsoft?" antes de llamar a la tool (human-in-the-loop).
