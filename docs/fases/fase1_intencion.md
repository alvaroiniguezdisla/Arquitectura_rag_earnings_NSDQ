# Fase 1: Intencion y Decision (Tool Calling)

## Archivos y Componentes Clave

Estos son los "actores" que intervienen en esta fase:

1.  **`scripts/chat_cli.py` (La Ventanilla)**:
    *   **Rol**: Interfaz de usuario. Recibe el texto y muestra la respuesta.
    *   **Funcion Clave**: `main()` (el bucle while) llama a `llm.chat_with_tools(...)`.
2.  **`src/rag/generation/llm_groq.py` (El Cerebro/Mensajero)**:
    *   **Rol**: Prepara el paquete para la API, lo envia y procesa la respuesta.
    *   **Funcion Clave**: `chat_with_tools(...)`. Aqui ocurre toda la logica de decision.
3.  **`src/rag/generation/tools.py` (El Menu de Herramientas)**:
    *   **Rol**: Define QUE puede hacer el modelo (JSON Schemas) y COMO hacerlo (Python code).
    *   **Constante Clave**: `AVAILABLE_TOOLS_SCHEMAS` (el menu que lee el LLM).
    *   **Clase Clave**: `ToolManager` (el ejecutor real).
4.  **`src/rag/core/prompts.py` (La Personalidad)**:
    *   **Rol**: Instrucciones base ("Eres un experto financiero...").
5.  **`src/rag/core/memory.py` (La Libreta)**:
    *   **Rol**: Historial de chat para tener contexto.

---

## El Flujo Paso a Paso

Vamos a seguir el viaje de una pregunta: **"Cuales fueron los ingresos de Apple?"**.

### Paso 1: El Envio (Python -> Groq)
El usuario escribe la pregunta en `chat_cli.py`. El archivo `llm_groq.py` empaqueta todo en un JSON para enviarlo a la nube.

**Lo que enviamos a la API:**
```json
{
  "messages": [
    {"role": "system", "content": "Eres un experto financiero..."},
    {"role": "user", "content": "Hola, soy Juan"},
    {"role": "assistant", "content": "Hola Juan"},
    {"role": "user", "content": "Cuales fueron los ingresos de Apple?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_earnings_calls",
        "description": "Busca datos financieros...",
        "parameters": { ... }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "list_available_companies",
        "description": "Lista empresas disponibles..."
      }
    }
  ],
  "tool_choice": "auto"
}
```
*Le decimos al LLM: "Aqui tienes la pregunta, lo que hemos hablado antes, y 2 herramientas. Tu decides si usarlas."*

### Paso 2: La Decision (En la Nube)
El modelo Llama 3.3 recibe el paquete. Razona que no puede inventarse los ingresos de Apple. Decide que necesita usar la herramienta `search_earnings_calls`.

**Lo que Groq nos devuelve:**
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
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
**Nota**: `content` es `null` porque el LLM no te esta hablando a ti, esta pidiendo ayuda al sistema.

### Paso 3: La Recepcion e Interpretacion (Python)
De vuelta en `llm_groq.py`, nuestro codigo abre este paquete.

1.  **Chequeo**: Mira si `message.get("tool_calls")` tiene algo.
2.  **Caso SI (Tiene tool_calls)**:
    *   El LLM quiere buscar.
    *   El codigo pausa la conversacion ("Pensando...").
    *   Entra en el bucle `for tool_call in tool_calls:`.
    *   Llama al `ToolManager` para ejecutar la funcion REAL en Python.
    *   **AQUI TERMINA LA FASE 1 Y EMPIEZA LA FASE 2 (Recuperacion).**

3.  **Caso NO (El usuario dijo "Hola")**:
    *   `tool_calls` vendria vacio y `content` tendria "Hola! En que puedo ayudarte?".
    *   El codigo simplemente devuelve ese `content` y termina.

---

## Por que hacemos esto?
Este diseno desacopla la **INTENCION** (Fase 1) de la **EJECUCION** (Fase 2).
*   El LLM es el "cerebro" que decide QUE necesitamos.
*   Nuestro codigo es el "musculo" que busca el dato.
*   Si manana queremos anadir una herramienta de "Calculadora", solo la anadimos a `tools.py` y el LLM sabra cuando usarla, sin cambiar nada mas.

---

## Futuras Mejoras

1.  **Validacion de Argumentos Robusta**:
    *   El LLM a veces puede alucinar argumentos (ej: `year="last year"` en vez de `2023`). Se podria anadir validacion con Pydantic antes de ejecutar la tool.
2.  **Multiturn Tool Use (Agente Real)**:
    *   Ahora mismo: Pregunta -> Tool -> Respuesta.
    *   Futuro: Pregunta -> Tool 1 -> Tool 2 (basada en el resultado de la 1) -> Respuesta.
3.  **Feedback Loop en Error**:
    *   Si la tool falla, el LLM podria leer el error e intentar reformular su peticion automaticamente.
4.  **Desambiguacion con el Usuario**:
    *   Si la intencion no es clara (ej: "Cuanto gano?"), el LLM podria preguntar "Te refieres a Apple o Microsoft?" antes de llamar a la tool (human-in-the-loop).
