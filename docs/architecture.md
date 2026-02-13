# Arquitectura del Sistema RAG - Earnings Calls NASDAQ

## 1. Mapa de Componentes

El sistema tiene **6 componentes principales** organizados en 5 capas:

```mermaid
graph TB
    subgraph USUARIO ["CAPA 1: Interfaz"]
        CLI["chat_cli.py"]
    end

    subgraph CEREBRO ["CAPA 2: Cerebro"]
        MEM["MemoryManager"]
        PROMPT["System Prompt"]
        LLM["GroqLLM - Llama 3.3 70B"]
    end

    subgraph HERRAMIENTAS ["CAPA 3: Herramientas"]
        TM["ToolManager"]
        T1["search_earnings_calls"]
        T2["list_available_companies"]
    end

    subgraph BUSQUEDA ["CAPA 4: Motor de Busqueda"]
        RET["Retriever"]
        EMB["EmbeddingModel - MiniLM-L6-v2"]
    end

    subgraph DATOS ["CAPA 5: Almacenamiento"]
        UDS["UnifiedDocumentStore"]
        DB[("unified_store.db - SQLite")]
    end

    subgraph ML ["CAPA 6: Machine Learning"]
        PRED["FinancialPredictor"]
        MODEL["RandomForest .pkl"]
    end

    CLI -->|"pregunta + historial"| LLM
    MEM -->|"ultimos 10 msgs"| LLM
    PROMPT -->|"personalidad + reglas"| LLM

    LLM -->|"tool_calls JSON"| TM
    TM --> T1
    TM --> T2
    TM --> T3["predict_financial_outlook"]
    
    T1 --> RET
    T2 --> RET
    T3 --> RET
    T3 --> PRED
    
    RET --> EMB
    RET --> UDS
    UDS --> DB
    PRED --> MODEL
```

**Que hace cada capa:**

| Capa | Componente | Responsabilidad |
|------|-----------|-----------------|
| 1. Interfaz | `chat_cli.py` | Lee preguntas del usuario, muestra respuestas |
| 2. Cerebro | `GroqLLM` + `MemoryManager` + `System Prompt` | Decide si usar tools, genera respuestas, recuerda contexto |
| 3. Herramientas | `ToolManager` + 2 tools | Enruta las peticiones del LLM al componente correcto |
| 4. Busqueda | `Retriever` + `EmbeddingModel` | Convierte texto a vectores y coordina la busqueda semantica |
| 5. Almacenamiento | `UnifiedDocumentStore` + SQLite | Guarda y busca chunks (texto + metadata + vectores) |
| 6. Machine Learning | `FinancialPredictor` | Analiza sentimiento y genera predicciones de outlook (Pos/Neg) |

---

## 2. Flujo de Decision del LLM

Cada pregunta pasa por esta logica. Hay **4 caminos posibles**:

```mermaid
flowchart TD
    A["Pregunta del usuario"] --> B{"LLM analiza:<br/>Necesita datos financieros?"}

    B -->|"CAMINO 1<br/>Saludo, opinion, general..."| C["Responde directamente<br/>SIN buscar en BD"]
    B -->|"CAMINO 2<br/>Datos, ingresos, estrategia..."| D["search_earnings_calls"]
    B -->|"CAMINO 3<br/>Que empresas hay?"| E["list_available_companies"]
    B -->|"CAMINO 4<br/>Prediccion, Outlook futuro..."| M["predict_financial_outlook"]

    C --> R["Respuesta al usuario"]

    D --> F["ToolManager -> Retriever"]
    F --> G["Embedding + Busqueda vectorial"]
    G --> H["Filtrado por empresa + Boost temporal"]
    H --> I["LLM recibe chunks reales"]
    I --> R

    E --> J["ToolManager -> Retriever -> BD"]
    J --> K["SELECT DISTINCT companies"]
    K --> L["LLM recibe lista de tickers"]
    L --> R

    M --> P1["ToolManager -> Retriever (get chunks)"]
    P1 --> P2["FinancialPredictor.predict(text)"]
    P2 --> P3["RandomForest Inference"]
    P3 --> P4["JSON: {prediction: 'POSITIVE', confidence: 0.85}"]
    P4 --> R
```

**Ejemplos reales por camino:**

| Camino | Ejemplo | Que pasa | Llamadas al LLM |
|--------|---------|----------|-----------------|
| 1 - Sin tools | "Hola, que tal?" | Responde directo. No toca la BD | 1 |
| 2 - Search | "Ingresos de Apple en Q3 2020?" | Busca en BD, filtra AAPL, devuelve chunks | 2 |
| 3 - List | "Que empresas tienes?" | Query SQL simple, devuelve tickers | 2 |
| 4 - Predict | "Cual es el outlook de Apple para 2020?" | Recupera texto -> ML Model -> Prediccion JSON | 2 |

---

## 3. Flujo Detallado: Camino 1 — Sin Tools

Cuando el LLM decide que **NO necesita** buscar en la base de datos:

```mermaid
sequenceDiagram
    actor U as Usuario
    participant CLI as chat_cli.py
    participant MEM as MemoryManager
    participant LLM as Groq API

    U->>CLI: "Hola, quien eres?"

    CLI->>MEM: get_messages()
    MEM-->>CLI: historial previo (si hay)

    Note over CLI: Construye payload con<br/>System Prompt + Historia + Pregunta

    CLI->>LLM: POST /chat/completions con tools disponibles

    Note over LLM: Chain of Thought:<br/>"Es un saludo.<br/>No necesito datos financieros.<br/>Respondo directamente."

    LLM-->>CLI: content = "Hola! Soy un asistente..."<br/>tool_calls = null

    Note over CLI: tool_calls es NULL<br/>Devuelve content directamente

    CLI->>MEM: guarda pregunta + respuesta
    CLI-->>U: "Hola! Soy un asistente financiero..."

    Note right of CLI: COSTE: 1 llamada LLM<br/>0 busquedas BD<br/>0 embeddings
```

---

## 4. Flujo Detallado: Camino 2 — search_earnings_calls

Este es el flujo **mas complejo**. El LLM hace **2 llamadas**:

```mermaid
sequenceDiagram
    actor U as Usuario
    participant CLI as chat_cli.py
    participant MEM as MemoryManager
    participant LLM as Groq API
    participant TM as ToolManager
    participant RET as Retriever
    participant EMB as EmbeddingModel
    participant DB as UnifiedStore

    U->>CLI: "Ingresos de Apple en Q3 2020?"
    CLI->>MEM: get_messages()
    MEM-->>CLI: historial

    Note over CLI,LLM: === LLAMADA 1 al LLM (con tools) ===
    CLI->>LLM: messages + tool_schemas + tool_choice auto

    Note over LLM: "Pregunta financiera!<br/>Necesito search_earnings_calls.<br/>company_id = AAPL"

    LLM-->>CLI: tool_calls = search_earnings_calls<br/>args: query="Apple revenue Q3 2020", company_id="AAPL"

    Note over CLI,TM: === EJECUCION DE TOOLS ===
    CLI->>TM: execute_tool_call(tool_call)
    TM->>RET: search(query, filter="AAPL")
    RET->>EMB: embed_text("Apple revenue Q3 2020")
    EMB-->>RET: vector float32 x 384 dimensiones

    RET->>DB: search(vector, company="AAPL")
    Note over DB: 1. Distancia L2 con numpy<br/>2. Filtra solo AAPL<br/>3. Boost +5% si anno=2020<br/>4. Boost +3% si quarter=Q3
    DB-->>RET: Top 10 RetrievedChunks
    RET-->>TM: chunks con texto + score + metadata
    TM-->>CLI: JSON con 10 fragmentos reales

    Note over CLI,LLM: === LLAMADA 2 al LLM (con resultados) ===
    CLI->>LLM: mensajes originales + tool_results JSON

    Note over LLM: "Tengo datos reales de los earnings calls.<br/>Genero respuesta basada en estos fragmentos."
    LLM-->>CLI: "Apple's revenues in Q3 2020 were $59.5 billion..."

    CLI->>MEM: guarda pregunta + respuesta
    CLI-->>U: Respuesta con datos verificados

    Note right of CLI: COSTE: 2 llamadas LLM<br/>1 embedding<br/>1 busqueda vectorial BD
```

---

## 5. Flujo Detallado: Camino 3 — list_available_companies

El mas simple de los que usan tools:

```mermaid
sequenceDiagram
    actor U as Usuario
    participant CLI as chat_cli.py
    participant LLM as Groq API
    participant TM as ToolManager
    participant RET as Retriever
    participant DB as UnifiedStore

    U->>CLI: "Que empresas tienes?"

    Note over CLI,LLM: === LLAMADA 1 ===
    CLI->>LLM: messages + tool_schemas

    Note over LLM: "Pregunta sobre inventario.<br/>Uso list_available_companies."
    LLM-->>CLI: tool_calls = list_available_companies (sin args)

    Note over CLI,TM: === EJECUCION ===
    CLI->>TM: execute_tool_call(tool_call)
    TM->>RET: get_available_companies()
    RET->>DB: get_companies()
    Note over DB: SELECT DISTINCT company<br/>FROM chunks metadata
    DB-->>RET: AAPL, AMD, AMZN, ASML, CSCO,<br/>GOOGL, INTC, MSFT, MU, NVDA
    RET-->>TM: lista de tickers
    TM-->>CLI: JSON con 10 empresas

    Note over CLI,LLM: === LLAMADA 2 ===
    CLI->>LLM: mensajes + tool_results
    LLM-->>CLI: "Tengo info de: Apple (AAPL), AMD,<br/>Amazon (AMZN)..."

    CLI-->>U: Lista formateada de empresas

    Note right of CLI: COSTE: 2 llamadas LLM<br/>0 embeddings<br/>1 query SQL simple
```

---

## 6. Flujo Detallado: Camino 4 — Prediccion Financiera (ML)

Integra RAG clásico con un modelo de clasificación:

```mermaid
sequenceDiagram
    actor U as Usuario
    participant CLI as chat_cli.py
    participant LLM as Groq API
    participant TM as ToolManager
    participant RET as Retriever
    participant PRED as FinancialPredictor

    U->>CLI: "Cual es el outlook de Apple para Q3 2020?"

    Note over CLI,LLM: === LLAMADA 1 ===
    CLI->>LLM: messages + tools

    Note over LLM: "Pide una prediccion/perspectiva.<br/>Uso predict_financial_outlook."
    LLM-->>CLI: tool_calls = predict_financial_outlook(ticker="AAPL", year=2020, quarter=3)

    Note over CLI,TM: === EJECUCION ===
    CLI->>TM: execute_tool_call()
    TM->>RET: search(filter="AAPL", year=2020, quarter=3)
    RET-->>TM: Lista de Chunks (Texto del transcript)
    
    TM->>PRED: predict(text_chunks, revenue=None)
    Note over PRED: 1. Limpieza de texto<br/>2. Extraccion features (Sentiment, Words)<br/>3. Scaling<br/>4. Inference (RandomForest)
    PRED-->>TM: JSON {prediction: "POSITIVE", probability: 0.85, ...}
    
    TM-->>CLI: Resultado de la Tool

    Note over CLI,LLM: === LLAMADA 2 ===
    CLI->>LLM: mensajes + prediccion JSON
    LLM-->>CLI: "El modelo predice una perspectiva POSITIVA con 85% de confianza..."

    CLI-->>U: Respuesta final
```

---

## 6. La Busqueda Vectorial en Detalle

Cuando `search_earnings_calls` se ejecuta, hay **4 fases internas**:

```mermaid
flowchart TD
    subgraph F1["FASE 1 - Vectorizacion"]
        Q["Query: 'Apple revenue Q3 2020'"] --> EMB["MiniLM-L6-v2"]
        EMB --> VEC["Vector: float32 x 384 dims"]
    end

    subgraph F2["FASE 2 - Busqueda Bruta con NumPy"]
        VEC --> L2["Calcula distancia L2 con los 6176 vectores en cache RAM"]
        L2 --> TOP["np.argpartition: selecciona los N mas cercanos"]
    end

    subgraph F3["FASE 3 - Filtrado por Empresa"]
        TOP --> TICK["Ticker Mapping: 'Apple' se convierte en 'AAPL'"]
        TICK --> FILT["Elimina chunks de empresas que NO son AAPL"]
    end

    subgraph F4["FASE 4 - Reranking con Boost Temporal"]
        FILT --> SCORE["Score base: 1 / 1+distancia"]
        SCORE --> BY{"Anno del chunk = anno de la query?"}
        BY -->|SI| BOOST1["+5% al score"]
        BY -->|NO| BQ
        BOOST1 --> BQ{"Quarter del chunk = quarter de la query?"}
        BQ -->|SI| BOOST2["+3% al score"]
        BQ -->|NO| FINAL["Ordena por score final descendente"]
        BOOST2 --> FINAL
    end

    FINAL --> OUT["Top 10 RetrievedChunks al LLM"]
```

### Ticker Mapping Completo

El sistema traduce nombres comunes a tickers automaticamente:

| El usuario dice... | El sistema busca... |
|--------------------|---------------------|
| "Apple", "apple" | AAPL |
| "Google", "Alphabet" | GOOGL |
| "Microsoft" | MSFT |
| "Amazon" | AMZN |
| "NVIDIA", "Nvidia" | NVDA |
| "Intel" | INTC |
| "AMD" | AMD |
| "Micron" | MU |
| "ASML" | ASML |
| "Cisco" | CSCO |

---

## 7. Las 3 Herramientas (Tool Schemas)

El LLM recibe estas definiciones en **cada llamada**. Son su "menu":

### Tool 1: `search_earnings_calls`

| Campo | Valor |
|-------|-------|
| **Cuando usarla** | Cualquier pregunta sobre datos financieros, ingresos, estrategia, CEO, earnings |
| **Parametro `query`** | Obligatorio. Pregunta o keywords. Ej: "Apple revenue Q4 2019" |
| **Parametro `company_id`** | Opcional pero recomendado. Ticker o nombre. Ej: "AAPL", "Apple" |
| **Parametro `num_results`** | Opcional (default 10). Cuantos fragmentos devolver |
| **Retorna** | JSON array: `[{company, year, quarter, content, score}]` |

### Tool 2: `list_available_companies`

| Campo | Valor |
|-------|-------|
| **Cuando usarla** | Solo si preguntan "que empresas tienes?" |
| **Parametros** | Ninguno |
| **Retorna** | JSON array de tickers: `["AAPL", "AMD", ...]` |

### Tool 3: `predict_financial_outlook` (NUEVA)

| Campo | Valor |
|-------|-------|
| **Cuando usarla** | Preguntas sobre futuro, predicciones, outlook, perspectivas |
| **Parametros** | `company_id` (str), `year` (int), `quarter` (int) |
| **Retorna** | JSON: `{"prediction": "POSITIVE", "probability": float, "features": {...}}` |

### Como decide el LLM?

Se le envia un **System Prompt** con instrucciones tipo Chain of Thought:

```
1. Analiza la pregunta. Pide datos especificos (ingresos, estrategia, CEO)?
2. SI -> USA search_earnings_calls. No respondas de memoria.
3. NO -> Responde directamente.
```

El parametro `tool_choice: "auto"` permite que el LLM decida libremente en cada turno.

---

## 8. Pipeline Offline (Ingesta - Solo una vez)

Se ejecuta con: `python scripts/run_full_indexing.py`

```mermaid
flowchart LR
    KAG["JSONs Kaggle<br/>Earnings 2019-2020"] --> PC["prepare_corpus.py<br/>Limpia y extrae metadata"]
    PC --> CORPUS["corpus.jsonl<br/>97 documentos"]
    CORPUS --> S1["Step 1: Loader<br/>Lee JSONL"]
    S1 --> S2["Step 2: Chunking<br/>~800 chars, overlap 200"]
    S2 --> S3["Step 3: Embedding<br/>MiniLM texto->vector 384d"]
    S3 --> S4["Step 4: Indexing<br/>Guarda en SQLite"]
    S4 --> DB[("unified_store.db<br/>6176 chunks<br/>~25MB")]
```

### Esquema de la Base de Datos

Toda la info vive en **una tabla, un archivo**:

| Columna | Tipo | Para que sirve |
|---------|------|----------------|
| `chunk_id` | TEXT PK | ID unico del fragmento |
| `doc_id` | TEXT | Transcript padre (ej: "AAPL_Q3_2020") |
| `text` | TEXT | Fragmento de texto real (~800 chars) |
| `chunk_index` | INTEGER | Posicion dentro del documento |
| `metadata` | TEXT (JSON) | `{"company":"AAPL", "year":"2020", "quarter":"Q3"}` |
| `embedding` | BLOB | Vector de 384 float32 (~1.5KB) |

---

## 9. La Memoria (Contexto Conversacional)

El `MemoryManager` mantiene una **ventana deslizante** de los ultimos 10 mensajes:

```mermaid
sequenceDiagram
    actor U as Usuario
    participant MEM as MemoryManager
    participant LLM as LLM

    U->>MEM: "Ingresos de Apple en 2020?"
    Note over MEM: memoria = [user msg 1]
    MEM->>LLM: system + user: "Ingresos Apple?"
    LLM-->>MEM: "Apple registro $59.5B..."
    Note over MEM: memoria = [user 1, asst 1]

    U->>MEM: "Y en 2019?"
    Note over MEM: memoria = [user 1, asst 1, user 2]
    MEM->>LLM: system + user 1 + asst 1 + user 2
    Note over LLM: SABE que "en 2019"<br/>se refiere a Apple<br/>por la historia previa
    LLM-->>MEM: "En 2019, Apple reporto..."
    Note over MEM: memoria = [user 1, asst 1, user 2, asst 2]

    Note over MEM: Si len > 10 mensajes:<br/>Elimina los mas antiguos<br/>(ventana deslizante)
```

Esto permite **preguntas de seguimiento** sin repetir el contexto.

---

## 10. Pila Tecnologica

| Capa | Tecnologia | Justificacion |
|------|------------|---------------|
| **LLM** | Llama 3.3 70B via Groq API | Open Source, rendimiento top, inferencia <1s |
| **Embeddings** | all-MiniLM-L6-v2 | Rapido en CPU, 384 dims, buena calidad semantica |
| **Base de Datos** | SQLite (stdlib Python) | Un solo archivo, sin servidor, texto + vectores juntos |
| **Busqueda Vectorial** | NumPy (distancia L2 en RAM) | Simple, eficiente para ~6K vectores |
| **Interfaz** | CLI (Python input/print) | Rapido de iterar |
| **Orquestacion** | Python nativo | Control total, sin LangChain/LlamaIndex |
| **API HTTP** | requests | Llamadas directas a Groq |
| **Machine Learning** | scikit-learn | RandomForestClassifier, StandardScaler, TF-IDF/TextBlob |

---

## 11. Estructura de Archivos

```text
src/rag/
  core/
    config.py        # Rutas, constantes, modelos (centralizado)
    schema.py        # Dataclasses: Document, Chunk, RetrievedChunk
    prompts.py       # System prompt del LLM (chain of thought)
    memory.py        # Ventana deslizante de conversacion (10 msgs)
  pipeline/
    step1_loader.py      # Lee corpus.jsonl -> List of Documents
    step2_chunking.py    # Document -> Chunks (800 chars, 200 overlap)
    step3_embedding.py   # Texto -> vector de 384 dims (MiniLM)
    step4_indexing.py    # Chunks + vectors -> SQLite
  storage/
    unified_store.py     # BD unica: texto + metadata + vectores como BLOBs
  retrieval/
    retriever.py         # Coordina: ticker mapping + busqueda + boost
  generation/
    llm_groq.py          # Groq API wrapper + loop de tool calling (2 pasos)
    tools.py             # Schemas de 2 tools + ToolManager (router)

scripts/
  run_full_indexing.py   # Pipeline offline completo (steps 1-4)
  chat_cli.py            # Chat interactivo (punto de entrada)
  prepare_corpus.py      # Datos crudos Kaggle -> corpus.jsonl
```
