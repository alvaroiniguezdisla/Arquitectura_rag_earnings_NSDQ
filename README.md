# RAG Earnings NSDQ - Asistente Financiero Inteligente

Sistema de **Retrieval-Augmented Generation (RAG)** agéntico para analizar transcripts de earnings calls del NASDAQ (2019-2020).

---

## Caracteristicas Principales

*   **Agente Autonomo**: El sistema decide cuando consultar la base de datos financiera y cuando responder directamente.
*   **Recuperacion Optimizada**:
    *   **Metadata Boost**: Prioriza resultados del anno y trimestre mencionados.
    *   **Ticker Mapping**: Traduce nombres de empresas (ej: "Microsoft") a sus simbolos (MSFT).
    *   **Busqueda Semantica**: All-MiniLM-L6 + SQLite con vectores como BLOBs.
*   **Transparencia Total**: El chat muestra que herramientas esta usando en tiempo real.

---

## Instalacion Rapida

### 1. Preparar el Entorno
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Configuracion (.env)
```env
GROQ_API_KEY=tu_gsk_clave_aqui
```

---

## Como Empezar

### Paso 1: Indexacion (Construir la BD)
Procesa los transcripts y crea la base de datos unificada.
```bash
python scripts/run_full_indexing.py
```

### Paso 2: Ejecutar el Chat
```bash
python scripts/chat_cli.py
```

---

## Ejemplos de Preguntas

*   **Basica**: "Cuales fueron los ingresos de Apple en el Q4 de 2019?"
*   **Estrategia**: "Que dijo el CEO de NVIDIA sobre la demanda de chips en 2020?"
*   **Meta**: "De que empresas tienes informacion disponible?"

---

## Estructura del Proyecto

```text
src/rag/
  core/          # Configuracion y esquemas
  pipeline/      # ETL: Loader, Chunking, Embedding, Indexing
  storage/       # BD unificada SQLite (texto + metadata + vectores)
  retrieval/     # Motor de busqueda semantica
  generation/    # LLM Groq + Tool Calling
```

---

## Documentacion Extendida
- [Arquitectura Detallada](docs/architecture.md)
- [Fases del RAG](docs/fases_rag.md)
