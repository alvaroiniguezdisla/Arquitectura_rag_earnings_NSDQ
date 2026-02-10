# Roadmap: Proyecto RAG Financiero Local

Este documento define el plan **paso a paso** para construir nuestro sistema, priorizando simplicidad y buenas prácticas, como pediste.

## 🎯 Objetivo
Crear un asistente financiero capaz de leer transcripts de earnings (2019-2020) y responder preguntas citando fuentes, todo ejecutándose en tu PC.

## 🏗️ Arquitectura (Decisiones Clave)
Basado en lo que hemos hablado y las mejores prácticas para un MVP educativo:

*   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`. (Estándar, rápido, funciona en CPU).
*   **Vector DB**: `FAISS` + `SQLite`. 
    *   *Por qué:* FAISS es el motor de búsqueda vectorial por excelencia (Facebook Research). SQLite nos da la robustez para guardar los textos y metadatos sin depender de servidores externos complejos. Es la forma "artesanal" y robusta de hacerlo.
*   **LLM**: `Llama 3.3 70B Versatile` (vía Groq API en la nube).
*   **Chunking**: Ventana fija con solape (Simple y efectivo para empezar).

---

## 📅 Fases del Proyecto

### ✅ Fase 1: Datos (Completada)
*   [x] Descarga de Kaggle automatizada (`scripts/kaggle_download.py`)
*   [x] Limpieza y normalización de textos 2019-2020 (`scripts/prepare_corpus.py`)
*   [x] Resultado: `data/processed/corpus.jsonl` (4.3MB, ~200 documentos)

### ✅ Fase 2: Cimientos del Código (Completada)
Antes de procesar nada, necesitamos organizar la "casa".
*   [x] **Configuración Central (`src/rag/config.py`)**: Rutas, tamaños, modelos centralizados
*   [x] **Definiciones (`src/rag/schema.py`)**: Clases `Document`, `Chunk`, `RetrievedChunk`

### ✅ Fase 3: Procesamiento (Completada - La "fábrica" de vectores)
*   [x] **Ingesta (`src/rag/ingest.py`)**: Leer `corpus.jsonl` y convertir a objetos `Document`
*   [x] **Chunking (`src/rag/chunking.py`)**: Cortar textos en trozos de 800 caracteres con solape
*   [x] **Embeddings (`src/rag/embeddings.py`)**: Convertir texto a vectores con MiniLM
*   [x] **Indexado (`scripts/build_index.py`)**: Guardar en FAISS (vectores) y SQLite (metadata)

### ✅ Fase 4: El Cerebro — RAG (Completada)
*   [x] **Búsqueda (`src/rag/retriever.py`)**: Retriever que combina FAISS + SQLite para búsqueda semántica
*   [x] **Generación (`src/rag/llm_groq.py`)**: Wrapper de Groq API (Llama 3.3 70B) que genera respuestas con contexto RAG
*   [x] **Bases de datos vectorial (`src/rag/vdb_faiss.py`)**: Gestión completa del índice FAISS (add, search, save, load)
*   [x] **Base de datos metadata (`src/rag/vdb_sqlite.py`)**: Gestión SQLite para texto y metadata de chunks

### ✅ Fase 5: Interfaz (Completada)
*   [x] **Chat CLI (`scripts/chat_cli.py`)**: Interfaz de línea de comandos interactiva con soporte de exit/quit/help

### ✅ Fase 6: Tests (Completada)
*   [x] Test de ingesta (`tests/test_ingest.py`)
*   [x] Test de chunking (`tests/test_chunking.py`)
*   [x] Test de embeddings (`tests/test_embeddings.py`)
*   [x] Test de VDB FAISS+SQLite (`tests/test_vdb.py`)
*   [x] Test de retriever (`tests/test_retriever.py`)
*   [x] Test pipeline end-to-end (`tests/test_rag_pipeline.py`)

---

## 📝 Estado Actual
**🎉 MVP COMPLETADO.** El pipeline RAG funciona end-to-end:
- Datos descargados, limpiados e indexados (6176 chunks, 384 dimensiones)
- Búsqueda semántica operativa con FAISS + SQLite
- Generación de respuestas con Groq API (Llama 3.3 70B Versatile)
- Chat CLI funcional desde terminal
- Suite de tests completa

---

## 🔜 Posibles Mejoras Futuras
*   [ ] **Evaluación cuantitativa**: Métricas de precisión (MRR, recall@k) sobre un set de preguntas gold
*   [ ] **Interfaz web**: Streamlit o Gradio para una UI más visual
*   [ ] **Chunking inteligente**: Usar separadores semánticos (por párrafos/secciones) en vez de ventana fija
*   [ ] **Re-ranking**: Añadir un modelo de cross-encoder para reordenar resultados
*   [ ] **Historial de conversación**: Mantener contexto entre preguntas en el chat
*   [ ] **requirements.txt**: Crear archivo de dependencias del proyecto
*   [ ] **Docker**: Containerizar la aplicación para despliegue fácil
