# 🗺️ Roadmap de Mejoras — RAG Earnings NSDQ

> **Última actualización**: 16 Feb 2026
> Marca cada tarea con `[x]` cuando la completes.

---

## Fase 1: Higiene Crítica (🔴 P0)

> **Por qué primero**: Son bugs y fallos activos que afectan seguridad, estabilidad o que directamente rompen funcionalidad. No requieren rediseño, solo correcciones puntuales.

### 1.1 Eliminar `print(response_data)` de producción
- [x] ~~En `src/rag/generation/llm_groq.py` línea 90, eliminar o comentar `print(response_data)`~~ → Reemplazado por `logger.debug()`.
- **Por qué**: Vuelca toda la respuesta JSON de Groq (tokens, metadata, contenido) en la consola. Es un riesgo de seguridad y ruido innecesario.
- **Esfuerzo**: 1 minuto.

### 1.2 Arreglar test roto `test_rag_pipeline.py`
- [x] ~~En `tests/test_rag_pipeline.py` línea 50, cambiar `llm.generate_response(query, chunks)` por `llm.chat_with_tools(user_query=query)`~~ → Tests reescritos completamente con pytest y mocking.
- **Por qué**: `GroqLLM` no tiene método `generate_response`. Este test siempre falla con `AttributeError`.
- **Esfuerzo**: 5 minutos.

### 1.3 Añadir `timeout` y `retry` a las llamadas HTTP
- [x] En `llm_groq.py`, añadir `timeout=30` a ambos `requests.post(...)`.
- [x] Implementar retry con backoff exponencial (4 intentos) para errores 429/500/502/503/504 → Session + Retry adapter.
- **Por qué**: Sin timeout, una llamada puede colgar indefinidamente. Sin retry, un error transitorio de Groq (rate limit) mata la conversación entera.
- **Esfuerzo**: 30 minutos.

### 1.4 Reemplazar `print()` por `logging`
- [x] Crear un logger centralizado en `src/rag/core/logger.py`.
- [x] Reemplazar todos los `print()` del proyecto por `logger.info()`, `logger.debug()`, `logger.error()`.
- [x] Configurar nivel por defecto en `INFO` y permitir cambio vía variable de entorno `LOG_LEVEL`.
- **Por qué**: `print()` no permite controlar verbosidad, no tiene timestamps, no se puede redirigir a archivos, y no distingue entre info/error/debug.
- **Esfuerzo**: 1-2 horas.

### 1.5 Eliminar singleton global de `ToolManager`
- [x] Eliminar la línea `tool_manager = ToolManager()` del final de `tools.py`.
- [x] Mover la instanciación a `chat_cli.py` + inyección de dependencias en `GroqLLM.__init__(tool_manager=...)`.
- **Por qué**: Al importar `tools.py`, se carga automáticamente el Retriever + EmbeddingModel + SQLite + ML Model. Esto impide importar el módulo sin tener la BD real, y rompe cualquier test unitario aislado.
- **Esfuerzo**: 30 minutos.

---

## Fase 2: Calidad de Búsqueda (🟡 P1)

> **Por qué segundo**: Estas mejoras impactan directamente la **precisión de las respuestas** del sistema. Son el corazón del RAG.

### 2.1 Cosine Similarity en vez de L2
- [x] En `unified_store.py`, normalizar los vectores al cargarlos en memoria (`_load_vectors_to_memory`).
- [x] Reemplazar el cálculo L2 por dot product (`scores = cache @ query.T`).
- [x] Invertir el ranking: mayor score = mejor (en vez de menor distancia = mejor).
- **Por qué**: `all-MiniLM-L6-v2` está entrenado para cosine similarity. Usar L2 degrada el ranking.
- **Esfuerzo**: 20 minutos.

### 2.2 Chunking recursivo por frases
- [x] Implementar recursive splitter: separar por `\n\n` → `\n` → `. ` → ` `.
- [x] Cada chunk es una unidad semántica completa (no corta frases a medias).
- [x] Subir overlap a 200 caracteres (25% del chunk, estándar recomendado).
- [x] Re-ejecutar `run_full_indexing.py` para reconstruir la BD.
- **Por qué**: Cortar por caracteres puede partir una oración clave por la mitad, perdiendo significado.
- **Esfuerzo**: 1 hora.

### 2.3 Metadata como columnas SQL indexadas
- [ ] Añadir columnas `company TEXT`, `year INTEGER`, `quarter INTEGER` a tabla `chunks`.
- [ ] Crear índices SQL compuestos para filtrado rápido.
- [ ] Pre-filtrar en SQL ANTES de buscar vectores (en vez de post-filtrar ×500).
- [ ] Simplificar `get_companies()` a: `SELECT DISTINCT company FROM chunks`.
- **Por qué**: Actualmente parsea JSON de cada chunk en cada búsqueda. Con índices SQL es O(1).
- **Esfuerzo**: 1 hora.

### 2.4 Cross-encoder re-ranking
- [ ] Crear `src/rag/retrieval/reranker.py` con `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- [ ] Buscar top-30 con bi-encoder (rápido) → re-rankear con cross-encoder (preciso).
- [ ] Devolver solo los top-K refinados al LLM.
- **Por qué**: El patrón "retrieve & re-rank" mejora dramáticamente la precisión. Es estándar en 2025.
- **Esfuerzo**: 2 horas.

### 2.5 (Opcional) Hybrid Search: BM25 + Dense
- [ ] Añadir índice BM25 con `rank_bm25` (keyword search).
- [ ] Reciprocal Rank Fusion para combinar resultados BM25 + cosine.
- **Por qué**: Caza coincidencias textuales ("revenue Q4") y semánticas ("ingresos del trimestre").
- **Esfuerzo**: 2 horas.

---

## Fase 3: Robustez del Código (🟡 P1)

> **Por qué tercero**: Hace el proyecto mantenible y testeable a largo plazo.

### 3.1 Añadir `__init__.py` a todos los subpaquetes
- [ ] Crear `__init__.py` vacíos en: `core/`, `pipeline/`, `storage/`, `retrieval/`, `generation/`, `ml/`.
- **Por qué**: Define los paquetes explícitamente. Permite definir imports públicos y es buena práctica estándar.
- **Esfuerzo**: 10 minutos.

### 3.2 Tests unitarios con mocking
- [x] Refactorizar `test_tool_calling.py` para usar `unittest.mock.patch` en lugar de llamar a la API real de Groq.
- [x] Crear mocks de `Retriever` y `GroqLLM` para tests de `ToolManager`.
- [x] Los tests deben funcionar **sin** `GROQ_API_KEY` ni BD real. → 18 tests, todos pasan sin dependencias externas.
- **Por qué**: Tests que llaman APIs reales no se pueden usar en CI/CD, cuestan dinero, y fallan sin internet.
- **Esfuerzo**: 2 horas.

### 3.3 Implementar revenue real en el Predictor
- [ ] Opción A: Extraer revenue del texto del transcript usando regex/NER.
- [ ] Opción B: Añadir un dataset de revenue por empresa/trimestre como CSV.
- [ ] Eliminar el `current_revenue = 0.0` hardcodeado en `tools.py` línea 153.
- **Por qué**: El feature `revenue` del modelo ML siempre es 0, lo que reduce la precisión del predictor.
- **Esfuerzo**: 2 horas.

---

## Fase 4: Funcionalidades Avanzadas (🟢 P2)

> **Por qué cuarto**: Son mejoras de calidad que elevan el proyecto de MVP a producción.

### 4.1 Hybrid Search (BM25 + Vectorial)
- [ ] Instalar `rank_bm25`.
- [ ] Implementar búsqueda BM25 sobre los textos de chunks.
- [ ] Fusionar scores: `final_score = α * cosine_score + (1-α) * bm25_score` con `α=0.7`.
- **Por qué**: La búsqueda vectorial captura significado pero pierde keywords exactos. BM25 captura coincidencias exactas. Combinarlos cubre ambos flancos.
- **Esfuerzo**: 3 horas.

### 4.2 Streaming de respuestas
- [ ] Añadir parámetro `stream=True` a los payloads de Groq.
- [ ] Implementar lectura de Server-Sent Events (SSE) y mostrar token por token.
- **Por qué**: El usuario ve la respuesta progresivamente en vez de esperar varios segundos. Mejora la UX drásticamente.
- **Esfuerzo**: 2 horas.

### 4.3 Evaluación con RAGAS
- [ ] Instalar `ragas` o `deepeval`.
- [ ] Crear un dataset de evaluación con preguntas + respuestas esperadas.
- [ ] Medir métricas estándar: Faithfulness, Answer Relevancy, Context Precision.
- **Por qué**: Da métricas objetivas y comparables para medir el impacto de cada mejora.
- **Esfuerzo**: 3 horas.

### 4.4 Observabilidad (Tracing)
- [ ] Integrar LangFuse o Weights & Biases.
- [ ] Instrumentar: latencia por componente, tools usadas, chunks recuperados, tokens consumidos.
- **Por qué**: Permite diagnosticar problemas en producción y optimizar costes de API.
- **Esfuerzo**: 4 horas.

### 4.5 Guardar tool calls en la memoria
- [ ] Extender `MemoryManager` para almacenar mensajes de tipo `tool`.
- [ ] Incluir los resultados de tools en el historial para que el LLM pueda referenciar búsquedas anteriores.
- **Por qué**: Actualmente el LLM pierde el contexto de búsquedas pasadas entre turnos.
- **Esfuerzo**: 1 hora.

---

## Progreso General

| Fase | Tareas | Completadas |
|------|--------|-------------|
| 1. Higiene Crítica | 5 | 5/5 ✅ |
| 2. Calidad de Búsqueda | 4 | 0/4 |
| 3. Robustez del Código | 3 | 1/3 |
| 4. Funcionalidades Avanzadas | 5 | 0/5 |
| **TOTAL** | **17** | **6/17** |

