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
- [x] Añadir columnas `company TEXT`, `year INTEGER`, `quarter INTEGER` a tabla `chunks`.
- [x] Crear índices SQL compuestos para filtrado rápido.
- [x] Pre-filtrar en SQL ANTES de buscar vectores (en vez de post-filtrar ×500).
- [x] Simplificar `get_companies()` a: `SELECT DISTINCT company FROM chunks`.
- **Por qué**: Actualmente parsea JSON de cada chunk en cada búsqueda. Con índices SQL es O(1).
- **Esfuerzo**: 1 hora.

### 2.4 Cross-encoder re-ranking
- [x] Crear `src/rag/retrieval/reranker.py` con `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- [x] Buscar top-30 con bi-encoder (rápido) → re-rankear con cross-encoder (preciso).
- [x] Devolver solo los top-K refinados al LLM.
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
- [x] Crear `__init__.py` vacíos en: `core/`, `pipeline/`, `storage/`, `retrieval/`, `generation/`, `ml/`.
- **Por qué**: Define los paquetes explícitamente. Permite definir imports públicos y es buena práctica estándar.
- **Esfuerzo**: 10 minutos.

### 3.2 Tests unitarios con mocking
- [x] Refactorizar `test_tool_calling.py` para usar `unittest.mock.patch` en lugar de llamar a la API real de Groq.
- [x] Crear mocks de `Retriever` y `GroqLLM` para tests de `ToolManager`.
- [x] Los tests deben funcionar **sin** `GROQ_API_KEY` ni BD real. → 18 tests, todos pasan sin dependencias externas.
- **Por qué**: Tests que llaman APIs reales no se pueden usar en CI/CD, cuestan dinero, y fallan sin internet.
- **Esfuerzo**: 2 horas.

### 3.3 Implementar revenue real en el Predictor
- [x] Opción A: Extraer revenue del texto del transcript usando regex/NER. -- Implementado regex robusto.
- [ ] Opción B: Añadir un dataset de revenue por empresa/trimestre como CSV.
- [x] Eliminar el `current_revenue = 0.0` hardcodeado en `tools.py` línea 153.
- **Por qué**: El feature `revenue` del modelo ML siempre es 0, lo que reduce la precisión del predictor.
- **Esfuerzo**: 2 horas.

---

## Fase 4: Plan Maestro de Optimización ML (🏆 The "Time Capsule" Strategy)

> **Objetivo**: Crear un modelo de Inteligencia Artificial que sea matemáticamente válido para predecir el futuro (2020) basándose solo en el pasado (2016-2019), sin descargar datos externos.

### 4.1 Ingeniería de Features (El Cimiento Matemático)
**El Problema**: El modelo actual usa "Dólares Absolutos" ($). En 2016, ganar $50B era un récord. En 2020, es poco. Esto confunde al modelo.
**La Solución**: Enseñar al modelo a pensar en **% de Crecimiento**, que es una métrica universal y atemporal.

- [ ] **Paso 1: Calcular Growth en Training Data** (`train_ml_model.py`)
    - [ ] Ordenar datos por fecha.
    - [ ] Calcular `revenue_growth_qoq` = `(Revenue_Q - Revenue_Q-1) / Revenue_Q-1`.
    - [ ] Calcular `revenue_growth_yoy` = `(Revenue_Q - Revenue_Q-4) / Revenue_Q-4`.
- [ ] **Paso 2: Re-entrenar Scaler**
    - [ ] Eliminar la columna `revenue` ($) de las features de entrada.
    - [ ] Añadir `revenue_growth_qoq` y `revenue_growth_yoy` al `StandardScaler`.
- [ ] **Paso 3: Adaptar Predictor** (`predictor.py`)
    - [ ] Modificar `predict()` para que calcule el crecimiento al vuelo (necesitará el revenue del trimestre anterior como input o lo inferirá del texto).

### 4.2 Cerebro Financiero: FinBERT (El Salto de Calidad)
**El Problema**: `TextBlob` (actual) es un diccionario simple. Si dices "loss narrowed" (pérdida se reduce = BUENO), él lee "loss" (MALO).
**La Solución**: Usar `ProsusAI/finbert`, una red neuronal pre-entrenada con millones de noticias financieras.

- [ ] **Paso 1: Integración**
    - [ ] Instalar `transformers` y `torch`.
    - [ ] Crear clase `SentimentAnalyzer` en `src/rag/ml/sentiment.py` que cargue FinBERT.
- [ ] **Paso 2: Sustitución**
    - [ ] En `predictor.py`, reemplazar `TextBlob.sentiment` por `SentimentAnalyzer.predict()`.
    - [ ] Usar las probabilidades (Positive/Negative/Neutral) como features numéricas nuevas.

### 4.3 Explicabilidad (White Box AI)
**El Problema**: El usuario no se fía de una "Caja Negra" que solo dice "POSITIVE".
**La Solución**: Mostrar el "Por qué".

- [ ] **Paso 1: Extracción de Frases Clave**
    - [ ] Cuando FinBERT detecte un sentimiento fuerte, guardar la frase exacta (ej: *"record quebrant earnings"*).
- [ ] **Paso 2: Respuesta Estructurada**
    - [ ] Modificar el JSON de respuesta de la tool para incluir: `{"prediction": "POSITIVE", "reasoning": "Detectado sentimiento muy positivo en la frase 'highest revenue in history'"}`.

### 4.4 Validación "Regreso al Futuro" (La Demo Final)
**El Problema**: ¿Cómo demostramos que funciona sin esperar a 2027?
**La Solución**: Usar 2020 como nuestro "Banco de Pruebas".

- [ ] **Paso 1: Script de Validación**
    - [ ] Crear `scripts/ml/validate_time_travel.py`.
    - [ ] Cargar modelo entrenado (2016-2019).
    - [ ] Leer PDFs reales de 2020.
    - [ ] Comparar Predicción IA vs. Dato Real (que ya conocemos).
    - [ ] Generar reporte de precisión: "El modelo acertó 3 de 4 trimestres de Apple en 2020".

### 4.5 Extras (Si sobra tiempo)
- [ ] **Hybrid Search**: Implementar BM25 para mejorar la búsqueda de nombres propios.
- [ ] **Metadata SQL**: Extraer y guardar métricas clave en BD para consultas rápidas.

---

## Progreso General

| Fase | Tareas | Completadas |
|------|--------|-------------|
| 1. Higiene Crítica | 5 | 5/5 ✅ |
| 2. Calidad de Búsqueda | 5 | 4/5 |
| 3. Robustez del Código | 3 | 3/3 ✅ |
| 4. Funcionalidades Avanzadas | 5 | 0/5 |
| **TOTAL** | **18** | **12/18** |

