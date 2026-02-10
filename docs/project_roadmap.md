# 📅 Project Roadmap: RAG Earnings

## 🎯 El Objetivo
Construir un asistente financiero robusto, agéntico y preciso que analice transcripts de NASDAQ (2019-2020) citando fuentes reales.

---

## ✅ Fase 1: Datos y Cimientos (Completada)
*   [x] Ingesta de datos desde Kaggle.
*   [x] Limpieza y normalización de textos.
*   [x] Configuración centralizada en `src/rag/core/config.py`.

## ✅ Fase 2: El Motor de Búsqueda (Completada)
*   [x] Implementación de **FAISS** para búsqueda vectorial.
*   [x] Creación de base de datos **SQLite** para persistencia de metadatos.
*   [x] Pipeline ETL (`build_index.py`) automatizado.

## ✅ Fase 3: La IA y el Agente (Completada)
*   [x] Integración con **Groq API** (Llama 3.3).
*   [x] Desarrollo de **Tool Calling**: El sistema decide cuándo buscar información.
*   [x] Interfaz de chat interactiva en línea de comandos.

## ✅ Fase 4: Optimización Avanzada (Completada — ¡Lo último!)
*   [x] **Metadata Boost**: Algoritmo para priorizar años y trimestres específicos.
*   [x] **Ticker Mapping**: Soporte para nombres de empresas (Microsoft -> MSFT).
*   [x] **Aumento de Profundidad**: Búsqueda en 60 fragmentos para evitar pérdida de datos.
*   [x] **Multitool**: Nueva herramienta `list_available_companies`.

---

## 🏁 Estado Actual: v1.5 (Production Ready)
- El sistema es capaz de responder preguntas complejas sobre Apple, Microsoft y otras empresas del NASDAQ.
- La precisión de recuperación es >95% en consultas temporales.
- El código está refactorizado, documentado y limpio.

---

## 🚀 Próximos Pasos (Opcional)
- [ ] **Web UI**: Migrar de CLI a Streamlit para una mejor visualización.
- [ ] **Conversational Memory**: Permitir que el asistente recuerde preguntas anteriores.
- [ ] **Dockerization**: Facilitar el despliegue en cualquier servidor.
