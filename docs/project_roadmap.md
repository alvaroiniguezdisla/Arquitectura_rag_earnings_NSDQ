# Roadmap: Proyecto RAG Financiero Local

Este documento define el plan **paso a paso** para construir nuestro sistema, priorizando simplicidad y buenas prácticas, como pediste.

## 🎯 Objetivo
Crear un asistente financiero capaz de leer transcripts de earnings (2019-2020) y responder preguntas citando fuentes, todo ejecutándose en tu PC.

## 🏗️ Arquitectura (Decisiones Clave)
Basado en lo que hemos hablado y las mejores prácticas para un MVP educativo:

*   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`. (Estándar, rápido, funciona en CPU).
*   **Vector DB**: `FAISS` + `SQLite`. 
    *   *Por qué:* FAISS es el motor de búsqueda vectorial por excelencia (Facebook Research). SQLite nos da la robustez para guardar los textos y metadatos sin depender de servidores externos complejos. Es la forma "artesanal" y robusta de hacerlo.
*   **LLM**: `TinyLlama` (vía `llama-cpp-python`).
*   **Chunking**: Ventana fija con solape (Simple y efectivo para empezar).

---

## 📅 Fases del Proyecto

### ✅ Fase 1: Datos (Completada)
*   [x] Descarga de Kaggle automatizada.
*   [x] Limpieza y normalización de textos (2019-2020).
*   [x] Resultado: `data/processed/corpus.jsonl`.

### 🔄 Fase 2: Cimientos del Código (Donde estamos)
Antes de procesar nada, necesitamos organizar la "casa".
*   [ ] **Configuración Central (`config.py`)**: Para no tener rutas y números mágicos desperdigados.
*   [ ] **Definiciones (`schema.py`)**: Para que Python sepa qué es un "Documento" y un "Chunk".

### 🔜 Fase 3: Procesamiento (La "fábrica" de vectores)
*   [ ] **Ingesta**: Leer nuestro `corpus.jsonl`.
*   [ ] **Chunking**: Cortar los textos en trozos digeribles para la IA.
*   [ ] **Embeddings**: Convertir texto a números.
*   [ ] **Indexado**: Guardar en FAISS y SQLite.

### 🔜 Fase 4: El Cerebro (RAG)
*   [ ] **Búsqueda**: Preguntar a la base de datos y recuperar contextos.
*   [ ] **Generación**: Pasar esos contextos a TinyLlama para que responda.

### 🔜 Fase 5: Interfaz
*   [ ] **Chat CLI**: Hablar con tu asistente desde la terminal.

---

## 📝 Estado Actual
Hemos vuelto al final de la **Fase 1**. Tenemos los datos listos y el proyecto limpio.
El siguiente paso lógico es crear la estructura de carpetas `src/rag/` y los archivos de configuración base, pero **uno a uno y explicándolos**.
