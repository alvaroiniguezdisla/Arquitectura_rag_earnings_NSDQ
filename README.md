# 💼 RAG Earnings NSDQ - Asistente Financiero Inteligente

![Banner](https://img.shields.io/badge/RAG-Financial_Analysis-blue?style=for-the-badge) ![Groq](https://img.shields.io/badge/LLM-Llama_3.3_70B-orange?style=for-the-badge) ![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-green?style=for-the-badge)

Sistema de **Retrieval-Augmented Generation (RAG)** agéntico diseñado para analizar transcripts de conferencias de resultados financieros (Earnings Calls) de empresas del NASDAQ (2019-2020). 

---

## 🌟 Características Principales

*   **🧠 Agente Autónomo**: El sistema decide inteligentemente cuándo consultar la base de datos financiera y cuándo responder directamente.
*   **🔍 Recuperación Optimizada**:
    *   **Metadata Boost**: Prioriza automáticamente los resultados del año y trimestre mencionados en tu pregunta.
    *   **Ticker Mapping**: Traduce nombres de empresas (ej: "Microsoft") a sus símbolos (MSFT) para búsquedas ultra-precisas.
    *   **Búsqueda Semántica**: Motor FAISS + All-MiniLM-L6 para entender el contexto, no solo palabras clave.
*   **📊 Transparencia Total**: El chat muestra qué herramientas está usando y cuánta información está recuperando en "tiempo real".

---

## 🛠️ Instalación Rápida

### 1. Preparar el Entorno
```bash
# Crear entorno virtual
python -m venv .venv
# Activar (Windows)
.venv\Scripts\activate
# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración (.env)
Configura tu API Key en la raíz del proyecto:
```env
GROQ_API_KEY=tu_gsk_clave_aqui
```

---

## 🏗️ Cómo Empezar

### Paso 1: Indexación (Construir el Cerebro)
Procesa los transcripts originales y crea las bases de datos vectoriales y de metadatos.
```bash
python scripts/build_index.py
```

### Paso 2: Ejecutar el Chat
Lanza la interfaz interactiva para hablar con el asistente.
```bash
python scripts/chat_cli.py
```

---

## 💡 Ejemplos de Preguntas

*   **Básica**: "¿Cuáles fueron los ingresos de Apple en el Q4 de 2019?"
*   **Estrategia**: "¿Qué dijo el CEO de NVIDIA sobre la demanda de chips en 2020?"
*   **Meta**: "¿De qué empresas tienes información disponible?" (Usa la herramienta `list_available_companies`).

---

## 📂 Estructura del Proyecto

```text
src/rag/
├── generation/    # 🧠 Cerebro: Llama 3.3, Tools y prompts
├── retrieval/     # 🔍 Lupa: Lógica de búsqueda avanzada y boost
├── storage/       # 💾 Memoria: FAISS (vectores) y SQLite (textos)
├── pipeline/      # 🏭 Fábrica: Loader, Chunking y Embedding
└── core/          # ⚙️ Motor: Configuración y esquemas
```

---

## 📄 Documentación Extendida
- [🏗️ Arquitectura Detallada](docs/architecture.md)
- [📅 Roadmap del Proyecto](docs/project_roadmap.md)
