# Arquitectura RAG - Fases del Proceso

El flujo completo de la arquitectura RAG. Desde que el usuario formula una pregunta hasta que recibe una respuesta.

El proceso se divide en tres fases:

## [Fase 1: Intencion y Decision (El Cerebro)](fases/fase1_intencion.md)
**"Que quiere el usuario? Necesito buscar informacion?"**
- **Actor Principal**: LLM (Llama 3 via Groq) + Tool Manager.
- **Objetivo**: Entender la pregunta y decidir si usar herramientas o responder directamente.
- **Mecanismo**: Tool Calling (Function Calling).

## [Fase 2: Recuperacion (El Buscador)](fases/fase2_recuperacion.md)
**"Buscando la aguja en el pajar."**
- **Actor Principal**: Retriever + UnifiedDocumentStore.
- **Objetivo**: Buscar en la BD unificada (SQLite) los fragmentos de texto mas relevantes.
- **Componentes**: Embeddings (MiniLM), SQLite con vectores como BLOBs, cache numpy en RAM.

## [Fase 3: Generacion (El Orador)](fases/fase3_generacion.md)
**"Redactando la respuesta perfecta."**
- **Actor Principal**: LLM (con contexto).
- **Objetivo**: Tomar la pregunta original y los fragmentos recuperados para generar una respuesta precisa.
- **Salida**: La respuesta final que ve el usuario.
