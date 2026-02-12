# Fase 2: Recuperacion (Retrieval)

## Archivos y Componentes Clave

1.  **`src/rag/retrieval/retriever.py` (El Coordinador)**:
    *   **Rol**: Recibe la pregunta, genera el vector, y delega al Store Unificado.
    *   **Metodo Clave**: `search(query, top_k, filter_company)`.
2.  **`src/rag/storage/unified_store.py` (La BD Unificada)**:
    *   **Rol**: Almacena texto, metadata Y vectores en una sola tabla SQLite. Busqueda vectorial con numpy en RAM.
    *   **Metodos Clave**: `search(query_vector, ...)`, `add_documents(chunks, embeddings)`.
3.  **`src/rag/core/schema.py` (Los Tipos)**:
    *   **Rol**: Define `Chunk` y `RetrievedChunk`.

---

## El Flujo Paso a Paso

Ejemplo: **"Ingresos de Apple en 2020"**.

### Paso 1: Vectorizacion (Texto -> Numeros)
El modelo MiniLM convierte la pregunta en 384 numeros.
*   **Input**: `"ingresos Apple 2020"`
*   **Output (Vector)**: `[-0.015, 0.812, 0.009, ...]` (384 floats)

### Paso 2: Busqueda Vectorial (Numpy en RAM)
Se calculan distancias L2 entre el query vector y TODOS los vectores en cache.
*   **Input**: Vector `[-0.015, ...]`
*   **Output**: Los N indices con menor distancia.
*   *Nota*: Como es busqueda semantica, algunos resultados pueden ser de otras empresas con texto similar.

### Paso 3: Hidratacion (SQLite)
Se recuperan los textos y metadata de los chunks encontrados.
*   **Query SQL**: `SELECT * FROM chunks WHERE chunk_id IN (...)`
*   **Resultado**: Objetos con texto, company, year, quarter.

### Paso 4: Filtrado y Ranking (Python)
El `UnifiedDocumentStore` aplica reglas de negocio:

1.  **Filtro por Empresa**:
    *   Usuario pidio "Apple" (`AAPL`).
    *   Chunk de AAPL -> Pasa.
    *   Chunk de MSFT -> Eliminado.
2.  **Score y Boost**:
    *   Distancia -> Score: `1 / (1 + distancia)`.
    *   Si el usuario dijo "2020" y el chunk es de 2020: `+0.05`.
    *   Si menciono "Q4" y el chunk es Q4: `+0.03`.

---

## Arquitectura de Almacenamiento

Todo esta en un unico archivo `unified_store.db`:

| Columna | Tipo | Descripcion |
|---------|------|-------------|
| `chunk_id` | TEXT PK | Identificador unico |
| `doc_id` | TEXT | Documento padre |
| `text` | TEXT | Fragmento de texto |
| `chunk_index` | INTEGER | Orden dentro del documento |
| `metadata` | TEXT (JSON) | Company, year, quarter, etc. |
| `embedding` | BLOB | Vector de 384 float32 (~1.5KB) |

**Sin FAISS. Sin archivos separados. Una sola fuente de verdad.**
