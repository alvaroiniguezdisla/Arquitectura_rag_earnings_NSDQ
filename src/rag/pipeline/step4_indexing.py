from typing import List
from pathlib import Path
import time
from tqdm import tqdm

from src.rag.core.schema import Chunk
from src.rag.storage.unified_store import UnifiedDocumentStore
from src.rag.pipeline.step3_embedding import EmbeddingModel
from src.rag.core.config import BATCH_SIZE, SQLITE_DB_PATH, EMBEDDING_DIMENSION

def index_data(chunks: List[Chunk], embed_model: EmbeddingModel):
    """
    Paso 4: Indexar chunks en la BD unificada (SQLite-only).
    Cada chunk se guarda con su vector en la misma fila.
    """
    print(f"\n>> [Step 4] Indexing Data")
    print(f"   - BD Unificada: {SQLITE_DB_PATH}")
    
    # Inicializar Store (SQLite-only)
    store = UnifiedDocumentStore(SQLITE_DB_PATH, dimension=EMBEDDING_DIMENSION)
    
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"   Procesando {len(chunks)} chunks en {total_batches} batches...")
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="   Indexing Progress"):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_texts = [c.text for c in batch_chunks]
        
        # Generar embeddings
        embeddings = embed_model.embed_texts(batch_texts)
        
        # Guardar en BD unificada (texto + metadata + vector en la misma fila)
        store.add_documents(batch_chunks, embeddings)
    
    # Recargar cache
    store.save()
    
    elapsed = time.time() - start_time
    print(f"   OK Indexing Complete in {elapsed:.2f}s")
    print(f"   - Total chunks: {store.count()}")

if __name__ == "__main__":
    print("Este modulo esta disenado para ser importado por el script principal.")
