from typing import List
from pathlib import Path
import time
from tqdm import tqdm

from src.rag.core.schema import Chunk
from src.rag.storage.vector_store import VectorDB
from src.rag.storage.metadata_store import MetadataDB
from src.rag.pipeline.step3_embedding import EmbeddingModel
from src.rag.core.config import BATCH_SIZE, FAISS_INDEX_PATH, SQLITE_DB_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

def index_data(chunks: List[Chunk], embed_model: EmbeddingModel):
    """
    Paso 4: Indexar chunks en FAISS y guardar metadatos en SQLite.
    Recibe el modelo de embedding ya inicializado (Paso 3).
    """
    print(f"\n🏭 [Step 4] Indexing Data")
    print(f"   - Vector Store: FAISS ({FAISS_INDEX_PATH})")
    print(f"   - Metadata Store: SQLite ({SQLITE_DB_PATH})")
    
    # Inicializar componentes de almacenamiento
    meta_db = MetadataDB(SQLITE_DB_PATH)
    vector_db = VectorDB(FAISS_INDEX_PATH, dimension=EMBEDDING_DIMENSION)
    
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"   Procesando {len(chunks)} chunks en {total_batches} batches...")
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="   Indexing Progress"):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_texts = [c.text for c in batch_chunks]
        batch_ids = [c.chunk_id for c in batch_chunks]
        
        # Generar embeddings
        embeddings = embed_model.embed_texts(batch_texts)
        
        # Guardar en SQLite (metadata + texto)
        meta_db.upsert_chunks(batch_chunks)
        
        # Guardar en FAISS (vectores)
        vector_db.add_vectors(embeddings, batch_ids)
    
    # Persistir índice FAISS
    vector_db.save_index()
    
    elapsed = time.time() - start_time
    print(f"   ✅ Indexing Complete in {elapsed:.2f}s")
    print(f"   - SQLite chunks: {meta_db.count_chunks()}")
    print(f"   - FAISS vectors: {vector_db.count()}")

if __name__ == "__main__":
    # Para pruebas aisladas
    print("Este módulo está diseñado para ser importado por el script principal.")
