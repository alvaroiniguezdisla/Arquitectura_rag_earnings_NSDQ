"""
Script para indexar todo el corpus (corpus.jsonl) en las bases de datos.
Este es el "botón mágico" que crea la base de datos completa.
"""
import sys
from pathlib import Path
import time

# Agregar el proyecto al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.core.config import (
    CORPUS_FILE, CHUNK_SIZE, CHUNK_OVERLAP, 
    SQLITE_DB_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION, BATCH_SIZE
)
from src.rag.pipeline.step1_loader import load_processed_corpus
from src.rag.pipeline.step2_chunking import chunk_documents
from src.rag.pipeline.step3_embedding import EmbeddingModel


def build_index():
    """
    Pipeline completo de indexación:
    1. Cargar corpus
    2. Ch unking
    3. Embeddings
    4. Guardar en SQLite + FAISS
    """
    start_time = time.time()
    
    print("=" * 60)
    print("🏭 CONSTRUCCIÓN DEL ÍNDICE RAG")
    print("=" * 60)
    
    # Paso 1: Cargar datos
    print(f"\n📂 [1/5] Cargando corpus desde: {CORPUS_FILE}")
    documents = load_processed_corpus(CORPUS_FILE)
    
    if not documents:
        print("❌ No se encontraron documentos. Verifica el corpus.")
        return
    
    print(f"   ✅ Cargados {len(documents)} documentos")
    
    # Paso 2: Chunking
    print(f"\n🔪 [2/5] Cortando documentos (tamaño={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"   ✅ Generados {len(chunks)} chunks")
    print(f"   📊 Promedio: {len(chunks) / len(documents):.1f} chunks por documento")
    

    # Paso 3: Inicializar Modelo de Embedding
    print(f"\n🧠 [3/4] Cargando Modelo de Embedding")
    print(f"   Modelo: {EMBEDDING_MODEL_NAME}")
    embed_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
    
    # Paso 4: Indexado (Storage)
    from src.rag.pipeline.step4_indexing import index_data
    index_data(chunks, embed_model)
    
    # Resumen final
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ INDEXACIÓN COMPLETADA")
    print("=" * 60)
    print(f"⏱️  Tiempo total: {elapsed:.2f} segundos")
    print("\n🎉 ¡Listo para hacer búsquedas!")


if __name__ == "__main__":
    build_index()
