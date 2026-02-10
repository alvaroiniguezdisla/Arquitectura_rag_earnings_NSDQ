"""
Script para indexar todo el corpus (corpus.jsonl) en las bases de datos.
Este es el "botón mágico" que crea la base de datos completa.
"""
import sys
from pathlib import Path
import time

# Agregar el proyecto al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.config import (
    CORPUS_FILE, CHUNK_SIZE, CHUNK_OVERLAP, 
    SQLITE_DB_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION, BATCH_SIZE
)
from src.rag.ingest import load_processed_corpus
from src.rag.chunking import chunk_documents
from src.rag.embeddings import EmbeddingModel
from src.rag.vdb_sqlite import MetadataDB
from src.rag.vdb_faiss import VectorDB


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
    
    # Paso 3: Inicializar bases de datos
    print(f"\n🗄️  [3/5] Inicializando bases de datos")
    print(f"   - SQLite: {SQLITE_DB_PATH}")
    print(f"   - FAISS: {FAISS_INDEX_PATH}")
    
    meta_db = MetadataDB(SQLITE_DB_PATH)
    vector_db = VectorDB(FAISS_INDEX_PATH, dimension=EMBEDDING_DIMENSION)
    
    # Paso 4: Embeddings e Indexado
    print(f"\n🔢 [4/5] Generando embeddings e indexando")
    print(f"   Modelo: {EMBEDDING_MODEL_NAME}")
    
    embed_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
    
    # Procesar en batches para no saturar la RAM
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"   Procesando {len(chunks)} chunks en {total_batches} batches...")
    
    from tqdm import tqdm
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="   Indexing"):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_texts = [c.text for c in batch_chunks]
        batch_ids = [c.chunk_id for c in batch_chunks]
        
        # Generar embeddings
        embeddings = embed_model.embed_texts(batch_texts)
        
        # Guardar en SQLite (metadata + texto)
        meta_db.upsert_chunks(batch_chunks)
        
        # Guardar en FAISS (vectores)
        vector_db.add_vectors(embeddings, batch_ids)
    
    # Paso 5: Persistir índice FAISS
    print(f"\n💾 [5/5] Guardando índice FAISS en disco")
    vector_db.save_index()
    
    # Resumen final
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ INDEXACIÓN COMPLETADA")
    print("=" * 60)
    print(f"⏱️  Tiempo total: {elapsed:.2f} segundos")
    print(f"📄 Documentos indexados: {len(documents)}")
    print(f"🔪 Chunks creados: {len(chunks)}")
    print(f"🗄️  SQLite chunks: {meta_db.count_chunks()}")
    print(f"🔢 FAISS vectores: {vector_db.count()}")
    print(f"\n📂 Archivos creados:")
    print(f"   - {SQLITE_DB_PATH}")
    print(f"   - {FAISS_INDEX_PATH}")
    print("\n🎉 ¡Listo para hacer búsquedas!")


if __name__ == "__main__":
    build_index()
