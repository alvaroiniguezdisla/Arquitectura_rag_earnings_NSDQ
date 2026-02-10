"""
Test para verificar que las bases de datos (SQLite + FAISS) funcionan correctamente.
"""
import sys
from pathlib import Path

# Agregar el proyecto al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.schema import Chunk
from src.rag.vdb_sqlite import MetadataDB
from src.rag.vdb_faiss import VectorDB
from src.rag.embeddings import EmbeddingModel

if __name__ == "__main__":
    print("=== Test de Bases de Datos (SQLite + FAISS) ===\n")
    
    # 1. Crear datos de prueba
    print("📝 Creando datos de prueba...")
    test_chunks = [
        Chunk(
            chunk_id="test_001",
            doc_id="apple_2020_q1",
            text="Apple reportó ingresos récord en el primer trimestre de 2020.",
            chunk_index=0,
            metadata={"year": 2020, "company": "Apple"}
        ),
        Chunk(
            chunk_id="test_002",
            doc_id="apple_2020_q1",
            text="Las ventas de iPhone aumentaron significativamente.",
            chunk_index=1,
            metadata={"year": 2020, "company": "Apple"}
        ),
        Chunk(
            chunk_id="test_003",
            doc_id="google_2020_q2",
            text="Google anunció crecimiento en publicidad digital.",
            chunk_index=0,
            metadata={"year": 2020, "company": "Google"}
        )
    ]
    
    # 2. Test SQLite
    print("\n🗄️ Testeando SQLite...")
    db_path = Path("data/indexes/test_metadata.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # Limpiar test anterior
    
    metadata_db = MetadataDB(db_path)
    metadata_db.upsert_chunks(test_chunks)
    
    # Verificar inserción
    count = metadata_db.count_chunks()
    print(f"   ✅ Chunks guardados: {count}")
    
    # Recuperar un chunk
    retrieved = metadata_db.get_chunk("test_001")
    print(f"   ✅ Chunk recuperado: {retrieved.text[:50]}...")
    
    # 3. Test FAISS
    print("\n🔍 Testeando FAISS...")
    index_path = Path("data/indexes/test_vectors.index")
    if index_path.exists():
        index_path.unlink()  # Limpiar test anterior
    id_map_path = index_path.with_suffix('.id_map.pkl')
    if id_map_path.exists():
        id_map_path.unlink()
    
    vector_db = VectorDB(index_path, dimension=384)
    
    # Generar embeddings para los chunks
    print("   🔄 Generando embeddings...")
    embed_model = EmbeddingModel()
    texts = [c.text for c in test_chunks]
    vectors = embed_model.embed_texts(texts)
    
    # Añadir al índice
    vector_db.add_vectors(vectors, [c.chunk_id for c in test_chunks])
    print(f"   ✅ Vectores indexados: {vector_db.count()}")
    
    # 4. Test de búsqueda
    print("\n🔎 Testeando búsqueda...")
    query = "ingresos de Apple"
    query_vector = embed_model.embed_text(query)
    
    chunk_ids, distances = vector_db.search(query_vector, top_k=2)
    
    print(f"   Query: '{query}'")
    print(f"   Resultados:")
    for i, (cid, dist) in enumerate(zip(chunk_ids, distances)):
        chunk = metadata_db.get_chunk(cid)
        print(f"      {i+1}. [{cid}] Distancia: {dist:.4f}")
        print(f"         Texto: {chunk.text[:60]}...")
    
    # 5. Test de persistencia
    print("\n💾 Testeando persistencia...")
    vector_db.save_index()
    
    # Recargar
    vector_db2 = VectorDB(index_path, dimension=384)
    print(f"   ✅ Índice recargado: {vector_db2.count()} vectores")
    
    # Limpiar archivos de test
    print("\n🧹 Limpiando archivos de test...")
    
    # Cerrar conexiones explícitamente antes de borrar
    del metadata_db  # Libera la conexión SQLite
    del vector_db2   # Libera referencias FAISS
    
    # Ahora sí, borrar archivos
    import time
    time.sleep(0.5)  # Dar tiempo a Windows para liberar los archivos
    
    try:
        db_path.unlink()
        index_path.unlink()
        id_map_path.unlink()
        print("   ✅ Archivos de test eliminados")
    except PermissionError:
        print("   ⚠️ No se pudieron borrar archivos (Windows los mantiene abiertos)")
        print("   ℹ️  Puedes borrarlos manualmente o ignorarlos")
    
    print("\n✅ Test completado exitosamente")
