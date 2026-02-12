"""
Test para verificar que chunking.py funciona correctamente.
"""
import sys
from pathlib import Path

# Agregar el proyecto al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.pipeline.step1_loader import load_processed_corpus
from src.rag.pipeline.step2_chunking import chunk_documents

if __name__ == "__main__":
    print("=== Test de Chunking ===\n")
    
    # 1. Cargar documentos
    print("Cargando documentos...")
    documents = load_processed_corpus()
    
    if not documents:
        print("ERROR: No se cargaron documentos")
        sys.exit(1)
    
    print(f"OK Cargados {len(documents)} documentos\n")
    
    # 2. Hacer chunking
    print("Cortando documentos en chunks...")
    chunks = chunk_documents(documents)
    
    print(f"OK Generados {len(chunks)} chunks\n")
    
    # 3. Mostrar estadisticas
    print(f"Estadisticas:")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Promedio chunks por documento: {len(chunks) / len(documents):.1f}")
    
    # 4. Ejemplo de chunks
    print(f"\nEjemplo del primer documento:")
    first_doc_chunks = [c for c in chunks if c.doc_id == documents[0].doc_id]
    print(f"   - Documento ID: {documents[0].doc_id}")
    print(f"   - Numero de chunks: {len(first_doc_chunks)}")
    print(f"   - Chunk 0 (primeros 200 chars):")
    print(f"     {first_doc_chunks[0].text[:200]}...")
    
    if len(first_doc_chunks) > 1:
        print(f"\n   - Chunk 1 (primeros 200 chars):")
        print(f"     {first_doc_chunks[1].text[:200]}...")
        
        # Verificar solape
        overlap_text = first_doc_chunks[0].text[-100:]
        if overlap_text in first_doc_chunks[1].text[:150]:
            print(f"\n   OK Solape detectado correctamente entre chunks")
        else:
            print(f"\n   WARN No se detecto solape entre chunks")
    
    print(f"\nOK Test completado exitosamente")
