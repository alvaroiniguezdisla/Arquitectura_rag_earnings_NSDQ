"""
Test simple para verificar que ingest.py funciona correctamente.
"""
import sys
from pathlib import Path

# Agregar el proyecto al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.ingest import load_processed_corpus

if __name__ == "__main__":
    print("=== Test de Ingesta ===\n")
    
    # Cargar documentos
    documents = load_processed_corpus()
    
    if not documents:
        print("❌ No se cargaron documentos")
        sys.exit(1)
    
    print(f"\n✅ Total documentos: {len(documents)}")
    print(f"\n📄 Ejemplo del primer documento:")
    print(f"   - ID: {documents[0].doc_id}")
    print(f"   - Metadata: {documents[0].metadata}")
    print(f"   - Texto (primeros 200 caracteres):")
    print(f"     {documents[0].text[:200]}...")
    
    print(f"\n📊 Resumen:")
    print(f"   - Documentos cargados: {len(documents)}")
    print(f"   - Tamaño promedio: {sum(len(d.text) for d in documents) // len(documents)} caracteres")
