"""
Test para verificar que el retriever funciona correctamente.
"""
import sys
from pathlib import Path

# Agregar el proyecto al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.retriever import Retriever

if __name__ == "__main__":
    print("=== Test de Retriever (Búsqueda Semántica) ===\n")
    
    # Inicializar retriever
    retriever = Retriever()
    
    # Preguntas de prueba
    queries = [
        "¿Cuáles fueron los ingresos de Apple en 2020?",
        "¿Qué dijo Google sobre publicidad?",
        "Estrategia de crecimiento de Microsoft"
    ]
    
    print("\n" + "=" * 70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: '{query}'")
        print("-" * 70)
        
        # Buscar
        results = retriever.search(query, top_k=3)
        
        if not results:
            print("   ❌ No se encontraron resultados")
            continue
        
        # Mostrar resultados
        for j, chunk in enumerate(results, 1):
            print(f"\n   [{j}] Score: {chunk.score:.4f} | Doc: {chunk.doc_id}")
            print(f"       Texto: {chunk.text[:150]}...")
    
    print("\n" + "=" * 70)
    print("\n✅ Test completado exitosamente")
