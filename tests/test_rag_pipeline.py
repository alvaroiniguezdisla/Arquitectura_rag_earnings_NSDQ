"""
Test rápido del sistema RAG end-to-end (sin interacción).
"""
import sys
from pathlib import Path

# Agregar el proyecto al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.retrieval.retriever import Retriever
from src.rag.generation.llm_groq import GroqLLM


def test_rag_pipeline():
    """Test rápido del pipeline completo."""
    print("=" * 70)
    print("🧪 TEST RAG PIPELINE (End-to-End)")
    print("=" * 70)
    
    # 1. Inicializar componentes
    print("\n[1/3] Inicializando componentes...")
    try:
        retriever = Retriever()
        llm = GroqLLM()
        print("   ✅ Retriever y LLM inicializados")
    except Exception as e:
        print(f"   ❌ Error en inicialización: {e}")
        return
    
    # 2. Hacer una búsqueda
    print("\n[2/3] Testeando búsqueda...")
    query = "¿Cuáles fueron los ingresos de Apple en 2020?"
    print(f"   Query: '{query}'")
    
    try:
        chunks = retriever.search(query, top_k=5)
        print(f"   ✅ Encontrados {len(chunks)} chunks relevantes")
        
        if chunks:
            print(f"\n   Top chunk (score: {chunks[0].score:.4f}):")
            print(f"   Doc: {chunks[0].doc_id}")
            print(f"   Texto: {chunks[0].text[:100]}...")
    except Exception as e:
        print(f"   ❌ Error en búsqueda: {e}")
        return
    
    # 3. Generar respuesta con LLM
    print("\n[3/3] Generando respuesta con LLM...")
    try:
        response = llm.generate_response(query, chunks)
        print(f"   ✅ Respuesta generada ({len(response)} caracteres)")
        print("\n" + "=" * 70)
        print("📝 RESPUESTA:")
        print("=" * 70)
        print(response)
        print("=" * 70)
    except Exception as e:
        print(f"   ❌ Error en generación: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ Test completado exitosamente!")


if __name__ == "__main__":
    test_rag_pipeline()
