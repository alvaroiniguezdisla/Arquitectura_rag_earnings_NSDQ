"""
Test para verificar que embeddings.py funciona correctamente.
"""
import sys
from pathlib import Path

# Agregar el proyecto al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.pipeline.step3_embedding import EmbeddingModel

if __name__ == "__main__":
    print("=== Test de Embeddings ===\n")
    
    # 1. Cargar modelo
    print("Cargando modelo de embeddings...")
    embed_model = EmbeddingModel()
    
    # 2. Probar con un texto
    texto_prueba = "Apple reported record revenue in Q1 2020."
    print(f"\nTexto de prueba: '{texto_prueba}'")
    
    vector = embed_model.embed_text(texto_prueba)
    
    print(f"\nResultado:")
    print(f"   - Dimension del vector: {vector.shape}")
    print(f"   - Primeros 10 valores: {vector[:10]}")
    print(f"   - Tipo: {type(vector)}")
    
    # 3. Probar con multiples textos (batch)
    textos_batch = [
        "Microsoft lanza nuevo producto de IA",
        "Los ingresos de Google crecieron un 15%",
        "Tesla anuncia expansion en Europa"
    ]
    
    print(f"\nProbando batch de {len(textos_batch)} textos...")
    vectores = embed_model.embed_texts(textos_batch)
    
    print(f"\nResultado batch:")
    print(f"   - Shape: {vectores.shape}")
    print(f"   - Esperado: ({len(textos_batch)}, {embed_model.dimension})")
    
    # 4. Verificar que textos similares tienen vectores similares
    print(f"\nVerificando similitud...")
    texto1 = "Apple reported financial results"
    texto2 = "Apple announced quarterly earnings"
    texto3 = "The weather today is sunny"
    
    v1 = embed_model.embed_text(texto1)
    v2 = embed_model.embed_text(texto2)
    v3 = embed_model.embed_text(texto3)
    
    # Calcular similitud coseno
    from numpy.linalg import norm
    sim_12 = (v1 @ v2) / (norm(v1) * norm(v2))
    sim_13 = (v1 @ v3) / (norm(v1) * norm(v3))
    
    print(f"   - Similitud (texto1 vs texto2): {sim_12:.4f} (esperado: alta)")
    print(f"   - Similitud (texto1 vs texto3): {sim_13:.4f} (esperado: baja)")
    
    if sim_12 > sim_13:
        print(f"\n   OK Los textos similares tienen mayor similitud")
    else:
        print(f"\n   WARN Algo raro: la similitud no es la esperada")
    
    print(f"\nOK Test completado exitosamente")
