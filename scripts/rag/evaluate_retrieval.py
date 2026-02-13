
import json
import sys
import os
import time
from pathlib import Path

# Añadir src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.generation.llm_groq import GroqLLM
from src.rag.generation.tools import tool_manager

def evaluate_end_to_end(dataset_path: str):
    print(f"\n🚀 Iniciando Evaluación End-to-End: {dataset_path}")
    print("="*60)
    
    # 1. Cargar Dataset
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el dataset en {dataset_path}")
        return

    # 2. Inicializar el Cerebro (LLM)
    llm = GroqLLM()
    
    total_questions = len(dataset)
    retrieval_hits = 0      # ¿Encontramos el dato?
    generation_hits = 0     # ¿El LLM respondió bien?
    mrr_sum = 0.0           # Suma para Mean Reciprocal Rank
    
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        key_terms = item["key_terms"]
        expected_company = item["expected_company"]
        
        print(f"\n📝 Pregunta [{i}/{total_questions}]: '{question}'")
        
        # --- A. GENERACIÓN (El Chat) ---
        start_time = time.time()
        try:
            # SLEEP para evitar Rate Limit (Groq es estricto en tier free)
            print("   ⏳ Esperando 20s para respetar Rate Limit de Groq...")
            time.sleep(20) 
            response = llm.chat_with_tools(question)
        except Exception as e:
            print(f"   💥 Error en LLM: {e}")
            # Si falla el LLM, saltamos a la siguiente pregunta pero NO contamos en retrieval
            # (porque no podemos evaluar la respuesta generada)
            continue
        duration = time.time() - start_time
        
        print(f"   🤖 Respuesta ({duration:.1f}s):\n   '{response}'")
        
        # --- B. EVALUACIÓN DE RESPUESTA (Generation Score) ---
        # Verificamos si la respuesta final contiene los términos clave
        gen_success = True
        missing_gen = []
        for term in key_terms:
            if term.lower() not in response.lower():
                gen_success = False
                missing_gen.append(term)
        
        if gen_success:
            print(f"   ✅ GENERACIÓN: Correcta (Contiene {key_terms})")
            generation_hits += 1
        else:
            print(f"   ❌ GENERACIÓN: Incorrecta (Falta: {missing_gen})")

        # --- C. EVALUACIÓN DE RETRIEVAL (MRR & Recall) ---
        # Hack: Validamos "a posteriori" buscando de nuevo con la tool
        # para ver en qué posición estaba el dato.
        # (En un sistema real, instrumentaríamos el ToolManager para que nos devolviera esto)
        
        raw_results_json = tool_manager.search_earnings_calls(
            query=question, 
            company_id=expected_company, 
            num_results=15
        )
        chunks = json.loads(raw_results_json)
        
        # ESTRATEGIA MEJORADA: 
        # En lugar de buscar en UN solo chunk, buscamos en el contexto acumulado de los Top N.
        # El LLM lee todo el contexto, así que si los datos están dispersos en el Top 5, cuenta como éxito.
        
        found_rank = -1
        
        # Vamos acumulando texto y chequeando
        accumulated_text = ""
        for idx, chunk in enumerate(chunks, 1):
            accumulated_text += " " + chunk['content'].lower()
            
            # Verificamos si CON ESTE NUEVO CHUNK ya tenemos todos los términos
            missing_terms = [t for t in key_terms if t.lower() not in accumulated_text]
            
            if not missing_terms:
                found_rank = idx
                # El score del chunk que "completó" el hallazgo (o el más relevante si fue el 1)
                found_score = chunk.get('score', 0.0)
                break
        
        if found_rank != -1:
            print(f"   🔍 RETRIEVAL: Datos encontrados en el Rank #{found_rank} (Score: {found_score})")
            retrieval_hits += 1
            mrr_sum += (1.0 / found_rank)
        else:
            print(f"   🚫 RETRIEVAL: Datos NO encontrados en Top-15")
        
        print("-" * 60)

    # --- MÉTRICAS FINALES ---
    avg_mrr = mrr_sum / total_questions
    recall = (retrieval_hits / total_questions) * 100
    gen_acc = (generation_hits / total_questions) * 100
    
    print("\n📊 REPORTE FINAL")
    print(f"1. Retrieval Recall @ 15:  {recall:.1f}% ({retrieval_hits}/{total_questions})")
    print(f"2. Mean Reciprocal Rank:   {avg_mrr:.3f} (Cuanto más cerca de 1.0, mejor)")
    print(f"3. Generation Accuracy:    {gen_acc:.1f}% ({generation_hits}/{total_questions})")
    
    if gen_acc >= 80:
        print("\n🏆 CONCLUSIÓN: El sistema está listo para producción.")
    else:
        print("\n⚠️ CONCLUSIÓN: Se requieren ajustes en el prompt o retrieval.")

if __name__ == "__main__":
    dataset_path = "data/gold_dataset.json"
    evaluate_end_to_end(dataset_path)
