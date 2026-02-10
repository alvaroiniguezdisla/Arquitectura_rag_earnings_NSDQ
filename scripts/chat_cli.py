"""
Chat CLI - Interfaz de línea de comandos para el asistente RAG.
"""
import sys
from pathlib import Path

# Agregar el proyecto al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.retriever import Retriever
from src.rag.llm_groq import GroqLLM


def print_banner():
    """Muestra el banner de bienvenida."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║        💼 Asistente Financiero RAG - Earnings Calls         ║
║                                                              ║
║  Dataset: NASDAQ 2019-2020 Earnings Call Transcripts        ║
║  Modelo: Llama 3.1 (vía Groq API)                          ║
╚══════════════════════════════════════════════════════════════╝

Comandos:
  - Escribe tu pregunta y presiona Enter
  - 'exit' o 'quit' para salir
  - 'help' para ver comandos

"""
    print(banner)


def print_help():
    """Muestra ayuda."""
    help_text = """
📚 Ejemplos de preguntas:
  - ¿Cuáles fueron los ingresos de Apple en 2020?
  - ¿Qué dijo el CEO de Google sobre publicidad?
  - Resume la estrategia de Microsoft
  - ¿Qué empresas mencionaron COVID-19?
    """
    print(help_text)


def format_sources(chunks):
    """Formatea las fuentes recuperadas."""
    sources_text = "\n📚 Fuentes consultadas:\n"
    for i, chunk in enumerate(chunks, 1):
        company = chunk.metadata.get("company", "Unknown")
        year = chunk.metadata.get("year", "Unknown")
        score = chunk.score
        sources_text += f"   [{i}] {company} - {year} (relevancia: {score:.2f})\n"
    return sources_text


def main():
    """Loop principal del chat."""
    print_banner()
    
    # Inicializar sistema RAG
    print("🔄 Inicializando sistema RAG...")
    
    try:
        retriever = Retriever()
        llm = GroqLLM()
        print("✅ Sistema listo\n")
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
        print("\n💡 Asegúrate de:")
        print("   1. Tener GROQ_API_KEY en el archivo .env")
        print("   2. Haber ejecutado scripts/build_index.py")
        return
    
    # Loop principal
    while True:
        try:
            # Leer entrada del usuario
            query = input("\n💬 Tu pregunta: ").strip()
            
            if not query:
                continue
            
            # Comandos especiales
            if query.lower() in ['exit', 'quit', 'salir']:
                print("\n👋 ¡Hasta luego!")
                break
            
            if query.lower() == 'help':
                print_help()
                continue
            
            # Procesar pregunta
            print("\n🔍 Buscando información relevante...")
            retrieved_chunks = retriever.search(query, top_k=5)
            
            if not retrieved_chunks:
                print("❌ No encontré información relevante en la base de datos.")
                continue
            
            print("🤖 Generando respuesta...")
            response = llm.generate_response(query, retrieved_chunks)
            
            # Mostrar respuesta
            print("\n" + "=" * 70)
            print("📝 Respuesta:")
            print("-" * 70)
            print(response)
            print("=" * 70)
            
            # Mostrar fuentes
            print(format_sources(retrieved_chunks))
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
