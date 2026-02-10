"""
Chat CLI - Interfaz de línea de comandos para el asistente RAG (con Tool Calling).
"""
import sys
from pathlib import Path

# Agregar el proyecto al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.generation.llm_groq import GroqLLM


def print_banner():
    """Muestra el banner de bienvenida."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║        💼 Asistente Financiero RAG - Earnings Calls         ║
║  Dataset: NASDAQ 2019-2020 Earnings Call Transcripts        ║
║  Modelo: Llama 3.3 (vía Groq API) + Tool Calling           ║
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
  - Hola, ¿quién eres? (pregunta general, no debería buscar)
    """
    print(help_text)


def main():
    """Loop principal del chat."""
    print_banner()
    
    # Inicializar solo el LLM (el retriever se inicializa dentro del ToolManager si hace falta)
    print("🔄 Inicializando sistema (LLM + Tools)...")
    
    try:
        llm = GroqLLM()
        print("✅ Sistema listo. ¡Pregúntame!\n")
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
        print("\n💡 Asegúrate de:")
        print("   1. Tener GROQ_API_KEY en el archivo .env")
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
            
            # Procesar pregunta con Tool Calling
            print("\n🤖 Pensando... (tu asistente decidirá si buscar info o responder directo)")
            response = llm.chat_with_tools(query)
            
            # Mostrar respuesta
            print("\n" + "=" * 70)
            print("📝 Respuesta:")
            print("-" * 70)
            print(response)
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
