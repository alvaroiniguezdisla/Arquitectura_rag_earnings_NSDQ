"""
Chat CLI - Interfaz de línea de comandos para el asistente RAG (con Tool Calling).
"""
import sys
from pathlib import Path

# Agregar el proyecto al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.generation.llm_groq import GroqLLM
from src.rag.generation.tools import ToolManager
from src.rag.core.memory import MemoryManager
from src.rag.core.logger import get_logger

logger = get_logger(__name__)


def print_banner():
    """Muestra el banner de bienvenida."""
    banner = """
==============================================================
        Asistente Financiero RAG - Earnings Calls         
  Dataset: NASDAQ 2019-2020 Earnings Call Transcripts        
  Modelo: Llama 3.3 (via Groq API) + Tool Calling           
==============================================================

Comandos:
  - Escribe tu pregunta y presiona Enter
  - 'exit' o 'quit' para salir
  - 'help' para ver comandos

"""
    print(banner)


def print_help():
    """Muestra ayuda."""
    help_text = """
 Ejemplos de preguntas:
  - Cuales fueron los ingresos de Apple en 2020?
  - Que dijo el CEO de Google sobre publicidad?
  - Resume la estrategia de Microsoft
  - Hola, quien eres? (pregunta general, no deberia buscar)
    """
    print(help_text)


def main():
    """Loop principal del chat."""
    print_banner()
    
    # Inicializar Componentes
    logger.info("Inicializando sistema (LLM + Tools + Memoria)...")
    
    try:
        # 1. Creamos la memoria
        memory = MemoryManager(limit=10) # Recuerda los ultimos 10 mensajes
        
        # 2. Creamos el ToolManager (Retriever + Predictor)
        tool_mgr = ToolManager()
        
        # 3. Creamos el LLM y le pasamos el ToolManager
        llm = GroqLLM(tool_manager=tool_mgr)
        
        logger.info("Sistema listo")
        print("\nOK Sistema listo. Preguntame!\n")
    except Exception as e:
        logger.error(f"Error al inicializar: {e}")
        print(f"XX Error al inicializar: {e}")
        print("\n** Asegurate de:")
        print("   1. Tener GROQ_API_KEY en el archivo .env")
        return
    
    # Loop principal
    while True:
        try:
            # Leer entrada del usuario
            query = input("\n>> Tu pregunta: ").strip()
            
            if not query:
                continue
            
            # Comandos especiales
            if query.lower() in ['exit', 'quit', 'salir']:
                print("\n Hasta luego!")
                break
            
            if query.lower() == 'help':
                print_help()
                continue
            
            # Procesar pregunta con Tool Calling + MEMORIA
            print("\n.. Pensando... (Recuperando contexto y decidiendo...)")
            
            # Le pasamos al LLM la historia previa para que tenga contexto
            # (El LLM recibe: System Prompt + Historia + Pregunta Actual)
            response = llm.chat_with_tools(user_query=query, history=memory.get_messages())
            
            # Guardamos lo que ha pasado en la memoria para la próxima vez
            memory.add_user_message(query)
            memory.add_assistant_message(response)
            
            # Mostrar respuesta
            print("\n" + "=" * 70)
            print(" Respuesta:")
            print("-" * 70)
            print(response)
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n\n Hasta luego!")
            break
        except Exception as e:
            logger.error(f"Error procesando pregunta: {e}")
            print(f"\nXX Error: {e}")


if __name__ == "__main__":
    main()
