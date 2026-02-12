from typing import List, Dict, Any, Callable, Optional
import json
from src.rag.retrieval.retriever import Retriever
from src.rag.core.schema import RetrievedChunk

# --- 1. Definición de Esquemas (La "Carta" del Menú) ---

SEARCH_EARNINGS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_earnings_calls",
        "description": (
            "URGENTE: Herramienta OBLIGATORIA para cualquier pregunta sobre datos financieros, "
            "resultados, ingresos, beneficios, estrategia, directivos o comentarios de earnings calls.\n"
            "NO intentes responder de memoria. Si el usuario pregunta 'dame los ingresos de Apple', "
            "USA ESTA HERRAMIENTA.\n"
            "Retorna fragmentos de texto reales de las conferencias."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La pregunta completa o palabras clave. Ej: 'Apple revenue Q4 2019', 'Microsoft cloud strategy'."
                },
                "company_id": {
                    "type": "string",
                    "description": "El Ticker o Nombre de la empresa, si aplica. Ej: 'AAPL', 'Apple', 'MSFT'. Ayuda mucho a filtrar."
                },
                "num_results": {
                    "type": "integer",
                    "description": "Cantidad de fragmentos a leer. Por defecto 10.",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
}

LIST_COMPANIES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_available_companies",
        "description": "Usa esta herramienta SI Y SOLO SI el usuario pregunta qué empresas hay disponibles o qué datos tenemos."
    }
}

AVAILABLE_TOOLS_SCHEMAS = [SEARCH_EARNINGS_TOOL_SCHEMA, LIST_COMPANIES_TOOL_SCHEMA]


class ToolManager:
    """
    Gestor de herramientas.
    Ahora actúa como un simple 'Router' o 'Conector':
    Recibe la petición del LLM y la delega al experto correspondiente (Retriever).
    """
    
    def __init__(self):
        # Inicializar el retriever una sola vez
        print("   [ToolManager] Conectando con el Retriever...")
        self.retriever = Retriever()
        
    def list_available_companies(self) -> str:
        """Delega en el Retriever para ver qué hay."""
        print("   [Tool usada] Listando empresas...")
        companies = self.retriever.get_available_companies()
        return json.dumps(companies)

    def search_earnings_calls(self, query: str, company_id: Optional[str] = None, num_results: int = 10) -> str:
        """
        Delega la búsqueda inteligente al Retriever.
        """
        print(f"   [Tool usada] Buscando: '{query}' (Filtro: {company_id})")
        
        # Delegamos toda la lógica compleja (filtros, boost, etc.) al Retriever
        chunks = self.retriever.search(query, top_k=num_results, filter_company=company_id)
        
        if not chunks:
            return json.dumps({"results": [], "message": "No se encontró información relevante en la base de datos."})
            
        # Formateamos la respuesta para el LLM (JSON ligero)
        results = []
        for chunk in chunks:
            results.append({
                "company": chunk.metadata.get("company", "Unknown"),
                "year": chunk.metadata.get("year", "Unknown"),
                "quarter": chunk.metadata.get("quarter", "Unknown"),
                "content": chunk.text,
                "score": round(chunk.score, 4)
            })
            
        return json.dumps(results, ensure_ascii=False)

    def execute_tool_call(self, tool_call) -> str:
        """
        Despacha la llamada a la función correcta.
        """
        try:
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            
            if isinstance(arguments, str):
                function_args = json.loads(arguments)
            else:
                function_args = arguments
            
            if function_name == "search_earnings_calls":
                return self.search_earnings_calls(
                    query=function_args.get("query"),
                    company_id=function_args.get("company_id"),
                    num_results=function_args.get("num_results", 10)
                )
            elif function_name == "list_available_companies":
                return self.list_available_companies()
            else:
                return json.dumps({"error": f"Herramienta desconocida: {function_name}"})
        except Exception as e:
            return json.dumps({"error": f"Error ejecutando tool: {str(e)}"})


# Instancia global del ToolManager
tool_manager = ToolManager()
