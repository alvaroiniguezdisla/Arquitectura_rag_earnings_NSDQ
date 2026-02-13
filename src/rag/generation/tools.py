from typing import List, Dict, Any, Callable, Optional
import json
from src.rag.retrieval.retriever import Retriever
from src.rag.ml.predictor import FinancialPredictor
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

PREDICT_OUTLOOK_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "predict_financial_outlook",
        "description": (
            "Usa esta herramienta cuando el usuario pida una PREDICCIÓN, PROYECCIÓN o ANÁLISIS DE SENTIMIENTO sobre el futuro de la empresa. "
            "Ej: '¿Cuál es el outlook de Apple?', '¿Subirán los ingresos de Microsoft?', 'Dame una predicción basada en el call'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "El tema específico a analizar. Ej: 'revenue growth', 'guidance', 'outlook'."
                },
                "company_id": {
                    "type": "string",
                    "description": "Ticker de la empresa. OBLIGATORIO. Ej: 'AAPL', 'MSFT'."
                },
                "year": {
                    "type": "integer",
                    "description": "Año del reporte. Ej: 2020."
                },
                "quarter": {
                    "type": "integer",
                    "description": "Trimestre. Ej: 3."
                }
            },
            "required": ["query", "company_id", "year", "quarter"]
        }
    }
}

AVAILABLE_TOOLS_SCHEMAS = [SEARCH_EARNINGS_TOOL_SCHEMA, LIST_COMPANIES_TOOL_SCHEMA, PREDICT_OUTLOOK_TOOL_SCHEMA]


class ToolManager:
    """
    Gestor de herramientas.
    Ahora actúa como un simple 'Router' o 'Conector':
    Recibe la petición del LLM y la delega al experto correspondiente (Retriever).
    """
    
    def __init__(self):
        # Inicializar el retriever una sola vez
        print("   [ToolManager] Conectando con el Retriever y ML Predictor...")
        self.retriever = Retriever()
        self.predictor = FinancialPredictor()
        
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

    def predict_financial_outlook(self, query: str, company_id: str, year: int, quarter: int) -> str:
        """
        Ejecuta el pipeline de ML:
        1. Busca texto relevante (Guidance/Outlook) usando el Retriever.
        2. Manda ese texto al FinancialPredictor.
        3. Devuelve resultado estructurado.
        """
        print(f"   [Tool usada] Prediciendo Outlook para {company_id} Q{quarter} {year}...")
        
        # 1. Recuperar contexto (textos que hablen de guidance, outlook, future)
        # Forzamos la query para buscar partes predictivas
        search_query = f"{query} guidance outlook future expectations"
        chunks = self.retriever.search(search_query, top_k=5, filter_company=company_id, filter_year=year, filter_quarter=quarter)
        
        if not chunks:
            return json.dumps({"error": f"No se encontraron transcripts para {company_id} Q{quarter} {year}."})
            
        # Concatenar texto
        full_text = "\n".join([c.text for c in chunks])
        
        # 2. Obtener Revenue (Si el retriever devolviera metadatos ricos podríamos sacarlo de ahí, 
        # pero por ahora usaremos 0 o un placeholder, ya que el modelo depende más del sentimiento)
        # TODO: Implementar búsqueda de revenue real o pasarlo como argumento si el LLM lo sabe.
        current_revenue = 0.0 
        
        # 3. Predecir
        result = self.predictor.predict(full_text, current_revenue=current_revenue, quarter=quarter)
        
        # 4. Formatear
        response = {
            "source_chunks": len(chunks),
            "prediction_result": result,
            "note": "Predicción basada en análisis de sentimiento del transcript recuperado."
        }
        
        return json.dumps(response, ensure_ascii=False)

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
            elif function_name == "predict_financial_outlook":
                return self.predict_financial_outlook(
                    query=function_args.get("query"),
                    company_id=function_args.get("company_id"),
                    year=function_args.get("year"),
                    quarter=function_args.get("quarter")
                )
            else:
                return json.dumps({"error": f"Herramienta desconocida: {function_name}"})
        except Exception as e:
            return json.dumps({"error": f"Error ejecutando tool: {str(e)}"})


# Instancia global del ToolManager
tool_manager = ToolManager()
