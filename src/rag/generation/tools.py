from typing import List, Dict, Any, Callable, Optional
import json
from src.rag.retrieval.retriever import Retriever
from src.rag.core.schema import RetrievedChunk

# 1. Mapeo de nombres comunes a Tickers (mejora la precisión del filtro)
TICKER_MAP = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "NVIDIA": "NVDA",
    "INTEL": "INTC",
    "CISCO": "CSCO",
    "ASML": "ASML",
    "MICRON": "MU",
    "AMD": "AMD"
}

# 2. Definición del Esquema de la Herramienta (JSON Schema)
SEARCH_EARNINGS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_earnings_calls",
        "description": "Busca fragmentos de transcripts de earnings calls (2019-2020). "
                       "SIEMPRE incluye el nombre de la empresa en el parámetro 'query' para mejores resultados. "
                       "Usa esta herramienta para cualquier dato financiero, estrategias o comentarios de ejecutivos.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Consulta de búsqueda. DEBE incluir el nombre de la empresa (ej: 'Apple revenue Q4 2019', 'Microsoft cloud strategy')."
                },
                "company_id": {
                    "type": "string",
                    "description": "Símbolo o nombre de la empresa a filtrar (ej: 'AAPL', 'Apple'). Altamente recomendado para evitar ruidos de otras empresas."
                },
                "num_results": {
                    "type": "integer",
                    "description": "Número de trozos a recuperar. Mínimo recomendado: 8-10.",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
}

# 2. Esquema para listar empresas disponibles
LIST_COMPANIES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_available_companies",
        "description": "Retorna la lista de todas las empresas (symbols) que tienen transcripts disponibles en la base de datos (2019-2020)."
    }
}

# Lista de todas las herramientas disponibles para el LLM
AVAILABLE_TOOLS_SCHEMAS = [SEARCH_EARNINGS_TOOL_SCHEMA, LIST_COMPANIES_TOOL_SCHEMA]


class ToolManager:
    """Clase para gestionar la ejecución de herramientas locales."""
    
    def __init__(self):
        # Inicializar el retriever una sola vez
        print("   [ToolManager] Inicializando Retriever...")
        self.retriever = Retriever()
        
    def list_available_companies(self) -> str:
        """Retorna los símbolos de empresas únicos en la base de datos."""
        print("   [Tool usada] Listando empresas disponibles...")
        # Acceso directo a la metadata_db para eficiencia
        import sqlite3
        import json
        with sqlite3.connect(self.retriever.metadata_db.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT metadata FROM chunks")
            rows = cursor.fetchall()
            companies = set()
            for r in rows:
                meta = json.loads(r[0])
                if meta.get("company"):
                    companies.add(meta.get("company"))
        
        return json.dumps(sorted(list(companies)))

    def search_earnings_calls(self, query: str, company_id: Optional[str] = None, num_results: int = 15) -> str:
        """
        Implementación real de la herramienta de búsqueda con soporte de filtrado por empresa y boost temporal.
        """
        filter_msg = f" (Filtrando por: {company_id})" if company_id else ""
        search_num = max(num_results, 15)
        
        # Aumentamos agresivamente k para compensar ruidos de embeddings
        search_k = search_num + 50
        print(f"   [Tool usada] Buscando: '{query}'{filter_msg} (pidiendo {search_k} chunks a la DB)")
        
        chunks: List[RetrievedChunk] = self.retriever.search(query, top_k=search_k)
        
        if not chunks:
            return json.dumps({"results": [], "message": "No se encontró información relevante."})
        
        # 1. Filtro por empresa
        filtered_chunks = []
        if company_id:
            top_company = company_id.upper().strip()
            ticker = TICKER_MAP.get(top_company, top_company) # Intentar mapeo (ej: Microsoft -> MSFT)
            
            for chunk in chunks:
                company_meta = str(chunk.metadata.get("company", "")).upper()
                # Coincidencia exacta o contenida (ej: 'AAPL' in 'AAPL' o 'MSFT' in 'MICROSOFT' si se mapeó)
                if ticker in company_meta or company_meta in ticker:
                    filtered_chunks.append(chunk)
            
            if not filtered_chunks:
                print(f"   [Tool Manager] No se hallaron chunks de {company_id} ({ticker}) en los top {search_k}.")
                filtered_chunks = chunks
        else:
            filtered_chunks = chunks

        # 2. Boost Temporal (Basado en año y trimestre en la query)
        import re
        query_upper = query.upper()
        # Buscar años (2019, 2020) y trimestres (Q1, Q2, Q3, Q4)
        found_years = re.findall(r"(2019|2020)", query_upper)
        found_quarters = re.findall(r"(Q[1-4])", query_upper)

        if found_years or found_quarters:
            # print(f"   [Tool Manager] Aplicando boost para: years={found_years}, quarters={found_quarters}")
            for chunk in filtered_chunks:
                boost = 0.0
                chunk_meta = chunk.metadata
                
                # Boost por año coincidente (0.1 es un boost significativo)
                if str(chunk_meta.get("year")) in found_years:
                    boost += 0.05
                
                # Boost por trimestre coincidente
                if chunk_meta.get("quarter") in found_quarters:
                    boost += 0.05
                
                chunk.score += boost

            # Re-ordenar después del boost
            filtered_chunks.sort(key=lambda x: x.score, reverse=True)

        # Truncar al número solicitado
        final_chunks = filtered_chunks[:search_num]
        
        # Formatear resultados como JSON
        results = []
        for chunk in final_chunks:
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
        Despacha la llamada a la función correcta basada en el nombre de la herramienta.
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
