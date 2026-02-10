import os
import requests
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from src.rag.core.schema import RetrievedChunk
from src.rag.core.config import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS
from src.rag.generation.tools import tool_manager, AVAILABLE_TOOLS_SCHEMAS

# Cargar variables de entorno
load_dotenv()


class GroqLLM:
    """
    Wrapper para Groq API (LLM en la nube).
    Soporta generación simple (RAG directo) y Tool Calling (agente).
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Inicializa el cliente Groq.
        
        Args:
            api_key: API key de Groq (si no se pasa, se lee de .env)
            model: Modelo a usar (si no se pasa, usa LLM_MODEL_NAME de config.py)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY no encontrada. "
                "Agrégala al archivo .env o pásala como parámetro."
            )
        
        self.model = model or LLM_MODEL_NAME
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_with_tools(self, user_query: str) -> str:
        """
        Ejecuta el flujo completo de Tool Calling:
        1. Envía pregunta + definición de tools al LLM
        2. Si el LLM pide usar tools -> las ejecuta y devuelve resultados
        3. El LLM genera la respuesta final con la info de las tools
        
        Args:
            user_query: Pregunta del usuario
            
        Returns:
            Respuesta final del asistente
        """
        # Historial de mensajes de la conversación actual
        messages = [
            {
                "role": "system", 
                "content": "You are a specialized financial assistant for NASDAQ 2019-2020 Earnings Calls.\n"
                           "RESOURCES:\n"
                           "- `list_available_companies`: Use this for general questions about what companies or data are available.\n"
                           "- `search_earnings_calls`: Use this for ANY specific question about results, revenue, or strategy.\n"
                           "RULES:\n"
                           "1. MANDATORY: For search, ALWAYS fill both `query` and `company_id`.\n"
                           "2. If a user asks 'What companies do you have?', use `list_available_companies` first.\n"
                           "3. Never guess data. If the tool returns empty, say the 2019-2020 database doesn't have it."
            },
            {"role": "user", "content": user_query}
        ]
        
        # --- PASO 1: Primera llamada al LLM (con tools) ---
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": AVAILABLE_TOOLS_SCHEMAS,
            "tool_choice": "auto",  # El LLM decide si usar tools o no
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            
            response_message = response_data["choices"][0]["message"]
            tool_calls = response_message.get("tool_calls")
            
            # Si el LLM NO quiere usar herramientas, devolvemos su respuesta directa
            if not tool_calls:
                return response_message["content"]
            
            # --- PASO 2: Ejecutar herramientas ---
            # Añadir la respuesta del asistente (con la intención de llamar tools) al historial
            messages.append(response_message)
            
            for tool_call in tool_calls:
                # Ejecutar la función real
                tool_result_json = tool_manager.execute_tool_call(tool_call)
                
                # Añadir el resultado al historial como mensaje de rol 'tool'
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "content": tool_result_json
                })
            
            # --- PASO 3: Segunda llamada al LLM (con resultados) ---
            # Ahora el LLM tiene la pregunta original + su decisión de llamar tools + los resultados
            final_payload = {
                "model": self.model,
                "messages": messages,
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS
            }
            
            final_response = requests.post(self.api_url, headers=self.headers, json=final_payload)
            final_response.raise_for_status()
            final_data = final_response.json()
            
            return final_data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            return f"Error en Groq API: {str(e)}"
        except Exception as e:
            return f"Error inesperado: {str(e)}"


