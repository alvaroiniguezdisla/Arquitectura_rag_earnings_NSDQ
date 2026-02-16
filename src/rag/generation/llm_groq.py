import os
import requests
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from src.rag.core.schema import RetrievedChunk
from src.rag.core.config import (
    LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    HTTP_TIMEOUT, HTTP_MAX_RETRIES, HTTP_BACKOFF_FACTOR,
)
from src.rag.core.prompts import FINANCIAL_ASSISTANT_PROMPT
from src.rag.generation.tools import ToolManager, AVAILABLE_TOOLS_SCHEMAS
from src.rag.core.logger import get_logger

# Cargar variables de entorno
load_dotenv()

logger = get_logger(__name__)


class GroqLLM:
    """
    Wrapper para Groq API (LLM en la nube).
    Soporta generación simple (RAG directo) y Tool Calling (agente).
    """
    
    def __init__(self, api_key: str = None, model: str = None, tool_manager: "ToolManager" = None):
        """
        Inicializa el cliente Groq.
        
        Args:
            api_key: API key de Groq (si no se pasa, se lee de .env)
            model: Modelo a usar (si no se pasa, usa LLM_MODEL_NAME de config.py)
            tool_manager: Instancia de ToolManager (si no se pasa, se crea una nueva).
                          Permite inyectar mocks en tests o configuraciones custom.
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
        
        # --- Tool Manager (inyección de dependencias) ---
        # Si no te pasan uno, se crea aquí (lazy: solo cuando realmente lo necesitas)
        self.tool_manager = tool_manager or ToolManager()
        
        # --- Session con retry automático ---
        # Si Groq devuelve 429 (rate limit) o 5xx (error de servidor),
        # reintenta hasta HTTP_MAX_RETRIES veces con espera exponencial.
        self.session = requests.Session()
        retry_strategy = Retry(
            total=HTTP_MAX_RETRIES,
            backoff_factor=HTTP_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def chat_with_tools(self, user_query: str, history: List[Dict[str, str]] = None) -> str:
        """
        Ejecuta el flujo completo de Tool Calling:
        1. Prepara el contexto (Sistema + Historia + Pregunta)
        2. Envía al LLM
        3. Si el LLM pide usar tools -> las ejecuta y devuelve resultados
        4. El LLM genera la respuesta final con la info de las tools
        
        Args:
            user_query: Pregunta del usuario
            history: (Opcional) Lista de mensajes anteriores para dar contexto
            
        Returns:
            Respuesta final del asistente
        """
        # 1. Mensaje del Sistema (Personalidad)
        sys_msg = {
            "role": "system", 
            "content": FINANCIAL_ASSISTANT_PROMPT
        }
        
        # 2. Construimos la lista completa de mensajes
        messages = [sys_msg]
        
        # Si hay historia, la añadimos antes de la pregunta actual
        if history:
            messages.extend(history)
            
        # Añadimos la pregunta actual al final
        messages.append({"role": "user", "content": user_query})
        
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
            response = self.session.post(
                self.api_url, headers=self.headers, json=payload,
                timeout=HTTP_TIMEOUT
            )
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"Groq API response: {json.dumps(response_data, indent=2)}")
            
            response_message = response_data["choices"][0]["message"]
            tool_calls = response_message.get("tool_calls")
            
            # Si el LLM NO quiere usar herramientas, devolvemos su respuesta directa
            if not tool_calls:
                logger.info("LLM respondió directamente (sin tools)")
                return response_message["content"]
            
            # --- PASO 2: Ejecutar herramientas ---
            logger.info(f"LLM solicitó {len(tool_calls)} tool(s)")
            # Añadir la respuesta del asistente (con la intención de llamar tools) al historial
            messages.append(response_message)
            
            for tool_call in tool_calls:
                # Ejecutar la función real
                tool_result_json = self.tool_manager.execute_tool_call(tool_call)
                
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
            
            final_response = self.session.post(
                self.api_url, headers=self.headers, json=final_payload,
                timeout=HTTP_TIMEOUT
            )
            final_response.raise_for_status()
            final_data = final_response.json()
            
            logger.info("Respuesta final generada con contexto de tools")
            return final_data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en Groq API: {e}")
            return f"Error en Groq API: {str(e)}"
        except Exception as e:
            logger.error(f"Error inesperado: {e}")
            return f"Error inesperado: {str(e)}"
