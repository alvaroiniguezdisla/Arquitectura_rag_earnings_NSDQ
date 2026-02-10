import os
import requests
from typing import List
from dotenv import load_dotenv

from src.rag.schema import RetrievedChunk
from src.rag.config import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Cargar variables de entorno
load_dotenv()


class GroqLLM:
    """
    Wrapper para Groq API (LLM en la nube).
    Genera respuestas usando los chunks recuperados como contexto.
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
    
    def generate_response(
        self, 
        query: str, 
        retrieved_chunks: List[RetrievedChunk],
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """
        Genera una respuesta usando RAG.
        
        Args:
            query: Pregunta del usuario
            retrieved_chunks: Chunks recuperados por el retriever
            max_tokens: Longitud máxima de la respuesta (usa config si no se pasa)
            temperature: Temperatura para generación (usa config si no se pasa)
            
        Returns:
            Respuesta generada por el LLM
        """
        # Usar valores de config si no se pasan
        max_tokens = max_tokens or LLM_MAX_TOKENS
        temperature = temperature or LLM_TEMPERATURE
        
        # Construir el contexto a partir de los chunks
        context = self._build_context(retrieved_chunks)
        
        # Crear el prompt del sistema
        system_prompt = """Eres un asistente financiero experto en earnings calls (conferencias de resultados financieros).
Tu trabajo es responder preguntas usando ÚNICAMENTE la información del contexto proporcionado.

Reglas:
1. Responde SOLO con información del contexto
2. Cita la empresa y el trimestre cuando sea relevante
3. Si la información no está en el contexto, di "No tengo información sobre eso en los documentos disponibles"
4. Sé conciso y directo
"""
        
        # Crear el prompt del usuario
        user_prompt = f"""Contexto de earnings calls:
{context}

Pregunta: {query}

Respuesta:"""
        
        # Hacer la petición a Groq
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            return data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            return f"Error al conectar con Groq API: {str(e)}"
    
    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Construye el contexto a partir de los chunks recuperados.
        
        Args:
            chunks: Lista de chunks recuperados
            
        Returns:
            Contexto formateado como string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Extraer metadata útil
            company = chunk.metadata.get("company", "Unknown")
            year = chunk.metadata.get("year", "Unknown")
            
            context_parts.append(
                f"[Documento {i}] {company} - {year}\n{chunk.text}\n"
            )
        
        return "\n---\n".join(context_parts)
