from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# --- Rutas Base ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

# --- Archivos Clave ---
CORPUS_FILE = PROCESSED_DATA_DIR / "corpus.jsonl"
SQLITE_DB_PATH = INDEX_DIR / "unified_store.db"

# --- Configuración RAG (MVP) ---
# Tamaño de los trozos de texto
CHUNK_SIZE = 800  
# Solape para no cortar frases
CHUNK_OVERLAP = 100 
# Cuántos trozos recuperar por pregunta
TOP_K = 12 

# Modelo de Embeddings (Local y rápido)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Dimensión de los vectores del modelo de embeddings
EMBEDDING_DIMENSION = 384

# Modelo LLM (Groq API)
# Llama 3.3 70B Versatile - sucesor de llama3-70b-8192 (deprecado por Groq)
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
# Temperatura para la generación (0-1, menor = más conservador)
LLM_TEMPERATURE = 0.3
# Máximo de tokens en la respuesta
LLM_MAX_TOKENS = 512

# Procesamiento por lotes (batching)
# Tamaño de batch para generar embeddings (evita saturar RAM)
BATCH_SIZE = 64

# --- Configuración HTTP (resiliencia) ---
# Tiempo máximo de espera por respuesta (segundos)
HTTP_TIMEOUT = 30
# Máximo de reintentos ante errores transitorios (429, 500, 502, 503, 504)
HTTP_MAX_RETRIES = 4
# Factor de backoff exponencial: espera 1s, 2s, 4s, 8s entre reintentos
HTTP_BACKOFF_FACTOR = 1


# --- Mapeos de Negocio ---
# Mapeo de nombres comunes a Tickers (mejora la precisión del filtro)
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


