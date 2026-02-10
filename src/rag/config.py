from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# --- Rutas Base ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

# --- Archivos Clave ---
CORPUS_FILE = PROCESSED_DATA_DIR / "corpus.jsonl"
SQLITE_DB_PATH = INDEX_DIR / "metadata.db"
FAISS_INDEX_PATH = INDEX_DIR / "vectors.index"

# --- Configuración RAG (MVP) ---
# Tamaño de los trozos de texto
CHUNK_SIZE = 800  
# Solape para no cortar frases
CHUNK_OVERLAP = 100 
# Cuántos trozos recuperar por pregunta
TOP_K = 8 

# Modelo de Embeddings (Local y rápido)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Modelo LLM (Ruta local o nombre)
# Ajusta esto a donde tengas tu modelo .gguf
# Ejemplo: "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = os.getenv("MODEL_PATH", "models/tinyllama.gguf")
