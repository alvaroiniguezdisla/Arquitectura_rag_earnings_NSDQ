import sys
from pathlib import Path
import time

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.pipeline.step1_loader import load_processed_corpus
from src.rag.pipeline.step2_chunking import chunk_documents
from src.rag.pipeline.step3_embedding import EmbeddingModel
from src.rag.pipeline.step4_indexing import index_data
from src.rag.core.config import CORPUS_FILE
from src.rag.core.logger import get_logger

logger = get_logger(__name__)

def run_full_pipeline():
    logger.info("Iniciando Re-Indexado Completo (Fase 2)...")
    start_time = time.time()
    
    # 1. Cargar Corpus (Ya filtrado por prepare_corpus.py)
    if not CORPUS_FILE.exists():
        logger.error(f"No existe {CORPUS_FILE}. Ejecuta prepare_corpus.py primero.")
        return
        
    documents = load_processed_corpus(CORPUS_FILE)
    if not documents:
        logger.error("Corpus vacio.")
        return

    # 2. Chunking
    logger.info("[Step 2] Chunking Documents...")
    chunks = chunk_documents(documents)
    logger.info(f"Generados {len(chunks)} chunks.")

    # 3. Model Loading
    logger.info("[Step 3] Loading Embedding Model...")
    embed_model = EmbeddingModel()

    # 4. Indexing (Unified Store)
    index_data(chunks, embed_model)
    
    total_time = time.time() - start_time
    logger.info(f"Pipeline Finalizado en {total_time:.2f}s")
    logger.info("Ahora puedes probar el chat con: python scripts/chat_cli.py")

if __name__ == "__main__":
    run_full_pipeline()
