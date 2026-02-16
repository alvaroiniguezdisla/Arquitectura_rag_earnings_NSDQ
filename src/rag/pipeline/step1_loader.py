import json
from pathlib import Path
from typing import List

from src.rag.core.schema import Document
from src.rag.core.config import CORPUS_FILE
from src.rag.core.logger import get_logger

logger = get_logger(__name__)


def load_processed_corpus(corpus_path: Path = CORPUS_FILE) -> List[Document]:
    """
    Lee el archivo corpus.jsonl y convierte cada línea a un objeto Document.
    
    Args:
        corpus_path: Ruta al archivo corpus.jsonl
        
    Returns:
        Lista de objetos Document listos para procesar
    """
    documents = []
    
    if not corpus_path.exists():
        logger.error(f"No se encuentra el archivo: {corpus_path}")
        return documents
    
    logger.info(f"Leyendo corpus desde: {corpus_path}")
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                data = json.loads(line.strip())
                
                # Crear objeto Document desde el JSON
                doc = Document(
                    doc_id=data.get("doc_id", f"doc_{line_num}"),
                    text=data.get("text", ""),
                    metadata=data.get("metadata", {})
                )
                
                documents.append(doc)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Error en linea {line_num}: {e}")
                continue
    
    logger.info(f"Cargados {len(documents)} documentos")
    return documents
