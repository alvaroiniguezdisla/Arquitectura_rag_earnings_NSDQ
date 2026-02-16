from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.core.config import EMBEDDING_MODEL_NAME
from src.rag.core.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Wrapper para el modelo de embeddings.
    Convierte texto en vectores numéricos.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Inicializa el modelo de embeddings.
        
        Args:
            model_name: Nombre del modelo en HuggingFace
        """
        logger.info(f"Cargando modelo de embeddings: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Modelo cargado. Dimensión de vector: {self.dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convierte un texto en un vector numérico.
        
        Args:
            text: Texto a convertir
            
        Returns:
            Vector numpy de dimensión (dimension,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convierte múltiples textos en vectores (batch processing).
        
        Args:
            texts: Lista de textos
            
        Returns:
            Array numpy de dimensión (len(texts), dimension)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
