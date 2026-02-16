from typing import List, Optional
from pathlib import Path

from src.rag.core.schema import RetrievedChunk
from src.rag.pipeline.step3_embedding import EmbeddingModel
from src.rag.storage.unified_store import UnifiedDocumentStore
from src.rag.retrieval.reranker import Reranker
from src.rag.core.config import SQLITE_DB_PATH, TOP_K, EMBEDDING_DIMENSION, RERANK_INITIAL_TOP_K
from src.rag.core.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Sistema de busqueda semantica.
    Usa UnifiedDocumentStore (SQLite-only) para recuperar informacion.
    Incorpora Re-ranking con Cross-Encoder para mejorar precision.
    """

    def __init__(
        self,
        db_path: Path = SQLITE_DB_PATH,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        logger.info("Inicializando UnifiedStore...")

        # Store unificado (un solo archivo .db)
        self.store = UnifiedDocumentStore(db_path, dimension=EMBEDDING_DIMENSION)

        # Modelo de embeddings
        if embedding_model is None:
            self.embed_model = EmbeddingModel()
        else:
            self.embed_model = embedding_model
            
        # Modelo Cross-Encoder para Re-ranking
        self.reranker = Reranker()

    def get_available_companies(self) -> List[str]:
        """Retorna la lista de empresas disponibles."""
        return self.store.get_companies()

    def search(self, query: str, top_k: int = TOP_K, filter_company: Optional[str] = None, filter_year: Optional[int] = None, filter_quarter: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Busca chunks relevantes delegando en el Store Unificado y re-rankeando.

        Args:
            query: Pregunta del usuario
            top_k: Numero de resultados finales
            filter_company: Ticker o nombre para filtrar
            filter_year: Año (int)
            filter_quarter: Trimestre (int)

        Returns:
            Lista de RetrievedChunk ordenados por relevancia (Cross-Encoder score)
        """
        # Paso 1: Convertir la pregunta a vector
        query_vector = self.embed_model.embed_text(query)

        # Paso 2: Delegar al Store (recuperamos mas candidatos para re-rankear)
        # Recuperamos max(RERANK_INITIAL_TOP_K, top_k * 3) para tener margen
        initial_k = max(RERANK_INITIAL_TOP_K, top_k * 3)
        
        candidates = self.store.search(
            query_vector=query_vector,
            top_k=initial_k,
            filter_company=filter_company,
            filter_year=filter_year,
            filter_quarter=filter_quarter,
            query_text=query
        )
        
        # Paso 3: Re-ranking con Cross-Encoder
        # score original (cosine) se sobrescribe con score del cross-encoder
        results = self.reranker.rerank(query, candidates, top_k=top_k)

        return results
