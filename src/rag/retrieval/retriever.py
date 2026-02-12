from typing import List, Optional
from pathlib import Path

from src.rag.core.schema import RetrievedChunk
from src.rag.pipeline.step3_embedding import EmbeddingModel
from src.rag.storage.unified_store import UnifiedDocumentStore
from src.rag.core.config import SQLITE_DB_PATH, TOP_K, EMBEDDING_DIMENSION


class Retriever:
    """
    Sistema de busqueda semantica.
    Usa UnifiedDocumentStore (SQLite-only) para recuperar informacion.
    """

    def __init__(
        self,
        db_path: Path = SQLITE_DB_PATH,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        print("Retriever: Inicializando UnifiedStore...")

        # Store unificado (un solo archivo .db)
        self.store = UnifiedDocumentStore(db_path, dimension=EMBEDDING_DIMENSION)

        # Modelo de embeddings
        if embedding_model is None:
            self.embed_model = EmbeddingModel()
        else:
            self.embed_model = embedding_model

    def get_available_companies(self) -> List[str]:
        """Retorna la lista de empresas disponibles."""
        return self.store.get_companies()

    def search(self, query: str, top_k: int = TOP_K, filter_company: Optional[str] = None) -> List[RetrievedChunk]:
        """
        Busca chunks relevantes delegando en el Store Unificado.

        Args:
            query: Pregunta del usuario
            top_k: Numero de resultados
            filter_company: Ticker o nombre para filtrar

        Returns:
            Lista de RetrievedChunk ordenados
        """
        # Paso 1: Convertir la pregunta a vector
        query_vector = self.embed_model.embed_text(query)

        # Paso 2: Delegar al Store (pasamos query_text para boost temporal)
        results = self.store.search(
            query_vector=query_vector,
            top_k=top_k,
            filter_company=filter_company,
            query_text=query
        )

        return results
