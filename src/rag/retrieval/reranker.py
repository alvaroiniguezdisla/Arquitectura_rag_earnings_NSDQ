from typing import List
from sentence_transformers import CrossEncoder
from src.rag.core.schema import RetrievedChunk
from src.rag.core.config import RERANKER_MODEL_NAME
from src.rag.core.logger import get_logger

logger = get_logger(__name__)

class Reranker:
    """
    Componente de Re-ranking usando Cross-Encoders.
    Mejora la precisión re-evaluando los top-N candidatos recuperados por el Bi-Encoder.
    """
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model_name = model_name
        logger.info(f"Cargando modelo Cross-Encoder: {model_name} ...")
        # device='cpu' es seguro, si hay GPU disponible sentence-transformers lo usa auto si no se especifica?
        # Mejor dejar default.
        self.model = CrossEncoder(model_name)
        logger.info("Modelo Cross-Encoder cargado correctamente.")

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """
        Re-ordena una lista de chunks asignando un nuevo score de relevancia.
        
        Args:
            query: La consulta del usuario.
            chunks: Lista de candidatos recuperados inicialmente.
            top_k: Número de chunks a devolver después del re-ranking.
            
        Returns:
            Lista de top_k chunks ordenados por el nuevo score.
        """
        if not chunks:
            return []

        # Preparar pares (query, texto)
        # CrossEncoder espera lista de tuplas o listas: [[q, doc1], [q, doc2], ...]
        pairs = [[query, chunk.text] for chunk in chunks]

        # Predicción (scores no acotados, suelen ser logits)
        scores = self.model.predict(pairs)

        # Actualizar scores e imprimir debug
        for chunk, new_score in zip(chunks, scores):
            # Guardamos el score original en metadata si queremos compararlos luego (opcional)
            chunk.metadata["initial_score"] = chunk.score
            chunk.score = float(new_score)

        # Ordenar descendente por nuevo score
        chunks.sort(key=lambda x: x.score, reverse=True)

        # Retornar top-k
        return chunks[:top_k]
