from typing import List, Optional, Dict
from pathlib import Path

from src.rag.core.schema import RetrievedChunk
from src.rag.pipeline.step3_embedding import EmbeddingModel
from src.rag.storage.vector_store import VectorDB
from src.rag.storage.metadata_store import MetadataDB
from src.rag.core.config import FAISS_INDEX_PATH, SQLITE_DB_PATH, TOP_K, EMBEDDING_DIMENSION


class Retriever:
    """
    Sistema de búsqueda semántica (el corazón del RAG).
    Combina FAISS (búsqueda vectorial) + SQLite (metadata).
    """
    
    def __init__(
        self,
        faiss_index_path: Path = FAISS_INDEX_PATH,
        sqlite_db_path: Path = SQLITE_DB_PATH,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        Inicializa el retriever.
        
        Args:
            faiss_index_path: Ruta al índice FAISS
            sqlite_db_path: Ruta a la base de datos SQLite
            embedding_model: Modelo de embeddings (se crea si no se pasa)
        """
        print("Retriever: Inicializando...")
        
        # Cargar bases de datos
        self.vector_db = VectorDB(faiss_index_path, dimension=EMBEDDING_DIMENSION)
        self.metadata_db = MetadataDB(sqlite_db_path)
        
        # Cargar modelo de embeddings
        if embedding_model is None:
            self.embed_model = EmbeddingModel()
        else:
            self.embed_model = embedding_model
        
        print(f"   OK: FAISS: {self.vector_db.count()} vectores")
        print(f"   OK: SQLite: {self.metadata_db.count_chunks()} chunks")
    
    def search(self, query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
        """
        Busca chunks relevantes para la consulta.
        
        Args:
            query: Pregunta del usuario
            top_k: Número de resultados a devolver
            
        Returns:
            Lista de RetrievedChunk ordenados por relevancia
        """
        # Paso 1: Convertir la pregunta a vector
        query_vector = self.embed_model.embed_text(query)
        
        # Paso 2: Buscar en FAISS los vecinos más cercanos
        chunk_ids, distances = self.vector_db.search(query_vector, top_k)
        
        # Paso 3: Recuperar metadata y texto de SQLite
        chunks_list = self.metadata_db.get_chunks_by_ids(chunk_ids)
        
        # Paso 4: Reordenar los resultados de SQLite para que coincidan con chunk_ids (orden de FAISS)
        # Esto es CRÍTICO para que el zip con distances sea correcto.
        chunk_map = {c.chunk_id: c for c in chunks_list}
        ordered_chunks = []
        for cid in chunk_ids:
            if cid in chunk_map:
                ordered_chunks.append(chunk_map[cid])
        
        # Paso 5: Crear objetos RetrievedChunk con scores
        retrieved_chunks = []
        # Nota: Usamos zip con ordered_chunks, pero distances tiene la misma longitud que chunk_ids.
        # Si algún ID no se encontró en SQLite, usamos el índice de distance correspondiente.
        for i, chunk in enumerate(ordered_chunks):
            # Obtener el índice original del ID para buscar su distancia
            # (En teoría el orden ya coincide, pero buscamos el id para ser seguros)
            dist_idx = chunk_ids.index(chunk.chunk_id)
            distance = distances[dist_idx]
            
            # Convertir distancia euclidiana a "score de similitud"
            score = 1.0 / (1.0 + distance)
            
            retrieved_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=score,
                doc_id=chunk.doc_id,
                metadata=chunk.metadata
            )
            retrieved_chunks.append(retrieved_chunk)
        
        return retrieved_chunks
