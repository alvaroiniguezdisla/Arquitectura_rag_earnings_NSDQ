from typing import List
from pathlib import Path

from src.rag.schema import RetrievedChunk
from src.rag.embeddings import EmbeddingModel
from src.rag.vdb_faiss import VectorDB
from src.rag.vdb_sqlite import MetadataDB
from src.rag.config import FAISS_INDEX_PATH, SQLITE_DB_PATH, TOP_K, EMBEDDING_DIMENSION


class Retriever:
    """
    Sistema de búsqueda semántica (el corazón del RAG).
    Combina FAISS (búsqueda vectorial) + SQLite (metadata).
    """
    
    def __init__(
        self,
        faiss_index_path: Path = FAISS_INDEX_PATH,
        sqlite_db_path: Path = SQLITE_DB_PATH,
        embedding_model: EmbeddingModel = None
    ):
        """
        Inicializa el retriever.
        
        Args:
            faiss_index_path: Ruta al índice FAISS
            sqlite_db_path: Ruta a la base de datos SQLite
            embedding_model: Modelo de embeddings (se crea si no se pasa)
        """
        print("🔍 Inicializando Retriever...")
        
        # Cargar bases de datos
        self.vector_db = VectorDB(faiss_index_path, dimension=EMBEDDING_DIMENSION)
        self.metadata_db = MetadataDB(sqlite_db_path)
        
        # Cargar modelo de embeddings
        if embedding_model is None:
            self.embed_model = EmbeddingModel()
        else:
            self.embed_model = embedding_model
        
        print(f"   ✅ FAISS: {self.vector_db.count()} vectores")
        print(f"   ✅ SQLite: {self.metadata_db.count_chunks()} chunks")
    
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
        chunks = self.metadata_db.get_chunks_by_ids(chunk_ids)
        
        # Paso 4: Crear objetos RetrievedChunk con scores
        retrieved_chunks = []
        for chunk, distance in zip(chunks, distances):
            # Convertir distancia euclidiana a "score de similitud"
            # Distancia baja = alta similitud
            # Usamos 1/(1+distance) para normalizar entre 0 y 1
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
