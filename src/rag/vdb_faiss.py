import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pickle

from src.rag.config import FAISS_INDEX_PATH


class VectorDB:
    """
    Gestiona el índice FAISS para búsqueda vectorial.
    """
    
    def __init__(self, index_path: Path = FAISS_INDEX_PATH, dimension: int = 384):
        """
        Inicializa o carga el índice FAISS.
        
        Args:
            index_path: Ruta al archivo del índice
            dimension: Dimensión de los vectores (384 para MiniLM)
        """
        self.index_path = index_path
        self.dimension = dimension
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Mapeo de posición en FAISS -> chunk_id
        self.id_map_path = self.index_path.with_suffix('.id_map.pkl')
        
        if self.index_path.exists():
            self.load_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Crea un nuevo índice FAISS vacío."""
        # IndexFlatL2: búsqueda exacta por distancia L2 (euclidiana)
        # Simple y efectivo para datasets pequeños-medianos
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunk_ids = []  # Lista que mapea posición -> chunk_id
        print(f"✅ Creado nuevo índice FAISS (dimensión={self.dimension})")
    
    def add_vectors(self, vectors: np.ndarray, chunk_ids: List[str]):
        """
        Añade vectores al índice.
        
        Args:
            vectors: Array numpy de shape (n, dimension)
            chunk_ids: Lista de IDs correspondientes a cada vector
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Dimensión incorrecta: {vectors.shape[1]} != {self.dimension}")
        
        # Convertir a float32 (requisito de FAISS)
        vectors = vectors.astype('float32')
        
        # Añadir al índice
        self.index.add(vectors)
        
        # Actualizar el mapeo de IDs
        self.chunk_ids.extend(chunk_ids)
    
    def search(self, query_vector: np.ndarray, top_k: int = 8) -> Tuple[List[str], List[float]]:
        """
        Busca los vectores más cercanos al query.
        
        Args:
            query_vector: Vector de consulta (dimension,)
            top_k: Número de resultados a retornar
            
        Returns:
            (chunk_ids, distancias)
        """
        # Asegurar shape correcto
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype('float32')
        
        # Buscar en FAISS
        distances, indices = self.index.search(query_vector, top_k)
        
        # Convertir índices a chunk_ids
        result_ids = [self.chunk_ids[i] for i in indices[0] if i < len(self.chunk_ids)]
        result_distances = distances[0].tolist()
        
        return result_ids, result_distances
    
    def save_index(self):
        """Guarda el índice FAISS y el mapeo de IDs en disco."""
        # Guardar índice FAISS
        faiss.write_index(self.index, str(self.index_path))
        
        # Guardar mapeo de IDs
        with open(self.id_map_path, 'wb') as f:
            pickle.dump(self.chunk_ids, f)
        
        print(f"✅ Índice guardado en: {self.index_path}")
        print(f"   - Vectores: {self.index.ntotal}")
        print(f"   - IDs mapeados: {len(self.chunk_ids)}")
    
    def load_index(self):
        """Carga el índice FAISS y el mapeo de IDs desde disco."""
        self.index = faiss.read_index(str(self.index_path))
        
        with open(self.id_map_path, 'rb') as f:
            self.chunk_ids = pickle.load(f)
        
        print(f"✅ Índice cargado desde: {self.index_path}")
        print(f"   - Vectores: {self.index.ntotal}")
        print(f"   - IDs mapeados: {len(self.chunk_ids)}")
    
    def count(self) -> int:
        """Retorna el número de vectores en el índice."""
        return self.index.ntotal
