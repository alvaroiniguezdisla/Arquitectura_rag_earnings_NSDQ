from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class Document:
    """
    Representa el documento original (todo el transcript).
    - doc_id: Identificador único (el nombre del archivo o hash).
    - text: El texto completo limpio.
    - metadata: Año, empresa, trimestre...
    """
    doc_id: str
    text: str
    metadata: Dict = field(default_factory=dict)
    
    @property
    def source(self) -> str:
        # Atajo para saber de dónde viene
        return self.metadata.get("source_path", "unknown")

@dataclass
class Chunk:
    """
    Representa un trozo o 'bocado' del documento.
    - chunk_id: ID único del trozo.
    - doc_id: ID del documento al que pertenece (padre).
    - text: El texto de este trozo (ej: 800 caracteres).
    - chunk_index: El número de orden (trozo 1, 2, 3...).
    """
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    metadata: Dict = field(default_factory=dict)

@dataclass
class RetrievedChunk:
    """
    Lo que te devuelve el buscador cuando preguntas.
    Es igual que un Chunk pero con 'score' (nota de relevancia).
    """
    chunk_id: str
    text: str
    score: float
    doc_id: str
    metadata: Dict = field(default_factory=dict)
