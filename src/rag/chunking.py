import hashlib
from typing import List

from src.rag.schema import Document, Chunk
from src.rag.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: List[Document], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """
    Corta una lista de documentos en trozos (chunks) más pequeños.
    
    Args:
        documents: Lista de objetos Document
        chunk_size: Tamaño de cada trozo en caracteres (por defecto: 800)
        overlap: Solape entre trozos para no cortar frases (por defecto: 100)
        
    Returns:
        Lista de objetos Chunk
    """
    all_chunks = []
    
    for doc in documents:
        text = doc.text
        doc_id = doc.doc_id
        
        # Si el documento es muy corto, no hace falta cortarlo
        if len(text) <= chunk_size:
            chunk_id = _generate_chunk_id(doc_id, 0, text)
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=text,
                chunk_index=0,
                metadata=doc.metadata.copy()
            )
            all_chunks.append(chunk)
            continue
        
        # Cortar el documento en trozos con solape
        chunk_index = 0
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Generar ID único para este trozo
            chunk_id = _generate_chunk_id(doc_id, chunk_index, chunk_text)
            
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                chunk_index=chunk_index,
                metadata=doc.metadata.copy()
            )
            
            all_chunks.append(chunk)
            
            # Avanzar, pero con solape
            start += (chunk_size - overlap)
            chunk_index += 1
    
    return all_chunks


def _generate_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """
    Genera un ID único y estable para un chunk.
    Usa hash del texto para que sea reproducible.
    """
    # Hash del contenido para que sea único
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f"{doc_id}_chunk_{chunk_index}_{text_hash}"
