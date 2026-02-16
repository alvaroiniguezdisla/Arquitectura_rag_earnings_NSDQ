import hashlib
import re
from typing import List, Optional

from src.rag.core.schema import Document, Chunk
from src.rag.core.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.rag.core.logger import get_logger

logger = get_logger(__name__)

def chunk_documents(documents: List[Document], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """
    Corta documentos usando una estrategia recursiva para respetar 
    las fronteras semánticas (párrafos -> líneas -> frases).
    
    Args:
        documents: Lista de objetos Document
        chunk_size: Tamaño objetivo del chunk (caracteres)
        overlap: Solape entre chunks para mantener contexto
        
    Returns:
        Lista de objetos Chunk
    """
    all_chunks = []
    
    # Separadores en orden de preferencia (semántico -> sintáctico)
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    for doc in documents:
        text = doc.text
        if not text:
            continue
            
        # Usamos el splitter recursivo
        text_chunks = _recursive_split(text, chunk_size, overlap, separators)
        
        for i, chunk_text in enumerate(text_chunks):
            # Generar ID único estable (hash del contenido)
            chunk_id = _generate_chunk_id(doc.doc_id, i, chunk_text)
            
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=chunk_text,
                chunk_index=i,
                metadata=doc.metadata.copy()
            )
            all_chunks.append(chunk)
            
    return all_chunks


def _recursive_split(text: str, chunk_size: int, overlap: int, separators: List[str]) -> List[str]:
    """
    Función core del Recursive Character Splitter.
    Intenta dividir por el separador más prioritario. Si el chunk resultante
    sigue siendo muy grande, baja al siguiente nivel de separador.
    """
    final_chunks = []
    
    # Caso base: si ya no hay separadores, cortamos a machete (caracteres)
    if not separators:
        return _split_by_chars(text, chunk_size, overlap)
    
    separator = separators[0]
    next_separators = separators[1:]
    
    # Si el separador es vacio (""), delegamos a caracteres
    if separator == "":
        return _split_by_chars(text, chunk_size, overlap)
        
    # Dividimos por el separador actual
    # Usamos regex escape para evitar problemas con puntos, etc.
    try:
        # Nota: split elimina el separador. Para ". " a veces queremos mantener el punto.
        # Por simplicidad en MVP, asumimos que perder el separador o reconcateneralo es aceptable
        # o usamos un split simple.
        splits = text.split(separator)
    except Exception:
        splits = [text]

    # Ahora reagrupamos los splits en chunks que quepan en chunk_size
    current_chunk = []
    current_length = 0
    
    for split in splits:
        # Si el split por sí mismo es más grande que el chunk_size, 
        # necesitamos dividirlo recursivamente con el siguiente separador
        if len(split) > chunk_size:
            # Si tenemos algo acumulado, lo guardamos antes
            if current_chunk:
                merged_text = separator.join(current_chunk)
                final_chunks.append(merged_text)
                # Reset con overlap (simplificado: mantenemos últimos N chars es complejo sin tokenizar)
                # En esta implementación simple vaciamos buffer. 
                # Mejorar overlap en recursivo es tricky sin librería.
                # Estrategia: Guardar chunk y resetear.
                current_chunk = []
                current_length = 0
            
            # Procesar el trozo gigante recursivamente
            sub_chunks = _recursive_split(split, chunk_size, overlap, next_separators)
            final_chunks.extend(sub_chunks)
            continue
            
        # Si cabe en el chunk actual, lo añadimos
        # Sumamos len(split) + len(separator) porque al unirlo volverá a estar
        if current_length + len(split) + len(separator) <= chunk_size:
            current_chunk.append(split)
            current_length += len(split) + len(separator)
        else:
            # No cabe, cerramos el chunk actual
            if current_chunk:
                merged_text = separator.join(current_chunk)
                final_chunks.append(merged_text)
                
                # --- Manejo de Overlap ---
                # Para mantener contexto, intentamos mantener los últimos elementos 
                # que quepan en el overlap.
                overlap_chunk = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) + len(separator) <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s) + len(separator)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_chunk.append(split)
                current_length = overlap_len + len(split) + len(separator)
            else:
                # Caso raro: el chunk estaba vacío pero el split no cabe (ya manejado arriba, pero por seguridad)
                current_chunk = [split]
                current_length = len(split)

    # Añadir lo que quede en el buffer
    if current_chunk:
        merged_text = separator.join(current_chunk)
        final_chunks.append(merged_text)
        
    return final_chunks

def _split_by_chars(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Corte final por caracteres (hard cut) con solape."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def _generate_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Genera ID único y estable."""
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f"{doc_id}_chunk_{chunk_index}_{text_hash}"
