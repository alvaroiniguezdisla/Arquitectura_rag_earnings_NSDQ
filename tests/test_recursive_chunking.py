
import pytest
from src.rag.core.schema import Document
from src.rag.pipeline.step2_chunking import chunk_documents, _recursive_split

class TestRecursiveChunking:
    
    def test_chunk_small_document(self):
        """Documentos pequeños no se deben cortar."""
        doc = Document(doc_id="doc1", text="Hola mundo. Esto es una prueba.", metadata={})
        chunks = chunk_documents([doc], chunk_size=100, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0].text == "Hola mundo. Esto es una prueba."

    def test_chunk_split_by_paragraphs(self):
        """Debe cortar por parrafos si es posible."""
        text = "Parrafo 1.\n\nParrafo 2.\n\nParrafo 3."
        # Chunk size pequeño para forzar split
        chunks = _recursive_split(text, chunk_size=15, overlap=0, separators=["\n\n", "\n", " "])
        
        # Deberia separar "Parrafo 1.", "Parrafo 2.", "Parrafo 3."
        assert len(chunks) == 3
        assert "Parrafo 1." in chunks[0]
        assert "Parrafo 2." in chunks[1]

    def test_chunk_split_by_sentences(self):
        """Debe cortar por frases si los parrafos son muy largos."""
        text = "Frase uno. Frase dos. Frase tres."
        # Chunk size para que quepa 1 frase pero no 2
        chunks = _recursive_split(text, chunk_size=12, overlap=0, separators=[". ", " "])
        
        # Nota: el separador ". " se pierde o se gestiona dependiendo de la logica.
        # En nuestra implementación simple con split(), se pierde el separador intermedio.
        assert len(chunks) >= 3

    def test_overlap_logic(self):
        """Verificar que el solape existe."""
        text = "ABCDEFGHIJ" 
        # Chunk 4, overlap 2
        # Expected: ABCD, CDEF, EFGH, GHIJ
        chunks = _recursive_split(text, chunk_size=4, overlap=2, separators=[""])
        
        assert len(chunks) > 1
        assert chunks[0] == "ABCD"
        # Overlap simple vs recursivo puede variar, pero debe haber solape
        assert "CD" in chunks[1] 

    def test_empty_document(self):
        """Documento vacio no rompe nada."""
        doc = Document(doc_id="doc_empty", text="", metadata={})
        chunks = chunk_documents([doc])
        assert len(chunks) == 0
