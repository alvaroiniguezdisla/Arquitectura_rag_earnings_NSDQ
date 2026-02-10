import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
import json

from src.rag.schema import Chunk
from src.rag.config import SQLITE_DB_PATH


class MetadataDB:
    """
    Gestiona la base de datos SQLite para guardar texto y metadata de chunks.
    """
    
    def __init__(self, db_path: Path = SQLITE_DB_PATH):
        """
        Inicializa la base de datos SQLite.
        
        Args:
            db_path: Ruta al archivo .db
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Crea las tablas si no existen."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id)
            """)
            conn.commit()
    
    def upsert_chunk(self, chunk: Chunk):
        """
        Inserta o actualiza un chunk en la base de datos.
        
        Args:
            chunk: Objeto Chunk a guardar
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, chunk_index, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.text,
                chunk.chunk_index,
                json.dumps(chunk.metadata)
            ))
            conn.commit()
    
    def upsert_chunks(self, chunks: List[Chunk]):
        """
        Inserta o actualiza múltiples chunks (batch).
        
        Args:
            chunks: Lista de objetos Chunk
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, chunk_index, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, [
                (c.chunk_id, c.doc_id, c.text, c.chunk_index, json.dumps(c.metadata))
                for c in chunks
            ])
            conn.commit()
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """
        Recupera un chunk por su ID.
        
        Args:
            chunk_id: ID del chunk
            
        Returns:
            Objeto Chunk o None si no existe
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM chunks WHERE chunk_id = ?
            """, (chunk_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return Chunk(
                chunk_id=row['chunk_id'],
                doc_id=row['doc_id'],
                text=row['text'],
                chunk_index=row['chunk_index'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Recupera múltiples chunks por sus IDs.
        
        Args:
            chunk_ids: Lista de IDs
            
        Returns:
            Lista de objetos Chunk
        """
        if not chunk_ids:
            return []
        
        placeholders = ','.join('?' * len(chunk_ids))
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT * FROM chunks WHERE chunk_id IN ({placeholders})
            """, chunk_ids)
            
            chunks = []
            for row in cursor.fetchall():
                chunks.append(Chunk(
                    chunk_id=row['chunk_id'],
                    doc_id=row['doc_id'],
                    text=row['text'],
                    chunk_index=row['chunk_index'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                ))
            
            return chunks
    
    def count_chunks(self) -> int:
        """Retorna el número total de chunks en la BD."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]
