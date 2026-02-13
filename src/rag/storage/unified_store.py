"""
UnifiedDocumentStore - Base de datos unica SQLite.

Cada chunk se almacena con su vector (embedding) en la misma fila.
Sin FAISS. Sin archivos separados. Un solo .db.

Tabla 'chunks':
  chunk_id    TEXT PRIMARY KEY
  doc_id      TEXT
  text        TEXT
  chunk_index INTEGER
  metadata    TEXT (JSON)
  embedding   BLOB (numpy array serializado)
"""
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import re

from src.rag.core.schema import Chunk, RetrievedChunk
from src.rag.core.config import SQLITE_DB_PATH, EMBEDDING_DIMENSION, TICKER_MAP


class UnifiedDocumentStore:
    """
    Base de datos unificada: texto + metadata + vectores en SQLite.
    Los vectores se cachean en memoria (numpy) para busqueda rapida.
    """

    def __init__(self, db_path: Path = SQLITE_DB_PATH, dimension: int = EMBEDDING_DIMENSION):
        self.db_path = db_path
        self.dimension = dimension
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Cache en memoria para busqueda vectorial
        self._chunk_ids_cache: List[str] = []
        self._embeddings_cache: Optional[np.ndarray] = None  # shape (N, dimension)

        self._init_db()
        self._load_vectors_to_memory()

        print(f">> UnifiedStore inicializado (SQLite-only)")
        print(f"   - BD: {self.db_path}")
        print(f"   - Chunks en BD: {self.count()}")
        print(f"   - Vectores en cache: {len(self._chunk_ids_cache)}")

    # ----------------------------------------------------------------
    # Inicializacion
    # ----------------------------------------------------------------

    def _init_db(self):
        """Crea la tabla unificada si no existe."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id    TEXT PRIMARY KEY,
                    doc_id      TEXT NOT NULL,
                    text        TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    metadata    TEXT,
                    embedding   BLOB
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id)")
            conn.commit()

    def _load_vectors_to_memory(self):
        """Carga todos los vectores de SQLite a un array numpy en RAM."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL"
            )
            rows = cursor.fetchall()

        if not rows:
            self._chunk_ids_cache = []
            self._embeddings_cache = None
            return

        ids = []
        vectors = []
        for chunk_id, emb_blob in rows:
            ids.append(chunk_id)
            vectors.append(np.frombuffer(emb_blob, dtype=np.float32))

        self._chunk_ids_cache = ids
        self._embeddings_cache = np.vstack(vectors)  # shape (N, dim)

    # ----------------------------------------------------------------
    # Escritura
    # ----------------------------------------------------------------

    def add_documents(self, chunks: List[Chunk], embeddings: np.ndarray):
        """
        Inserta chunks con sus vectores en la BD unificada.

        Args:
            chunks: Lista de Chunk (texto + metadata).
            embeddings: Matriz numpy (N, dimension) con los vectores.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Desajuste: {len(chunks)} chunks vs {len(embeddings)} vectores")

        embeddings = embeddings.astype(np.float32)

        rows = []
        for chunk, emb in zip(chunks, embeddings):
            rows.append((
                chunk.chunk_id,
                chunk.doc_id,
                chunk.text,
                chunk.chunk_index,
                json.dumps(chunk.metadata, ensure_ascii=False),
                emb.tobytes()  # Vector -> BLOB
            ))

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks
                    (chunk_id, doc_id, text, chunk_index, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

        # Actualizar cache en memoria (append)
        new_ids = [c.chunk_id for c in chunks]
        self._chunk_ids_cache.extend(new_ids)

        if self._embeddings_cache is None:
            self._embeddings_cache = embeddings
        else:
            self._embeddings_cache = np.vstack([self._embeddings_cache, embeddings])

    def save(self):
        """SQLite hace commit automatico. Este metodo solo recarga la cache."""
        self._load_vectors_to_memory()
        print(f"OK UnifiedStore guardado. {self.count()} chunks, {len(self._chunk_ids_cache)} vectores.")

    # ----------------------------------------------------------------
    # Busqueda
    # ----------------------------------------------------------------

    def search(self,
               query_vector: np.ndarray,
               top_k: int = 10,
               filter_company: Optional[str] = None,
               filter_year: Optional[int] = None,
               filter_quarter: Optional[int] = None,
               query_text: Optional[str] = None) -> List[RetrievedChunk]:
        """
        Busca los chunks mas similares al vector de consulta.

        1. Calcula distancias L2 contra todos los vectores en cache.
        2. Recupera metadata de SQLite para los top candidatos.
        3. Aplica filtros (empresa, año, trimestre) y boosting.

        Args:
            query_vector: Vector de la consulta (dimension,).
            top_k: Numero de resultados finales.
            filter_company: Ticker o nombre para filtrar.
            filter_year: Año (int).
            filter_quarter: Trimestre (int).
            query_text: Texto original de la query (para boosting temporal).

        Returns:
            Lista de RetrievedChunk ordenados por score.
        """
        if self._embeddings_cache is None or len(self._chunk_ids_cache) == 0:
            return []

        # --- 1. Busqueda Vectorial (numpy) ---
        query_vector = query_vector.astype(np.float32).reshape(1, -1)

        # Distancia L2 al cuadrado contra todos los vectores
        diffs = self._embeddings_cache - query_vector
        distances = np.sum(diffs ** 2, axis=1)

        # Candidatos: Si hay filtros estrictos, traemos MUCHOS mas para asegurar que queden suficientes tras filtrar
        multiplier = 4
        if filter_company or filter_year:
            multiplier = 500 # Aumentamos drásticamente para evitar "crowding out" en datasets pequeños/medianos
            
        fetch_k = top_k * multiplier
        fetch_k = min(fetch_k, len(distances))
        
        top_indices = np.argpartition(distances, fetch_k)[:fetch_k]
        top_indices = top_indices[np.argsort(distances[top_indices])]

        # --- 2. Recuperar datos de SQLite ---
        candidate_ids = [self._chunk_ids_cache[i] for i in top_indices]
        candidate_distances = [float(distances[i]) for i in top_indices]

        chunks_data = self._get_chunks_by_ids(candidate_ids)
        chunk_map = {c["chunk_id"]: c for c in chunks_data}

        # --- 3. Filtrado y Ranking ---
        target_ticker = None
        if filter_company:
            target_ticker = TICKER_MAP.get(filter_company.upper(), filter_company.upper())

        results = []
        for cid, dist in zip(candidate_ids, candidate_distances):
            if cid not in chunk_map:
                continue

            row = chunk_map[cid]
            meta = json.loads(row["metadata"]) if row["metadata"] else {}

            # Filtro por empresa
            if target_ticker:
                chunk_company = str(meta.get("company", "")).upper()
                if target_ticker not in chunk_company and chunk_company not in target_ticker:
                    continue
            
            # Filtro por Año
            if filter_year:
                # Metadata suele ser string o int. Normalizamos a int.
                try:
                    meta_year = int(meta.get("year", -1))
                    if meta_year != filter_year:
                        continue
                except (ValueError, TypeError):
                    continue # Si no tiene año valido y pedimos filtro, lo descartamos
            
            # Filtro por Trimestre
            if filter_quarter:
                meta_q_val = str(meta.get("quarter", ""))
                # Extraer digitos: "Q3" -> "3", "3" -> "3"
                digits = "".join([c for c in meta_q_val if c.isdigit()])
                
                if not digits:
                    continue # No se pudo determinar trimestre -> Descartamos

                try:
                    if int(digits) != filter_quarter:
                        continue
                except ValueError:
                    continue

            # Score base (distancia -> similitud)
            base_score = 1.0 / (1.0 + dist)

            # Boosting temporal
            boost = 0.0
            if query_text:
                query_upper = query_text.upper()
                chunk_year = str(meta.get("year", ""))
                chunk_q = str(meta.get("quarter", ""))
                if chunk_year and chunk_year in query_upper:
                    boost += 0.05
                if chunk_q and chunk_q in query_upper:
                    boost += 0.03

            results.append(RetrievedChunk(
                chunk_id=cid,
                text=row["text"],
                score=base_score + boost,
                doc_id=row["doc_id"],
                metadata=meta
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    # ----------------------------------------------------------------
    # Utilidades
    # ----------------------------------------------------------------

    def _get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict]:
        """Recupera filas de SQLite por IDs (sin embedding para no gastar RAM)."""
        if not chunk_ids:
            return []

        placeholders = ",".join("?" * len(chunk_ids))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                f"SELECT chunk_id, doc_id, text, chunk_index, metadata FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_companies(self) -> List[str]:
        """Devuelve la lista de empresas unicas en la BD."""
        companies = set()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT metadata FROM chunks")
            for (meta_json,) in cursor.fetchall():
                try:
                    meta = json.loads(meta_json)
                    company = meta.get("company")
                    if company:
                        companies.add(company)
                except (json.JSONDecodeError, TypeError):
                    continue
        return sorted(companies)

    def count(self) -> int:
        """Numero total de chunks en la BD."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]
