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
from src.rag.core.logger import get_logger

logger = get_logger(__name__)


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
        self._id_to_index: Dict[str, int] = {}

        self._initialize_db()
        self._ensure_schema() # Migracion
        self._load_vectors_to_memory()

        logger.info("UnifiedStore inicializado (SQLite-only + SQL Metadata)")
        logger.info(f"  BD: {self.db_path}")
        logger.info(f"  Chunks en BD: {self.count()}")
        logger.info(f"  Vectores en cache: {len(self._chunk_ids_cache)}")

    # ----------------------------------------------------------------
    # Inicializacion
    # ----------------------------------------------------------------

    def _initialize_db(self):
        """Crea la tabla unificada si no existe (Schema v2)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id    TEXT PRIMARY KEY,
                    doc_id      TEXT NOT NULL,
                    text        TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    metadata    TEXT,
                    embedding   BLOB,
                    company     TEXT,
                    year        INTEGER,
                    quarter     INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id)")
            # Indices de metadatos movidos a _ensure_schema para soportar migraciones

    def _ensure_schema(self):
        """Migracion automatica: añade columnas e indices si faltan."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(chunks)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            
            new_cols = {
                "company": "TEXT",
                "year": "INTEGER",
                "quarter": "INTEGER"
            }
            
            migrated = False
            for col, dtype in new_cols.items():
                if col not in existing_cols:
                    logger.info(f"Migrating Schema: Adding column {col}")
                    cursor.execute(f"ALTER TABLE chunks ADD COLUMN {col} {dtype}")
                    migrated = True

            # Asegurar indices (siempre, tras asegurar columnas)
            # Usamos try-except para evitar errores si el driver no soporta IF NOT EXISTS (raro)
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_company ON chunks(company)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_year ON chunks(year)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_quarter ON chunks(quarter)")
            except sqlite3.OperationalError:
                # Si falla por "already exists" a pesar de IF NOT EXISTS, ignoramos (aunque no deberia)
                logger.warning("Index creation warning (already exists?)")

            conn.commit()
            
            if migrated:
                self._migrate_metadata(conn)
            else:
                cursor.execute("SELECT count(*) FROM chunks WHERE company IS NULL")
                if cursor.fetchone()[0] > 0:
                    self._migrate_metadata(conn)

    def _migrate_metadata(self, conn):
        """Extrae metadata del JSON y la guarda en columnas SQL."""
        logger.info("Starting Metadata Migration (JSON -> SQL Columns)...")
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id, metadata FROM chunks WHERE company IS NULL")
        
        batch_size = 5000
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
                
            updates = []
            for cid, meta_raw in rows:
                if not meta_raw: continue
                try:
                    meta = json.loads(meta_raw)
                    company = meta.get("company")
                    year = int(meta.get("year")) if meta.get("year") else None
                    q_val = meta.get("quarter")
                    quarter = None
                    if q_val:
                        digits = "".join([c for c in str(q_val) if c.isdigit()])
                        if digits:
                            quarter = int(digits)
                    updates.append((company, year, quarter, cid))
                except:
                    continue
            
            if updates:
                cursor.executemany("UPDATE chunks SET company=?, year=?, quarter=? WHERE chunk_id=?", updates)
                conn.commit()
                
        logger.info("Metadata Migration Completed.")

    def _load_vectors_to_memory(self):
        """Carga vectores y construye indices."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL"
            )
            rows = cursor.fetchall()

        if not rows:
            self._chunk_ids_cache = []
            self._embeddings_cache = None
            self._id_to_index = {}
            return

        ids = []
        vectors = []
        for chunk_id, emb_blob in rows:
            ids.append(chunk_id)
            vectors.append(np.frombuffer(emb_blob, dtype=np.float32))

        self._chunk_ids_cache = ids
        self._id_to_index = {cid: idx for idx, cid in enumerate(ids)}
        
        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._embeddings_cache = matrix / norms

    # ----------------------------------------------------------------
    # Escritura
    # ----------------------------------------------------------------

    def add_documents(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Inserta chunks, vectores y columnas SQL."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Dim mismatch: {len(chunks)} vs {len(embeddings)}")

        embeddings = embeddings.astype(np.float32)

        data = []
        for chunk, emb in zip(chunks, embeddings):
            meta = chunk.metadata or {}
            company = meta.get("company")
            year = int(meta.get("year")) if meta.get("year") else None
            q_val = meta.get("quarter")
            quarter = None
            if q_val:
                digits = "".join([c for c in str(q_val) if c.isdigit()])
                if digits:
                    quarter = int(digits)

            data.append((
                chunk.chunk_id,
                chunk.doc_id,
                chunk.text,
                chunk.chunk_index,
                json.dumps(meta, ensure_ascii=False),
                emb.tobytes(),
                company,
                year,
                quarter
            ))

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks
                    (chunk_id, doc_id, text, chunk_index, metadata, embedding, company, year, quarter)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()

        # Update cache
        start_idx = len(self._chunk_ids_cache)
        new_ids = [c.chunk_id for c in chunks]
        self._chunk_ids_cache.extend(new_ids)
        for i, cid in enumerate(new_ids):
            self._id_to_index[cid] = start_idx + i

        if self._embeddings_cache is None:
            self._embeddings_cache = embeddings
        else:
            self._embeddings_cache = np.vstack([self._embeddings_cache, embeddings])

    def save(self):
        """SQLite hace commit automatico. Este metodo solo recarga la cache."""
        self._load_vectors_to_memory()
        logger.info(f"UnifiedStore guardado. {self.count()} chunks, {len(self._chunk_ids_cache)} vectores.")

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

        1. Calcula cosine similarity (dot product sobre vectores normalizados).
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

        # --- 1. Pre-filtrado SQL (Opcional) ---
        filtered_indices = None
        
        # Si hay filtros, reducimos el espacio de busqueda ANTES de calcular distancias
        if filter_company or filter_year or filter_quarter:
            sql_query = "SELECT chunk_id FROM chunks WHERE 1=1"
            params = []
            
            if filter_company:
                target_ticker = TICKER_MAP.get(filter_company.upper(), filter_company.upper())
                sql_query += " AND company LIKE ?"
                params.append(f"%{target_ticker}%")
            
            if filter_year:
                sql_query += " AND year = ?"
                params.append(filter_year)
                
            if filter_quarter:
                sql_query += " AND quarter = ?"
                params.append(filter_quarter)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(sql_query, params)
                allowed_ids = {row[0] for row in cursor.fetchall()}
            
            # Convertir IDs a indices de cache
            # interseccion rapida usando set
            filtered_indices = [
                self._id_to_index[cid] 
                for cid in allowed_ids 
                if cid in self._id_to_index
            ]
            
            if not filtered_indices:
                return [] 

        # --- 2. Busqueda Vectorial ---
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        qnorm = np.linalg.norm(query_vector)
        if qnorm > 0:
            query_vector = query_vector / qnorm

        # Si hay filtro, usamos numpy fancy indexing sobre subset
        if filtered_indices is not None:
            subset_vectors = self._embeddings_cache[filtered_indices]
            scores = (subset_vectors @ query_vector.T).flatten()
            
            fetch_k = min(top_k, len(scores))
            # Argpartition sobre scores subset
            top_local = np.argpartition(-scores, fetch_k)[:fetch_k]
            top_local = top_local[np.argsort(-scores[top_local])]
            
            # Recuperar IDs globales
            candidate_ids = [self._chunk_ids_cache[filtered_indices[i]] for i in top_local]
            candidate_scores = [float(scores[i]) for i in top_local]
            
        else:
            # Busqueda global (sin pre-filtro)
            scores = (self._embeddings_cache @ query_vector.T).flatten()
            
            # Candidatos extra si no habia filtro PRE pero queremos filtro POST (doble seguridad)
            # Aunque con pre-filtro ya no hace falta fetch_k gigante por crowding out
            fetch_k = min(top_k * 2, len(scores))
            
            top_indices = np.argpartition(-scores, fetch_k)[:fetch_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
            
            candidate_ids = [self._chunk_ids_cache[i] for i in top_indices]
            candidate_scores = [float(scores[i]) for i in top_indices]

        # --- 3. Recuperar Metadatos y Post-Procesado ---
        chunks_data = self._get_chunks_by_ids(candidate_ids)
        chunk_map = {c["chunk_id"]: c for c in chunks_data}

        results = []
        for cid, score in zip(candidate_ids, candidate_scores):
            if cid not in chunk_map: 
                continue
                
            row = chunk_map[cid]
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            
            # Boosting temporal
            final_score = score
            if query_text:
                query_upper = query_text.upper()
                c_year = str(meta.get("year", ""))
                c_q = str(meta.get("quarter", ""))
                if c_year and c_year in query_upper:
                    final_score += 0.05
                if c_q and c_q in query_upper:
                    final_score += 0.03

            results.append(RetrievedChunk(
                chunk_id=cid,
                text=row["text"],
                score=final_score,
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
        """Devuelve la lista de empresas unicas usando SQL directo."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT company FROM chunks WHERE company IS NOT NULL ORDER BY company")
            return [row[0] for row in cursor.fetchall()]

    def count(self) -> int:
        """Numero total de chunks en la BD."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]
