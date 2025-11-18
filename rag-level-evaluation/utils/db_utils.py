"""
Adatbázis utility függvények PostgreSQL + pgvector használatához.

Ez a modul biztosítja az adatbázis kapcsolatot és a chunk műveletek
(lekérdezés, vector similarity search) funkcióit.
"""
import psycopg
from psycopg.rows import dict_row
import json
from typing import List, Dict, Any, Optional
import sys
import os

# Szülő könyvtár hozzáadása a path-hoz, hogy importálni tudjuk a config-ot
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DB_CONFIG, DEFAULT_MATCH_COUNT, DEFAULT_MATCH_THRESHOLD


def get_db_connection():
    """
    Adatbázis kapcsolat létrehozása és visszaadása.

    Ez a függvény használja a config.py-ban definiált DB_CONFIG dictionary-t
    a kapcsolat felépítéséhez. A row_factory=dict_row beállítással minden
    sor dictionary-ként jön vissza, ami kényelmes a feldolgozáshoz.

    Returns:
        psycopg.Connection: Adatbázis kapcsolat objektum

    Raises:
        Exception: Ha nem sikerül kapcsolódni az adatbázishoz

    Példa:
        >>> conn = get_db_connection()
        >>> # Használat után:
        >>> conn.close()
    """
    try:
        conn = psycopg.connect(
            host=DB_CONFIG['host'],        # PostgreSQL szerver címe (pl. localhost)
            port=DB_CONFIG['port'],        # Port szám (általában 5432)
            dbname=DB_CONFIG['dbname'],    # Adatbázis neve
            user=DB_CONFIG['user'],        # Felhasználónév
            password=DB_CONFIG['password'],# Jelszó
            row_factory=dict_row           # Eredmények dict formátumban
        )
        return conn
    except Exception as e:
        print(f"Hiba az adatbázishoz való csatlakozás során: {e}")
        raise


def fetch_all_chunks(conn) -> List[Dict[str, Any]]:
    """
    Összes dokumentum chunk lekérése az adatbázisból.

    Ez a függvény lekéri az összes chunk-ot a document_chunks táblából,
    document_id és chunk_index szerint rendezve. Így a chunkokk a dokumentumon
    belüli sorrendjükben jelennek meg.

    Args:
        conn: Adatbázis kapcsolat objektum (get_db_connection()-től)

    Returns:
        Lista, amely dictionary-ket tartalmaz. Minden dictionary egy chunk adatait
        tartalmazza:
        - id: UUID, chunk egyedi azonosítója
        - document_id: UUID, a szülő dokumentum azonosítója
        - chunk_index: int, chunk pozíciója a dokumentumban (0-tól indexelt)
        - content: str, chunk szöveges tartalma
        - token_count: int, becsült token szám
        - metadata: dict/JSON, metaadatok (pl. chunking_strategy)
        - created_at: timestamp, létrehozás időpontja

    Raises:
        Exception: Ha hiba történik a lekérdezés során

    Példa:
        >>> conn = get_db_connection()
        >>> chunks = fetch_all_chunks(conn)
        >>> print(f"Összesen {len(chunks)} chunk található")
        >>> conn.close()
    """
    try:
        with conn.cursor() as cur:
            # SQL lekérdezés: minden mező lekérése a document_chunks táblából
            cur.execute("""
                SELECT
                    id,              -- Chunk egyedi azonosítója (UUID)
                    document_id,     -- Szülő dokumentum ID-ja
                    chunk_index,     -- Chunk pozíciója a dokumentumban
                    content,         -- Chunk szöveges tartalma
                    token_count,     -- Token szám (becsült)
                    metadata,        -- JSONB metaadatok
                    created_at       -- Létrehozás időpontja
                FROM document_chunks
                ORDER BY document_id, chunk_index  -- Rendezés dokumentumonként, majd indexenként
            """)
            chunks = cur.fetchall()  # Összes eredmény lekérése listába
            return chunks
    except Exception as e:
        print(f"Hiba a chunkök lekérdezése során: {e}")
        raise


def search_similar_chunks(
    conn,
    query_embedding: List[float],
    limit: int = DEFAULT_MATCH_COUNT,
    threshold: float = DEFAULT_MATCH_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Hasonló chunkök keresése vector similarity alapján.

    Ez a függvény a pgvector <=> operátort használja cosine distance számításhoz.
    A similarity score 1 - cosine_distance képlettel számítódik, így:
    - 1.0 = tökéletes egyezés (azonos vektorok)
    - 0.0 = teljesen eltérő vektorok

    A WHERE feltétel kiszűri a threshold alatti találatokat, az ORDER BY pedig
    a legközelebbi vektorokat hozza elöre (ascending cosine distance).

    Args:
        conn: Adatbázis kapcsolat objektum
        query_embedding: Kérdés embedding vektora (1536 dimenziós lista float-okkal)
        limit: Maximum hány találatot adjunk vissza (default: 5)
        threshold: Minimum similarity threshold (0.0-1.0, default: 0.3)
                   Csak ennél magasabb similarity-vel rendelkező chunkokat ad vissza

    Returns:
        Lista dictionary-kkel, minden elemben:
        - id: UUID, chunk azonosítója
        - document_id: UUID, szülő dokumentum
        - chunk_index: int, pozíció
        - content: str, chunk tartalma
        - similarity: float, hasonlósági érték (0.0-1.0)
        - token_count: int, token szám
        - metadata: dict, metaadatok

        A lista similarity szerint van rendezve, csökkenő sorrendben
        (legjobb találat először).

    Raises:
        Exception: Ha hiba történik a keresés során

    Megjegyzés:
        A pgvector <=> operátor COSINE DISTANCE-t számít, nem similarity-t!
        Ezért a similarity = 1 - distance formula szükséges.

        Példa similarity értékek:
        - 0.9-1.0: Nagyon hasonló (kiváló találat)
        - 0.7-0.9: Közepesen hasonló (jó találat)
        - 0.5-0.7: Gyengén hasonló (elfogadható)
        - 0.3-0.5: Alig hasonló (boundary)
        - <0.3: Irreleváns (kiszűrve threshold-dal)

    Példa:
        >>> from utils.openai_utils import generate_embedding
        >>> embedding = generate_embedding("Who is Mowgli?")
        >>> results = search_similar_chunks(conn, embedding, limit=5, threshold=0.3)
        >>> for r in results:
        >>>     print(f"Similarity: {r['similarity']:.4f} - {r['content'][:50]}...")
    """
    try:
        # Embedding vektor konvertálása JSON string-gé PostgreSQL számára
        # Példa: [0.1, 0.2, ..., 0.5] -> "[0.1, 0.2, ..., 0.5]"
        embedding_json = json.dumps(query_embedding)

        with conn.cursor() as cur:
            # Vector similarity search SQL lekérdezés
            cur.execute("""
                SELECT
                    id,
                    document_id,
                    chunk_index,
                    content,
                    1 - (embedding <=> %s::vector) AS similarity,  -- Cosine similarity számítás
                    token_count,
                    metadata
                FROM document_chunks
                WHERE 1 - (embedding <=> %s::vector) >= %s  -- Threshold szűrés
                ORDER BY embedding <=> %s::vector           -- Rendezés distance szerint (asc)
                LIMIT %s                                    -- Maximum N találat
            """, (embedding_json, embedding_json, threshold, embedding_json, limit))
            # Megjegyzés: embedding_json 4-szer szerepel, mert 4 helyen használjuk a query-ben

            results = cur.fetchall()  # Összes találat lekérése
            return results
    except Exception as e:
        print(f"Hiba a hasonló chunkök keresésekor: {e}")
        raise


def get_chunk_by_id(conn, chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    Egy konkrét chunk lekérése ID alapján.

    Hasznos, ha már ismerjük egy chunk ID-ját és le szeretnénk kérni
    a részleteit.

    Args:
        conn: Adatbázis kapcsolat objektum
        chunk_id: A chunk UUID azonosítója (string formátumban)

    Returns:
        Dictionary a chunk adataival, vagy None ha nem található.
        Dictionary tartalma:
        - id: UUID
        - document_id: UUID
        - chunk_index: int
        - content: str
        - token_count: int
        - metadata: dict
        - created_at: timestamp

    Raises:
        Exception: Ha hiba történik a lekérdezés során

    Példa:
        >>> chunk = get_chunk_by_id(conn, "123e4567-e89b-12d3-a456-426614174000")
        >>> if chunk:
        >>>     print(f"Chunk tartalma: {chunk['content']}")
        >>> else:
        >>>     print("Chunk nem található")
    """
    try:
        with conn.cursor() as cur:
            # Egyszerű SELECT WHERE id = ... lekérdezés
            cur.execute("""
                SELECT
                    id,
                    document_id,
                    chunk_index,
                    content,
                    token_count,
                    metadata,
                    created_at
                FROM document_chunks
                WHERE id = %s  -- ID alapú szűrés
            """, (chunk_id,))

            result = cur.fetchone()  # Csak egy eredmény várható (vagy None)
            return result
    except Exception as e:
        print(f"Hiba a chunk ID alapú lekérdezésekor: {e}")
        raise
