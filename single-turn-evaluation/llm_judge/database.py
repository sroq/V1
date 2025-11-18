"""
Adatbázis segédeszközök PostgreSQL + pgvector-hoz.

Ez a modul kapcsolódási segédeszközöket és lekérdezés segítőket biztosít
a PostgreSQL adatbázishoz, amely dokumentum chunk-okat és embeddinget tartalmaz.

A rag-level-evaluation projektből újrahaszn\u00e1lt kapcsolati logikát alkalmaz.
"""

import psycopg
from psycopg.rows import dict_row
from typing import List, Dict, Any, Optional
import logging

from config import DB_CONFIG

logger = logging.getLogger(__name__)


def get_db_connection():
    """
    Létrehoz egy PostgreSQL adatbázis kapcsolatot.

    Returns:
        psycopg.Connection: Adatbázis kapcsolat objektum

    Raises:
        psycopg.OperationalError: Ha a kapcsolódás sikertelen

    Példa:
        >>> conn = get_db_connection()
        >>> cursor = conn.cursor()
        >>> cursor.execute("SELECT 1")
    """
    try:
        conn = psycopg.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            dbname=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            row_factory=dict_row  # Eredmények dictionary-ként való visszaadása
        )
        logger.debug(f"Adatbázis kapcsolat létrehozva: {DB_CONFIG['database']}@{DB_CONFIG['host']}")
        return conn
    except psycopg.OperationalError as e:
        logger.error(f"Sikertelen adatbázis kapcsolódás: {e}")
        raise


def fetch_chunk_by_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    Lekér egy chunk-ot ID alapján.

    Args:
        chunk_id: A chunk egyedi azonosítója (UUID)

    Returns:
        Dictionary chunk adatokkal vagy None, ha nem található

    Példa:
        >>> chunk = fetch_chunk_by_id("752db7d8-9a5f-45c6-958b-025b59d39d70")
        >>> print(chunk['content'])
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                id,
                content,
                document_id,
                chunk_index,
                token_count,
                metadata
            FROM document_chunks
            WHERE id = %s
        """, (chunk_id,))

        result = cursor.fetchone()
        cursor.close()
        conn.close()

        return dict(result) if result else None

    except Exception as e:
        logger.error(f"Hiba a chunk lekérésekor {chunk_id}: {e}")
        conn.close()
        raise


def search_similar_chunks(
    query_embedding: List[float],
    top_k: int = 5,
    similarity_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Vektoros keresés a leginkább hasonló chunk-okért.

    Args:
        query_embedding: Query embedding vektor (1536 dim, OpenAI text-embedding-3-small)
        top_k: Hány chunk-ot adjon vissza (alapértelmezett: 5)
        similarity_threshold: Minimum koszinusz hasonlóság (alapértelmezett: 0.0)

    Returns:
        Lista chunk dictionary-kkel, rendezve hasonlósági pontszám szerint (magas → alacsony)

    Példa:
        >>> from openai import OpenAI
        >>> client = OpenAI()
        >>> response = client.embeddings.create(model="text-embedding-3-small", input="Ki az Mowgli?")
        >>> embedding = response.data[0].embedding
        >>> results = search_similar_chunks(embedding, top_k=3)
        >>> for chunk in results:
        ...     print(f"Pontszám: {chunk['similarity_score']:.3f}, Tartalom: {chunk['content'][:50]}...")
    """
    conn = get_db_connection()

    try:
        cursor = conn.cursor()

        # pgvector koszinusz távolság operátor: <=>
        # 1 - cosine_distance = cosine_similarity
        cursor.execute("""
            SELECT
                id,
                content,
                document_id,
                chunk_index,
                token_count,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity_score
            FROM document_chunks
            WHERE 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, similarity_threshold, query_embedding, top_k))

        results = cursor.fetchall()
        cursor.close()
        conn.close()

        return [dict(row) for row in results]

    except Exception as e:
        logger.error(f"Hiba a vektoros keresésben: {e}")
        conn.close()
        raise


def fetch_all_chunks(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Összes chunk lekérése az adatbázisból.

    Args:
        limit: Opcionális limit a visszaadott chunk-ok számára

    Returns:
        Lista chunk dictionary-kkel

    Példa:
        >>> chunks = fetch_all_chunks(limit=10)
        >>> print(f"Lekért {len(chunks)} chunk")
    """
    conn = get_db_connection()

    try:
        cursor = conn.cursor()

        query = """
            SELECT
                id,
                content,
                document_id,
                chunk_index,
                token_count,
                metadata
            FROM document_chunks
            ORDER BY document_id, chunk_index
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        return [dict(row) for row in results]

    except Exception as e:
        logger.error(f"Hiba az összes chunk lekérésekor: {e}")
        conn.close()
        raise


def get_database_stats() -> Dict[str, Any]:
    """
    Adatbázis statisztikák lekérése (chunk count, document count, stb.).

    Returns:
        Dictionary stat információkkal

    Példa:
        >>> stats = get_database_stats()
        >>> print(f"Összes chunk: {stats['total_chunks']}")
        >>> print(f"Összes dokumentum: {stats['total_documents']}")
    """
    conn = get_db_connection()

    try:
        cursor = conn.cursor()

        # Összes chunk
        cursor.execute("SELECT COUNT(*) as count FROM document_chunks")
        total_chunks = cursor.fetchone()['count']

        # Összes dokumentum
        cursor.execute("SELECT COUNT(DISTINCT document_id) as count FROM document_chunks")
        total_documents = cursor.fetchone()['count']

        # Átlagos chunk méret
        cursor.execute("SELECT AVG(token_count) as avg_size FROM document_chunks")
        avg_chunk_size = cursor.fetchone()['avg_size']

        cursor.close()
        conn.close()

        return {
            'total_chunks': total_chunks,
            'total_documents': total_documents,
            'avg_chunk_size': float(avg_chunk_size) if avg_chunk_size else 0.0
        }

    except Exception as e:
        logger.error(f"Hiba az adatbázis statisztikák lekérésekor: {e}")
        conn.close()
        raise


if __name__ == '__main__':
    # Adatbázis kapcsolat és segédeszközök tesztelése
    import sys

    logging.basicConfig(level=logging.INFO)

    try:
        logger.info("Adatbázis kapcsolat tesztelése...")
        conn = get_db_connection()
        conn.close()
        logger.info("✓ Adatbázis kapcsolat sikeres")

        logger.info("\nAdatbázis statisztikák lekérése...")
        stats = get_database_stats()
        logger.info(f"✓ Összes chunk: {stats['total_chunks']}")
        logger.info(f"✓ Összes dokumentum: {stats['total_documents']}")
        logger.info(f"✓ Átlagos chunk méret: {stats['avg_chunk_size']:.1f} token")

        logger.info("\nMinta chunk lekérése...")
        chunks = fetch_all_chunks(limit=1)
        if chunks:
            chunk = chunks[0]
            logger.info(f"✓ Minta chunk ID: {chunk['id']}")
            logger.info(f"✓ Tartalom előnézet: {chunk['content'][:100]}...")
        else:
            logger.warning("⚠ Nem található chunk az adatbázisban")

    except Exception as e:
        logger.error(f"✗ Adatbázis teszt sikertelen: {e}")
        sys.exit(1)
