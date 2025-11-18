"""
RAG Asszisztens Futtat\u00f3 Single-Turn Értékeléshez.

Ez a modul a teljes RAG pipeline-t implementálja:
1. Query embedding generálás (OpenAI)
2. Vektoros keresés (PostgreSQL + pgvector)
3. Kontextus összeállítás
4. Válasz generálás (GPT-4o mini)

Az értékelési pipeline-ban használjuk asszisztens válaszok generálásához.
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
import logging
import time

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    ASSISTANT_MODEL,
    TEMPERATURE_ASSISTANT,
    MAX_TOKENS_ASSISTANT,
    TOP_K_CHUNKS,
    SIMILARITY_THRESHOLD
)
from .database import search_similar_chunks
from .prompts import ASSISTANT_SYSTEM_PROMPT, build_assistant_user_prompt

logger = logging.getLogger(__name__)


class AssistantRunner:
    """
    RAG Asszisztens futtató felhasználói lekérdezésekre válaszok generálásához.

    Ez az osztály a teljes RAG pipeline-t kapszulálja a lekérdezéstől a válaszig,
    beleértve az embedding generálást, vektoros keresést és LLM válasz generálást.

    Attribútumok:
        client: OpenAI API kliens
        embedding_model: Modell neve az embeddinghez
        chat_model: Modell neve a chat completion-höz
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Asszisztens futtató inicializálása.

        Args:
            api_key: OpenAI API kulcs (alapértelmezett: OPENAI_API_KEY a config-ból)
        """
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        self.embedding_model = EMBEDDING_MODEL
        self.chat_model = ASSISTANT_MODEL
        self.temperature = TEMPERATURE_ASSISTANT
        self.max_tokens = MAX_TOKENS_ASSISTANT

        logger.info(f"AssistantRunner inicializálva: {self.chat_model}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Embedding vektor generálása szöveghez OpenAI használatával.

        Args:
            text: Bemeneti szöveg beágyazáshoz

        Returns:
            Float lista (1536 dimenzió text-embedding-3-small-hoz)

        Példa:
            >>> runner = AssistantRunner()
            >>> embedding = runner.generate_embedding("Ki Mowgli?")
            >>> len(embedding)
            1536
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            logger.debug(f"Embedding generálva: {len(embedding)} dimenzió")
            return embedding

        except Exception as e:
            logger.error(f"Hiba az embedding generáláskor: {e}")
            raise

    def search_context(
        self,
        query_embedding: List[float],
        top_k: int = TOP_K_CHUNKS,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Hasonló chunk-ok keresése vektoros keresés használatával.

        Args:
            query_embedding: Query embedding vektor
            top_k: Lekérendő chunk-ok száma
            threshold: Minimum hasonlósági küszöb

        Returns:
            Chunk dictionary-k listája hasonlósági pontszámokkal

        Példa:
            >>> runner = AssistantRunner()
            >>> embedding = runner.generate_embedding("Ki Mowgli?")
            >>> chunks = runner.search_context(embedding, top_k=3)
            >>> for chunk in chunks:
            ...     print(f"Pontszám: {chunk['similarity_score']:.3f}")
        """
        try:
            chunks = search_similar_chunks(
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=threshold
            )
            logger.info(f"Lekért {len(chunks)} chunk (küszöb={threshold})")
            return chunks

        except Exception as e:
            logger.error(f"Hiba a vektoros keresésben: {e}")
            raise

    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Válasz generálása LLM használatával a lekérdezés és kontextus alapján.

        Args:
            query: Felhasználó kérdése
            context_chunks: Lekért kontextus chunk-ok

        Returns:
            Dictionary válasz szöveggel és metaadatokkal

        Példa:
            >>> runner = AssistantRunner()
            >>> chunks = [...]  # Lekért chunk-ok
            >>> result = runner.generate_response("Ki Mowgli?", chunks)
            >>> print(result['response_text'])
        """
        try:
            # Felhasználói prompt építése kontextussal
            user_prompt = build_assistant_user_prompt(query, context_chunks)

            # LLM hívás
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Válasz kinyerése
            response_text = response.choices[0].message.content.strip()
            finish_reason = response.choices[0].finish_reason

            # Token használat
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            logger.info(f"Válasz generálva: {len(response_text)} karakter, {latency_ms}ms")

            return {
                'response_text': response_text,
                'finish_reason': finish_reason,
                'usage': usage,
                'latency_ms': latency_ms,
                'model': self.chat_model
            }

        except Exception as e:
            logger.error(f"Hiba a válasz generáláskor: {e}")
            raise

    def run_full_pipeline(
        self,
        query: str,
        top_k: int = TOP_K_CHUNKS,
        similarity_threshold: float = SIMILARITY_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Teljes RAG pipeline futtatása egy lekérdezéshez.

        Ez a fő belépési pont válaszok generálásához az értékelési pipeline-ban.

        Args:
            query: Felhasználó kérdése
            top_k: Lekérendő chunk-ok száma
            similarity_threshold: Minimum hasonlóság a lekéréshez

        Returns:
            Dictionary tartalmazva:
                - query: Eredeti lekérdezés
                - query_embedding: Embedding vektor
                - retrieved_chunks: Lekért chunk-ok listája
                - generated_response: LLM válasz szöveg
                - response_metadata: Token használat, latencia, stb.
                - pipeline_metadata: Időbélyegek, paraméterek

        Példa:
            >>> runner = AssistantRunner()
            >>> result = runner.run_full_pipeline("Ki Mowgli?")
            >>> print(f"Lekérdezés: {result['query']}")
            >>> print(f"Válasz: {result['generated_response']}")
            >>> print(f"Lekért {len(result['retrieved_chunks'])} chunk")
        """
        pipeline_start = time.time()

        try:
            # 1. lépés: Query embedding generálás
            logger.info(f"Pipeline kezdés: '{query}'")
            query_embedding = self.generate_embedding(query)

            # 2. lépés: Vektoros keresés
            retrieved_chunks = self.search_context(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=similarity_threshold
            )

            if not retrieved_chunks:
                logger.warning(f"Nem található chunk a lekérdezéshez: '{query}'")
                return {
                    'query': query,
                    'query_embedding': query_embedding,
                    'retrieved_chunks': [],
                    'generated_response': "Nem találtam releváns információt a kérdés megválaszolásához.",
                    'response_metadata': {
                        'error': 'no_chunks_found',
                        'usage': {},
                        'latency_ms': 0
                    },
                    'pipeline_metadata': {
                        'total_latency_ms': int((time.time() - pipeline_start) * 1000),
                        'top_k': top_k,
                        'similarity_threshold': similarity_threshold,
                        'chunks_retrieved': 0
                    }
                }

            # 3. lépés: Válasz generálás
            response_data = self.generate_response(query, retrieved_chunks)

            # Teljes pipeline latencia számítás
            total_latency_ms = int((time.time() - pipeline_start) * 1000)

            # Végső eredmény összeállítása
            result = {
                'query': query,
                'query_embedding': query_embedding,
                'retrieved_chunks': retrieved_chunks,
                'generated_response': response_data['response_text'],
                'response_metadata': {
                    'finish_reason': response_data['finish_reason'],
                    'usage': response_data['usage'],
                    'latency_ms': response_data['latency_ms'],
                    'model': response_data['model']
                },
                'pipeline_metadata': {
                    'total_latency_ms': total_latency_ms,
                    'top_k': top_k,
                    'similarity_threshold': similarity_threshold,
                    'chunks_retrieved': len(retrieved_chunks),
                    'embedding_model': self.embedding_model
                }
            }

            logger.info(f"Pipeline befejezve: {total_latency_ms}ms összesen")
            return result

        except Exception as e:
            logger.error(f"Pipeline sikertelen a lekérdezéshez '{query}': {e}")
            raise


# ============================================================================
# SEGÉDESZKÖZ FÜGGVÉNYEK
# ============================================================================

def run_assistant_batch(
    queries: List[str],
    runner: Optional[AssistantRunner] = None,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """
    Asszisztens futtatása lekérdezések batch-jén.

    Args:
        queries: Lekérdezés string-ek listája
        runner: AssistantRunner példány (új létrehozása ha None)
        show_progress: tqdm progress bar megjelenítése

    Returns:
        Eredmény dictionary-k listája (egy per lekérdezés)

    Példa:
        >>> queries = ["Ki Mowgli?", "Mi Bagheera?"]
        >>> results = run_assistant_batch(queries)
        >>> for r in results:
        ...     print(f"{r['query']}: {r['generated_response'][:50]}...")
    """
    if runner is None:
        runner = AssistantRunner()

    results = []

    # Opcionális progress bar
    iterator = queries
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(queries, desc="Asszisztens futtatása")
        except ImportError:
            logger.warning("tqdm nincs telepítve, progress bar letiltva")

    for query in iterator:
        try:
            result = runner.run_full_pipeline(query)
            results.append(result)
        except Exception as e:
            logger.error(f"Sikertelen lekérdezés feldolgozás '{query}': {e}")
            # Hiba eredmény hozzáadása
            results.append({
                'query': query,
                'error': str(e),
                'generated_response': None
            })

    return results


if __name__ == '__main__':
    # Asszisztens futtató tesztelése
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Teszt lekérdezés
    test_query = "Ki A Dzsungel könyve szerzője?"

    try:
        logger.info("=" * 80)
        logger.info("AssistantRunner tesztelése")
        logger.info("=" * 80)

        runner = AssistantRunner()

        logger.info(f"\nTeszt lekérdezés: '{test_query}'")
        result = runner.run_full_pipeline(test_query)

        logger.info("\n" + "=" * 80)
        logger.info("EREDMÉNYEK")
        logger.info("=" * 80)
        logger.info(f"Lekérdezés: {result['query']}")
        logger.info(f"Lekért chunk-ok: {len(result['retrieved_chunks'])}")

        if result['retrieved_chunks']:
            logger.info("\nLegjobb chunk:")
            chunk = result['retrieved_chunks'][0]
            logger.info(f"  Hasonlóság: {chunk['similarity_score']:.3f}")
            logger.info(f"  Tartalom: {chunk['content'][:200]}...")

        logger.info(f"\nGenerált válasz:")
        logger.info(f"  {result['generated_response']}")

        logger.info(f"\nMetaadatok:")
        logger.info(f"  Teljes latencia: {result['pipeline_metadata']['total_latency_ms']}ms")
        logger.info(f"  Tokenek: {result['response_metadata']['usage']['total_tokens']}")

        logger.info("\n✓ AssistantRunner teszt sikeres")

    except Exception as e:
        logger.error(f"✗ AssistantRunner teszt sikertelen: {e}")
        sys.exit(1)
