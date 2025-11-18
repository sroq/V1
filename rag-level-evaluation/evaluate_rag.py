"""
RAG retrieval teljesítmény értékelése generált kérdések használatával.

Ez a script az értékelési folyamat MÁSODIK LÉPÉSE:
1. Betölti a generált kérdéseket JSON fájlból
2. Minden kérdéshez elvégzi a vector similarity search-t
3. Ellenőrzi, hogy az eredeti chunk megtalálható-e a visszaadott eredmények között
4. Kiszámítja a metrikákat (Top-K Recall, Top-1 Precision)
5. Elmenti az értékelési eredményeket

FONTOS: Ez NEM az /api/chat endpoint-ot használja, hanem közvetlenül
a vector search-t végzi Python-ban, hogy meg tudjuk vizsgálni a chunk ID-kat!

Futtatás:
    python3 evaluate_rag.py

Input:
    data/generated_questions.json (generate_questions.py output-ja)

Output:
    data/evaluation_results.json

Időigény:
    ~2-3 mp / kérdés (embedding + vector search)
    100 kérdés esetén: ~5-10 perc
"""
import json
from tqdm import tqdm  # Progress bar
from utils import (
    get_db_connection,
    search_similar_chunks,
    generate_embedding,
    calculate_metrics
)
from config import QUESTIONS_FILE, EVALUATION_RESULTS_FILE, DEFAULT_MATCH_COUNT
import sys
import os


def main():
    """Fő függvény: RAG retrieval teljesítmény értékelése."""

    # ========================================================================
    # 1. KEZDŐ BANNER
    # ========================================================================
    print("=" * 80)
    print("RAG EVALUATION - RETRIEVAL EVALUATION")
    print("=" * 80)
    print()

    # ========================================================================
    # 2. KÉRDÉSEK FÁJL ELLENŐRZÉSE ÉS BETÖLTÉSE
    # ========================================================================
    # Ellenőrizzük, hogy létezik-e a kérdések fájlja
    if not os.path.exists(QUESTIONS_FILE):
        print(f"✗ Questions file not found: {QUESTIONS_FILE}")
        print("  Please run generate_questions.py first.")
        sys.exit(1)

    # Kérdések betöltése JSON fájlból
    print(f"Loading questions from {QUESTIONS_FILE}...")
    try:
        with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        print(f"✓ Loaded {len(questions_data)} questions")
    except Exception as e:
        print(f"✗ Failed to load questions: {e}")
        sys.exit(1)

    # Ha nincs kérdés, kilépünk
    if len(questions_data) == 0:
        print("\n⚠ No questions found. Please generate questions first.")
        sys.exit(0)

    # ========================================================================
    # 3. ADATBÁZIS KAPCSOLÓDÁS
    # ========================================================================
    print("\nConnecting to database...")
    try:
        conn = get_db_connection()
        print("✓ Database connection established")
    except Exception as e:
        print(f"✗ Failed to connect to database: {e}")
        sys.exit(1)

    # ========================================================================
    # 4. RAG RETRIEVAL ÉRTÉKELÉSE MINDEN KÉRDÉSHEZ
    # ========================================================================
    print(f"\nEvaluating RAG retrieval (Top-K={DEFAULT_MATCH_COUNT})...")

    # Lista az értékelési eredményekhez
    evaluation_results = []

    # Lista a sikertelen query-khez
    failed_queries = []

    # Iterálás minden kérdésen
    # questions_data.items() -> (chunk_id, {question, metadata, ...})
    for chunk_id, data in tqdm(questions_data.items(), desc="Evaluating"):
        question = data['question']          # A generált kérdés
        original_chunk_id = chunk_id         # Az eredeti chunk ID-ja

        try:
            # ================================================================
            # 4a. Embedding generálása a kérdéshez
            # ================================================================
            # Ez ugyanazt az embedding modellt használja, mint a chunkokhoz
            # (text-embedding-3-small), így szemantikailag hasonló vektorokat kapunk
            query_embedding = generate_embedding(question)

            # ================================================================
            # 4b. Vector similarity search az adatbázisban
            # ================================================================
            # Ez közvetlenül a pgvector <=> operátort használja
            # Visszaadja a top-K legközelebbi chunk-ot similarity score-ral
            retrieved_chunks = search_similar_chunks(
                conn,
                query_embedding,
                limit=DEFAULT_MATCH_COUNT  # Általában 5
            )

            # ================================================================
            # 4c. Chunk ID-k és similarity score-ok kinyerése
            # ================================================================
            # retrieved_chunks egy lista dictionary-kkel:
            # [{id: "abc", similarity: 0.85, content: "...", ...}, ...]
            retrieved_chunk_ids = [str(chunk['id']) for chunk in retrieved_chunks]
            retrieved_similarities = [chunk['similarity'] for chunk in retrieved_chunks]

            # ================================================================
            # 4d. Metrikák számítása
            # ================================================================
            # calculate_metrics() meghatározza:
            # - top_k_recall: benne van-e az eredeti chunk a top-K-ban?
            # - top_1_precision: az eredeti chunk az első helyen van-e?
            # - rank: hányadik helyen van (ha benne van)?
            # - similarity_score: mekkora a similarity (ha benne van)?
            metrics = calculate_metrics(
                original_chunk_id,
                retrieved_chunk_ids,
                retrieved_similarities
            )

            # ================================================================
            # 4e. Chunking stratégia kinyerése metadata-ból
            # ================================================================
            # Később ez alapján tudunk stratégiánként elemezni
            chunking_strategy = 'unknown'
            if data.get('metadata') and isinstance(data['metadata'], dict):
                chunking_strategy = data['metadata'].get('chunking_strategy', 'unknown')

            # ================================================================
            # 4f. Eredmény tárolása
            # ================================================================
            result = {
                'chunk_id': original_chunk_id,              # Eredeti chunk ID
                'question': question,                       # A kérdés
                'retrieved_chunk_ids': retrieved_chunk_ids, # Visszaadott chunk ID-k (lista)
                'retrieved_similarities': retrieved_similarities,  # Similarity score-ok (lista)
                'metrics': metrics,                         # Számított metrikák (dict)
                'chunking_strategy': chunking_strategy,     # Stratégia neve
                'token_count': data.get('token_count')      # Token szám
            }
            evaluation_results.append(result)

        except Exception as e:
            # Ha hiba történt (pl. OpenAI API hiba, DB hiba)
            print(f"\n⚠ Failed to evaluate question for chunk {chunk_id}: {e}")
            failed_queries.append(chunk_id)

    # ========================================================================
    # 5. ADATBÁZIS KAPCSOLAT LEZÁRÁSA
    # ========================================================================
    conn.close()

    # ========================================================================
    # 6. ÉRTÉKELÉSI EREDMÉNYEK MENTÉSE
    # ========================================================================
    print(f"\nSaving evaluation results to {EVALUATION_RESULTS_FILE}...")
    try:
        with open(EVALUATION_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(evaluation_results)} evaluation results")
    except Exception as e:
        print(f"✗ Failed to save results: {e}")
        sys.exit(1)

    # ========================================================================
    # 7. GYORS ÖSSZEFOGLALÓ
    # ========================================================================
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)

    total = len(evaluation_results)
    # Top-K Recall és Top-1 Precision összeszámolása
    top_k_recalls = sum(r['metrics']['top_k_recall'] for r in evaluation_results)
    top_1_precisions = sum(r['metrics']['top_1_precision'] for r in evaluation_results)

    if total > 0:
        print(f"Total queries evaluated: {total}")
        print(f"Top-K Recall: {(top_k_recalls / total) * 100:.2f}% ({top_k_recalls}/{total})")
        print(f"Top-1 Precision: {(top_1_precisions / total) * 100:.2f}% ({top_1_precisions}/{total})")

        if failed_queries:
            print(f"\nFailed queries: {len(failed_queries)}")

        print(f"\nDetailed analysis: Run analyze_results.py")
        print(f"Results saved to: {EVALUATION_RESULTS_FILE}")
    else:
        print("No results to analyze.")

    print()


# ============================================================================
# SCRIPT BELÉPÉSI PONT
# ============================================================================
if __name__ == '__main__':
    main()
