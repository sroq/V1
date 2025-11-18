"""
Metrika számítási és eredmény formázási függvények.

Ez a modul tartalmazza az értékelési metrikák számításához szükséges függvényeket:
- Egyedi query metrikák (Top-K Recall, Top-1 Precision, Rank, Similarity)
- Aggregált metrikák (átlagok, százalékok)
- Eredmények formázása és DataFrame konverzió
"""
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_metrics(
    original_chunk_id: str,
    retrieved_chunk_ids: List[str],
    retrieved_similarities: List[float],
    k: int = 5
) -> Dict[str, Any]:
    """
    Értékelési metrikák számítása egyetlen kérdéshez.

    Ez a fő metrika-számító függvény. Minden kérdéshez meghatározza:

    **Binary Single-Relevancia Metrikák (átnevezve):**
    1. Hit Rate@K: Benne van az eredeti chunk a top-K-ban? (0 vagy 1)
    2. First Position Accuracy: Az eredeti chunk az első helyen van? (0 vagy 1)
    3. Rank: Hányadik pozícióban van? (1-K vagy None)
    4. Similarity Score: Mekkora a similarity? (0-1 vagy None)

    **Klasszikus IR Metrikák:**
    5. Precision@K: Különböző K értékekre (1, 3, 5, 10)
    6. Recall@K: Különböző K értékekre (1, 3, 5, 10)

    Args:
        original_chunk_id: Az eredeti chunk ID-ja, amelyből a kérdést generáltuk
        retrieved_chunk_ids: A retrieved chunk ID-k listája (similarity szerint rendezve)
        retrieved_similarities: A hozzájuk tartozó similarity score-ok
        k: Top-K érték (default: 5)

    Returns:
        Dictionary a következő metrikákkal:

        Binary metrikák (átnevezve):
        - hit_rate_at_k: 1 ha a chunk a top-K-ban van, 0 ha nincs
        - first_position_accuracy: 1 ha a chunk az első helyen van, 0 ha nem
        - rank: A chunk pozíciója (1-től indexelt), None ha nincs benne
        - similarity_score: A chunk similarity score-ja, None ha nincs benne
        - found: Boolean, True ha megtalálható, False ha nem

        Klasszikus IR metrikák:
        - precision_at_1, precision_at_3, precision_at_5, precision_at_10
        - recall_at_1, recall_at_3, recall_at_5, recall_at_10

        Backward compatibility (alias):
        - top_k_recall: Alias for hit_rate_at_k
        - top_1_precision: Alias for first_position_accuracy

    Példa:
        >>> metrics = calculate_metrics(
        ...     original_chunk_id="abc-123",
        ...     retrieved_chunk_ids=["def-456", "abc-123", "ghi-789"],
        ...     retrieved_similarities=[0.95, 0.85, 0.75]
        ... )
        >>> print(metrics)
        {
            'hit_rate_at_k': 1,              # Benne van a top-3-ban
            'first_position_accuracy': 0,    # Nem első helyen van
            'rank': 2,                       # 2. helyen van
            'similarity_score': 0.85,        # Similarity score
            'found': True,                   # Megtalálható
            'precision_at_5': 0.2,           # 1/5 releváns
            'recall_at_5': 1.0,              # 1/1 megtalálva
            'top_k_recall': 1,               # Backward compat alias
            'top_1_precision': 0,            # Backward compat alias
        }
    """
    # =======================================================================
    # BINARY SINGLE-RELEVANCIA METRIKÁK (átnevezve)
    # =======================================================================
    # Alapértelmezett metrikák inicializálása (nincs találat esetére)
    metrics = {
        'hit_rate_at_k': 0,
        'first_position_accuracy': 0,
        'rank': None,
        'similarity_score': None,
        'found': False
    }

    # Ellenőrizzük, hogy az eredeti chunk benne van-e a retrieved listában
    if original_chunk_id in retrieved_chunk_ids:
        # Megtalálható!
        metrics['found'] = True

        # Rank számítása: hányadik pozícióban van? (1-től indexelve)
        # Példa: ha retrieved_chunk_ids[1] == original_chunk_id, akkor rank = 2
        rank = retrieved_chunk_ids.index(original_chunk_id) + 1  # +1 mert 1-től indexelünk
        metrics['rank'] = rank

        # Similarity score kinyerése a megfelelő pozícióból
        # rank-1 mert a lista 0-tól indexelt, de a rank 1-től
        metrics['similarity_score'] = retrieved_similarities[rank - 1]

        # Hit Rate@K: ha itt vagyunk, a chunk biztosan a top-K-ban van
        # (mert a search_similar_chunks maximum K darab chunk-ot ad vissza)
        metrics['hit_rate_at_k'] = 1

        # First Position Accuracy: csak akkor 1, ha a chunk az első helyen van
        if rank == 1:
            metrics['first_position_accuracy'] = 1

    # =======================================================================
    # KLASSZIKUS IR METRIKÁK (Precision@K, Recall@K)
    # =======================================================================
    # Single-relevancia esetén: relevant_chunk_ids = [original_chunk_id]
    relevant_chunk_ids = [original_chunk_id]

    # Precision@K és Recall@K számítása különböző K értékekre
    multi_k_metrics = calculate_metrics_at_multiple_k(
        relevant_chunk_ids,
        retrieved_chunk_ids,
        k_values=[1, 3, 5, 10]
    )

    # Klasszikus metrikák hozzáadása
    metrics.update(multi_k_metrics)

    # =======================================================================
    # BACKWARD COMPATIBILITY - Régi metrika nevek alias-ok
    # =======================================================================
    # Hogy a régi kódok továbbra is működjenek
    metrics['top_k_recall'] = metrics['hit_rate_at_k']
    metrics['top_1_precision'] = metrics['first_position_accuracy']

    return metrics


def aggregate_metrics(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Metrikák aggregálása az összes értékelési eredmény alapján.

    Ez a függvény kiszámítja az összesített statisztikákat:

    **Binary Single-Relevance Metrics (átnevezett):**
    - Hit Rate@K % (korábban: Top-K Recall) - hány %-ban találta meg a chunk-ot
    - First Position Accuracy % (korábban: Top-1 Precision) - hány %-ban volt első helyen
    - Átlagos rank és similarity (ha megtalálható volt)

    **Classical IR Metrics:**
    - MRR (Mean Reciprocal Rank) - átlagos reciprok rank
    - Precision@K és Recall@K (K=1,3,5,10) - klasszikus IR metrikák

    **Backward Compatibility:**
    - top_k_recall_percentage és top_1_precision_percentage aliasok megtartva

    Args:
        evaluation_results: Lista az értékelési eredményekkel (minden elem egy dict)
                            Minden elemnek tartalmaznia kell egy 'metrics' kulcsot

    Returns:
        Dictionary az aggregált metrikákkal:

        Binary metrics (új nevek):
        - total_queries: Összes kérdés száma
        - hit_rate_at_k_percentage: Hit Rate@K % (0-100)
        - first_position_accuracy_percentage: First Position Accuracy % (0-100)
        - average_rank: Átlagos rank (ha volt találat)
        - average_similarity: Átlagos similarity (ha volt találat)
        - found_count: Hány esetben találta meg a chunk-ot
        - found_percentage: Found % (0-100)

        Classical IR metrics:
        - mrr: Mean Reciprocal Rank (0-1)
        - avg_precision_at_1: Átlagos Precision@1 (0-1)
        - avg_precision_at_3: Átlagos Precision@3 (0-1)
        - avg_precision_at_5: Átlagos Precision@5 (0-1)
        - avg_precision_at_10: Átlagos Precision@10 (0-1)
        - avg_recall_at_1: Átlagos Recall@1 (0-1)
        - avg_recall_at_3: Átlagos Recall@3 (0-1)
        - avg_recall_at_5: Átlagos Recall@5 (0-1)
        - avg_recall_at_10: Átlagos Recall@10 (0-1)

        Backward compatibility aliases:
        - top_k_recall_percentage: Alias for hit_rate_at_k_percentage
        - top_1_precision_percentage: Alias for first_position_accuracy_percentage

    Példa:
        >>> results = [
        ...     {'metrics': {'hit_rate_at_k': 1, 'first_position_accuracy': 1, 'rank': 1,
        ...                  'similarity_score': 0.9, 'found': True, 'precision_at_1': 1.0,
        ...                  'recall_at_1': 1.0}},
        ...     {'metrics': {'hit_rate_at_k': 1, 'first_position_accuracy': 0, 'rank': 3,
        ...                  'similarity_score': 0.7, 'found': True, 'precision_at_3': 0.33,
        ...                  'recall_at_3': 1.0}},
        ...     {'metrics': {'hit_rate_at_k': 0, 'first_position_accuracy': 0, 'rank': None,
        ...                  'similarity_score': None, 'found': False, 'precision_at_5': 0.0,
        ...                  'recall_at_5': 0.0}}
        ... ]
        >>> agg = aggregate_metrics(results)
        >>> print(f"Hit Rate@K: {agg['hit_rate_at_k_percentage']:.1f}%")
        Hit Rate@K: 66.7%  # 2 out of 3
        >>> print(f"MRR: {agg['mrr']:.3f}")
        MRR: 0.583  # (1/1 + 1/3 + 0) / 3
    """
    total_queries = len(evaluation_results)

    # Ha nincs eredmény, adjunk vissza null értékeket
    if total_queries == 0:
        return {
            'total_queries': 0,
            # Binary metrics (új nevek)
            'hit_rate_at_k_percentage': 0.0,
            'first_position_accuracy_percentage': 0.0,
            'average_rank': None,
            'average_similarity': None,
            'found_count': 0,
            'found_percentage': 0.0,
            # Classical IR metrics
            'mrr': 0.0,
            'avg_precision_at_1': 0.0,
            'avg_precision_at_3': 0.0,
            'avg_precision_at_5': 0.0,
            'avg_precision_at_10': 0.0,
            'avg_recall_at_1': 0.0,
            'avg_recall_at_3': 0.0,
            'avg_recall_at_5': 0.0,
            'avg_recall_at_10': 0.0,
            # Backward compatibility
            'top_k_recall_percentage': 0.0,
            'top_1_precision_percentage': 0.0
        }

    # Binary metrics gyűjtése (backward compatible - ellenőrizzük mindkét nevet)
    hit_rates = []
    first_position_accuracies = []

    for r in evaluation_results:
        # Hit rate: próbáljuk az új nevet, ha nincs, használjuk a régit
        hit_rate = r['metrics'].get('hit_rate_at_k', r['metrics'].get('top_k_recall', 0))
        hit_rates.append(hit_rate)

        # First position accuracy: próbáljuk az új nevet, ha nincs, használjuk a régit
        first_pos = r['metrics'].get('first_position_accuracy', r['metrics'].get('top_1_precision', 0))
        first_position_accuracies.append(first_pos)

    # Rank és similarity csak azokhoz, ahol van érték (where not None)
    ranks = [r['metrics']['rank'] for r in evaluation_results if r['metrics']['rank'] is not None]
    similarities = [r['metrics']['similarity_score'] for r in evaluation_results if r['metrics']['similarity_score'] is not None]

    # Hány esetben találta meg a chunk-ot?
    found_count = sum(1 for r in evaluation_results if r['metrics']['found'])

    # MRR számítása a külön függvénnyel
    mrr_value = calculate_mrr(evaluation_results)

    # Classical IR metrics gyűjtése (Precision@K és Recall@K)
    # Backward compatible: ha nem léteznek, használjunk 0.0 default értéket
    precision_at_1_values = [r['metrics'].get('precision_at_1', 0.0) for r in evaluation_results]
    precision_at_3_values = [r['metrics'].get('precision_at_3', 0.0) for r in evaluation_results]
    precision_at_5_values = [r['metrics'].get('precision_at_5', 0.0) for r in evaluation_results]
    precision_at_10_values = [r['metrics'].get('precision_at_10', 0.0) for r in evaluation_results]

    recall_at_1_values = [r['metrics'].get('recall_at_1', 0.0) for r in evaluation_results]
    recall_at_3_values = [r['metrics'].get('recall_at_3', 0.0) for r in evaluation_results]
    recall_at_5_values = [r['metrics'].get('recall_at_5', 0.0) for r in evaluation_results]
    recall_at_10_values = [r['metrics'].get('recall_at_10', 0.0) for r in evaluation_results]

    # Aggregált metrikák számítása
    aggregated = {
        'total_queries': total_queries,

        # Binary metrics (új nevek)
        'hit_rate_at_k_percentage': (sum(hit_rates) / total_queries) * 100,
        'first_position_accuracy_percentage': (sum(first_position_accuracies) / total_queries) * 100,
        'average_rank': float(np.mean(ranks)) if ranks else None,
        'average_similarity': float(np.mean(similarities)) if similarities else None,
        'found_count': found_count,
        'found_percentage': (found_count / total_queries) * 100,

        # Classical IR metrics
        'mrr': mrr_value,
        'avg_precision_at_1': float(np.mean(precision_at_1_values)),
        'avg_precision_at_3': float(np.mean(precision_at_3_values)),
        'avg_precision_at_5': float(np.mean(precision_at_5_values)),
        'avg_precision_at_10': float(np.mean(precision_at_10_values)),
        'avg_recall_at_1': float(np.mean(recall_at_1_values)),
        'avg_recall_at_3': float(np.mean(recall_at_3_values)),
        'avg_recall_at_5': float(np.mean(recall_at_5_values)),
        'avg_recall_at_10': float(np.mean(recall_at_10_values)),
    }

    # Backward compatibility: régi nevek aliasként
    aggregated['top_k_recall_percentage'] = aggregated['hit_rate_at_k_percentage']
    aggregated['top_1_precision_percentage'] = aggregated['first_position_accuracy_percentage']

    # Embedding Quality Metrics (ÚJ)
    embedding_separation = analyze_embedding_separation(evaluation_results)
    embedding_roc_auc = calculate_embedding_roc_auc(evaluation_results)

    # Embedding quality metrics hozzáadása az aggregated dict-hez
    aggregated['embedding_quality'] = {
        **embedding_separation,
        'roc_auc': embedding_roc_auc
    }

    # Chunk Quality Metrics (ÚJ)
    chunk_size_consistency = calculate_chunk_size_consistency(evaluation_results)
    retrieval_by_size = calculate_retrieval_by_chunk_size(evaluation_results)
    position_stability = calculate_position_stability(evaluation_results)

    # Chunk quality metrics hozzáadása az aggregated dict-hez
    aggregated['chunk_quality'] = {
        'size_consistency': chunk_size_consistency,
        'retrieval_by_size_bucket': retrieval_by_size,
        'position_stability': position_stability
    }

    return aggregated


def format_results(
    evaluation_results: List[Dict[str, Any]],
    aggregated_metrics: Dict[str, Any]
) -> str:
    """
    Értékelési eredmények formázása olvasható string-gé.

    Ez a függvény egy szépen formázott szöveges összefoglalót készít
    az eredményekről, amely közvetlenül kiíratható a konzolra.
    Tartalmazza mind a binary, mind a classical IR metrikákat.

    Args:
        evaluation_results: Az értékelési eredmények listája (nem használt itt, de tartjuk API kompatibilitásért)
        aggregated_metrics: Az aggregált metrikák dictionary-je

    Returns:
        Formázott string az eredmények összefoglalójával

    Példa:
        >>> formatted = format_results(results, aggregated)
        >>> print(formatted)
        ================================================================================
        RAG EVALUATION RESULTS - SUMMARY
        ================================================================================

        Total Queries: 78
        Chunks Found in Results: 55 (70.51%)

        BINARY SINGLE-RELEVANCE METRICS:
          Hit Rate@K (chunk in top-K): 70.51%
          First Position Accuracy (chunk is #1): 45.21%

        CLASSICAL IR METRICS:
          MRR (Mean Reciprocal Rank): 0.5234

          Precision@K:
            P@1: 0.4521  P@3: 0.3654  P@5: 0.2987  P@10: 0.1876

          Recall@K:
            R@1: 0.4521  R@3: 0.6234  R@5: 0.7051  R@10: 0.7821

        Average Rank (when found): 2.15
        Average Similarity Score (when found): 0.6234

        ================================================================================
    """
    output = []
    # Fejléc
    output.append("=" * 80)
    output.append("RAG EVALUATION RESULTS - SUMMARY")
    output.append("=" * 80)
    output.append("")

    # Alapstatisztikák
    output.append(f"Total Queries: {aggregated_metrics['total_queries']}")
    output.append(f"Chunks Found in Results: {aggregated_metrics['found_count']} ({aggregated_metrics['found_percentage']:.2f}%)")
    output.append("")

    # Binary Single-Relevance Metrics (új nevek)
    output.append("BINARY SINGLE-RELEVANCE METRICS:")
    output.append(f"  Hit Rate@K (chunk in top-K): {aggregated_metrics['hit_rate_at_k_percentage']:.2f}%")
    output.append(f"  First Position Accuracy (chunk is #1): {aggregated_metrics['first_position_accuracy_percentage']:.2f}%")
    output.append("")

    # Classical IR Metrics
    output.append("CLASSICAL IR METRICS:")
    output.append(f"  MRR (Mean Reciprocal Rank): {aggregated_metrics['mrr']:.4f}")
    output.append("")

    # Precision@K táblázat
    output.append("  Precision@K:")
    precision_line = (
        f"    P@1: {aggregated_metrics['avg_precision_at_1']:.4f}  "
        f"P@3: {aggregated_metrics['avg_precision_at_3']:.4f}  "
        f"P@5: {aggregated_metrics['avg_precision_at_5']:.4f}  "
        f"P@10: {aggregated_metrics['avg_precision_at_10']:.4f}"
    )
    output.append(precision_line)
    output.append("")

    # Recall@K táblázat
    output.append("  Recall@K:")
    recall_line = (
        f"    R@1: {aggregated_metrics['avg_recall_at_1']:.4f}  "
        f"R@3: {aggregated_metrics['avg_recall_at_3']:.4f}  "
        f"R@5: {aggregated_metrics['avg_recall_at_5']:.4f}  "
        f"R@10: {aggregated_metrics['avg_recall_at_10']:.4f}"
    )
    output.append(recall_line)
    output.append("")

    # Átlagos rank (ha van)
    if aggregated_metrics['average_rank'] is not None:
        output.append(f"Average Rank (when found): {aggregated_metrics['average_rank']:.2f}")
    else:
        output.append("Average Rank: N/A (no chunks found)")

    # Átlagos similarity (ha van)
    if aggregated_metrics['average_similarity'] is not None:
        output.append(f"Average Similarity Score (when found): {aggregated_metrics['average_similarity']:.4f}")
    else:
        output.append("Average Similarity Score: N/A (no chunks found)")

    # Embedding Quality Metrics (ÚJ szekció)
    if 'embedding_quality' in aggregated_metrics:
        eq = aggregated_metrics['embedding_quality']
        output.append("")
        output.append("EMBEDDING QUALITY METRICS:")
        output.append(f"  Relevant Similarity (mean): {eq['relevant_similarity_mean']:.4f}")
        output.append(f"  Irrelevant Similarity (mean): {eq['irrelevant_similarity_mean']:.4f}")
        output.append(f"  Separation Margin: {eq['separation_margin']:.4f}")
        output.append(f"  ROC-AUC Score: {eq['roc_auc']:.4f}")
        output.append(f"  (Sample sizes: {eq['relevant_count']} relevant, {eq['irrelevant_count']} irrelevant)")

    # Chunk Quality Metrics (ÚJ szekció)
    if 'chunk_quality' in aggregated_metrics:
        cq = aggregated_metrics['chunk_quality']
        output.append("")
        output.append("CHUNK QUALITY METRICS:")

        # Size Consistency
        sc = cq['size_consistency']
        output.append(f"  Chunk Size Consistency Index (CSCI): {sc['csci']:.4f}")
        output.append(f"    Mean chunk size: {sc['mean_chunk_size']:.1f} tokens")
        output.append(f"    Std deviation: {sc['std_chunk_size']:.1f} tokens")
        output.append(f"    Range: {sc['min_chunk_size']}-{sc['max_chunk_size']} tokens")

        # Retrieval by Size Bucket
        rbs = cq['retrieval_by_size_bucket']
        if rbs:
            output.append("")
            output.append("  Retrieval Performance by Chunk Size:")
            for bucket_name in ['tiny', 'small', 'medium', 'large', 'xlarge']:
                if bucket_name in rbs:
                    bucket = rbs[bucket_name]
                    avg_rank_str = f"{bucket['avg_rank']:.2f}" if bucket['avg_rank'] is not None else 'N/A'
                    output.append(f"    {bucket_name.capitalize():8} ({bucket['chunk_count']:2} chunks): Hit Rate={bucket['hit_rate']:.2%}, Avg Rank={avg_rank_str}")

        # Position Stability
        ps = cq['position_stability']
        output.append("")
        output.append(f"  Position Stability (PSS): {ps['overall_stability_mean']:.4f} (±{ps['overall_stability_std']:.4f})")
        output.append(f"    Chunks analyzed: {ps['chunks_analyzed']}")

        # Top 3 most stable chunks
        if ps['top_stable_chunks']:
            output.append("    Top 3 Stable Chunks:")
            for i, chunk in enumerate(ps['top_stable_chunks'][:3], 1):
                output.append(f"      {i}. Chunk {chunk['chunk_id']}: PSS={chunk['pss']:.4f} ({chunk['appearances']} appearances)")

    output.append("")
    output.append("=" * 80)

    # Összeállítás egyetlen string-gé newline-okkal
    return "\n".join(output)


def results_to_dataframe(evaluation_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Értékelési eredmények konvertálása pandas DataFrame-mé elemzéshez.

    DataFrame előnyei:
    - Könnyen szűrhető, csoportosítható
    - CSV-be exportálható
    - Vizualizációhoz használható

    Args:
        evaluation_results: Az értékelési eredmények listája

    Returns:
        pandas DataFrame az eredményekkel, ahol minden sor egy kérdés

    DataFrame oszlopok:
        Alapadatok:
        - chunk_id: Eredeti chunk ID
        - question: Generált kérdés
        - chunking_strategy: A chunking stratégia neve
        - token_count: Token szám

        Binary metrics (új nevek):
        - hit_rate_at_k: 0 vagy 1
        - first_position_accuracy: 0 vagy 1
        - rank: Pozíció (1, 2, 3, ...) vagy None
        - similarity_score: Similarity érték vagy None
        - found: True/False

        Classical IR metrics:
        - precision_at_1, precision_at_3, precision_at_5, precision_at_10: 0-1
        - recall_at_1, recall_at_3, recall_at_5, recall_at_10: 0-1

        Backward compatibility (régi nevek):
        - top_k_recall: alias for hit_rate_at_k
        - top_1_precision: alias for first_position_accuracy

    Példa:
        >>> df = results_to_dataframe(results)
        >>> print(df.head())
        >>> # Szűrés: csak sikeres találatok
        >>> successful = df[df['found'] == True]
        >>> # CSV export
        >>> df.to_csv('results.csv', index=False)
    """
    rows = []

    # Minden eredményből létrehozunk egy sort
    for result in evaluation_results:
        metrics = result['metrics']

        # Backward compatible metric kiolvasás
        hit_rate = metrics.get('hit_rate_at_k', metrics.get('top_k_recall', 0))
        first_pos = metrics.get('first_position_accuracy', metrics.get('top_1_precision', 0))

        row = {
            # Alapadatok
            'chunk_id': result['chunk_id'],
            'question': result['question'],
            'chunking_strategy': result.get('chunking_strategy', 'unknown'),
            'token_count': result.get('token_count', None),

            # Binary metrics (új nevek) - backward compatible
            'hit_rate_at_k': hit_rate,
            'first_position_accuracy': first_pos,
            'rank': metrics['rank'],
            'similarity_score': metrics['similarity_score'],
            'found': metrics['found'],

            # Classical IR metrics - backward compatible (default 0.0)
            'precision_at_1': metrics.get('precision_at_1', 0.0),
            'precision_at_3': metrics.get('precision_at_3', 0.0),
            'precision_at_5': metrics.get('precision_at_5', 0.0),
            'precision_at_10': metrics.get('precision_at_10', 0.0),
            'recall_at_1': metrics.get('recall_at_1', 0.0),
            'recall_at_3': metrics.get('recall_at_3', 0.0),
            'recall_at_5': metrics.get('recall_at_5', 0.0),
            'recall_at_10': metrics.get('recall_at_10', 0.0),

            # Backward compatibility (régi nevek) - ez mindig elérhető
            'top_k_recall': hit_rate,
            'top_1_precision': first_pos,
        }
        rows.append(row)

    # DataFrame létrehozása a sorokból
    df = pd.DataFrame(rows)
    return df


def calculate_metrics_by_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Metrikák számítása chunking stratégia szerint csoportosítva.

    Ez a függvény lehetővé teszi, hogy összehasonlítsuk a különböző
    chunking stratégiák (fixed, semantic, recursive, document-specific)
    teljesítményét.

    Args:
        df: DataFrame az értékelési eredményekkel (results_to_dataframe() output)

    Returns:
        DataFrame a metrikákkal stratégiánként csoportosítva.
        Index: chunking_strategy
        Oszlopok:

        Binary metrics (új nevek):
        - hit_rate_at_k_pct: Hit Rate@K %
        - hit_rate_count: Sikeres retrieval-ok száma
        - total_queries: Összes kérdés ebben a stratégiában
        - first_pos_accuracy_pct: First Position Accuracy %
        - first_pos_count: Első helyen talált chunk-ok száma
        - avg_rank: Átlagos rank (ha találat)
        - avg_similarity: Átlagos similarity (ha találat)

        Classical IR metrics:
        - avg_precision_at_1, avg_precision_at_3, avg_precision_at_5, avg_precision_at_10
        - avg_recall_at_1, avg_recall_at_3, avg_recall_at_5, avg_recall_at_10

    Ha nincs 'chunking_strategy' oszlop, üres DataFrame-et ad vissza.

    Példa:
        >>> df = results_to_dataframe(results)
        >>> strategy_metrics = calculate_metrics_by_strategy(df)
        >>> print(strategy_metrics)
                          hit_rate_at_k_pct  ...  avg_recall_at_10
        chunking_strategy
        semantic                       75.5  ...             0.7550
        fixed                          68.2  ...             0.6820
    """
    # Ellenőrizzük, hogy van-e chunking_strategy oszlop
    if 'chunking_strategy' not in df.columns:
        return pd.DataFrame()

    # Pandas groupby + aggregálás
    # groupby('chunking_strategy'): stratégiánként csoportosít
    # agg(...): minden csoportra aggregációkat alkalmaz
    strategy_metrics = df.groupby('chunking_strategy').agg({
        # Binary metrics (új nevek)
        'hit_rate_at_k': ['mean', 'sum', 'count'],
        'first_position_accuracy': ['mean', 'sum'],
        'rank': 'mean',
        'similarity_score': 'mean',

        # Classical IR metrics
        'precision_at_1': 'mean',
        'precision_at_3': 'mean',
        'precision_at_5': 'mean',
        'precision_at_10': 'mean',
        'recall_at_1': 'mean',
        'recall_at_3': 'mean',
        'recall_at_5': 'mean',
        'recall_at_10': 'mean',
    }).round(4)  # 4 tizedesjegyre kerekítés

    # Az oszlopnevek átnevezése érthetőbb nevekre
    # Az agg() multi-level oszlopneveket hoz létre, ezeket egyszerűsítjük
    strategy_metrics.columns = [
        # Binary metrics
        'hit_rate_at_k_pct', 'hit_rate_count', 'total_queries',
        'first_pos_accuracy_pct', 'first_pos_count',
        'avg_rank', 'avg_similarity',

        # Classical IR metrics (ezek már átlagok, nem kell percent-té alakítani)
        'avg_precision_at_1',
        'avg_precision_at_3',
        'avg_precision_at_5',
        'avg_precision_at_10',
        'avg_recall_at_1',
        'avg_recall_at_3',
        'avg_recall_at_5',
        'avg_recall_at_10',
    ]

    # Százalékokká konvertálás (mean 0-1 tartományban van, szorozzuk 100-zal)
    strategy_metrics['hit_rate_at_k_pct'] *= 100
    strategy_metrics['first_pos_accuracy_pct'] *= 100

    return strategy_metrics


def calculate_mrr(evaluation_results: List[Dict[str, Any]]) -> float:
    """
    Mean Reciprocal Rank (MRR) számítása.

    Az MRR egy standard Information Retrieval metrika, amely méri,
    hogy átlagosan mennyire "előre" van az első releváns találat.

    Formula: MRR = 1/N × Σ(1/rank_i)
    ahol rank_i az első releváns chunk pozíciója az i-edik query esetén.

    Értelmezés:
    - Ha nincs releváns chunk a top-K-ban, 1/rank_i = 0
    - Ha a releváns chunk az 1. helyen van, 1/rank_i = 1.0
    - Ha a releváns chunk a 2. helyen van, 1/rank_i = 0.5
    - Ha a releváns chunk az 5. helyen van, 1/rank_i = 0.2

    Args:
        evaluation_results: Az értékelési eredmények listája

    Returns:
        MRR érték 0.0 és 1.0 között
        - 1.0: Minden query esetén az első helyen van a releváns chunk (tökéletes)
        - 0.5: Átlagosan a 2. helyen van
        - 0.0: Egyetlen query-nél sem találta meg a releváns chunk-ot

    Példa:
        >>> results = [
        ...     {'metrics': {'rank': 1}},   # RR = 1/1 = 1.0
        ...     {'metrics': {'rank': 3}},   # RR = 1/3 = 0.333
        ...     {'metrics': {'rank': None}} # RR = 0.0 (nincs találat)
        ... ]
        >>> mrr = calculate_mrr(results)
        >>> print(f"MRR: {mrr:.3f}")
        MRR: 0.444  # (1.0 + 0.333 + 0.0) / 3

    Különbség az average_rank-tól:
    - average_rank: Csak azokat veszi figyelembe, ahol VOLT találat
    - MRR: Minden query-t figyelembe vesz, 0-val számol ha nincs találat
    """
    if not evaluation_results:
        return 0.0

    reciprocal_ranks = []

    for result in evaluation_results:
        rank = result['metrics']['rank']

        if rank is not None:
            # Van találat: reciprok érték (1/rank)
            reciprocal_ranks.append(1.0 / rank)
        else:
            # Nincs találat: 0.0
            reciprocal_ranks.append(0.0)

    # Átlag számítása
    mrr = np.mean(reciprocal_ranks)

    return float(mrr)


def calculate_precision_at_k(
    relevant_chunk_ids: List[str],
    retrieved_chunk_ids: List[str],
    k: int
) -> float:
    """
    Precision@K számítása - Klasszikus Information Retrieval metrika.

    Precision@K = (releváns ÉS lekért top-K) / K

    Azt méri, hogy a top-K lekért chunk közül hány releváns.

    Single-relevancia esetén (jelenlegi RAG evaluation):
    - relevant_chunk_ids = [original_chunk_id]  # Csak 1 releváns chunk
    - K = 5 (top-5 retrieval)
    - Precision@5 lehetséges értékei:
      * 0.0 (0/5): Nincs találat a top-5-ben
      * 0.2 (1/5): Az 1 releváns chunk benne van a top-5-ben

    Multi-relevancia esetén (ha a jövőben több chunk is releváns):
    - relevant_chunk_ids = [chunk1, chunk2, chunk3]  # 3 releváns chunk
    - Ha mind a 3 benne van a top-5-ben: Precision@5 = 3/5 = 0.6
    - Ha csak 2 benne van: Precision@5 = 2/5 = 0.4

    Args:
        relevant_chunk_ids: Releváns chunk ID-k listája (ground truth)
        retrieved_chunk_ids: Lekért chunk ID-k listája (retrieval output)
        k: Hány top chunk-ot veszünk figyelembe (pl. 5)

    Returns:
        Precision@K érték 0.0 és 1.0 között

    Példa (single-relevancia):
        >>> precision_at_5 = calculate_precision_at_k(
        ...     relevant_chunk_ids=['abc-123'],
        ...     retrieved_chunk_ids=['def-456', 'abc-123', 'ghi-789', 'jkl-012', 'mno-345'],
        ...     k=5
        ... )
        >>> print(f"Precision@5: {precision_at_5:.2f}")
        Precision@5: 0.20  # 1 releváns / 5 lekért = 0.20
    """
    if k == 0:
        return 0.0

    # Top-K lekért chunk-ok (csak az első K darab)
    retrieved_at_k = set(retrieved_chunk_ids[:k])

    # Releváns chunk-ok (set-té konvertálva a gyors intersection-hoz)
    relevant_set = set(relevant_chunk_ids)

    # Hány releváns chunk van a top-K között? (intersection)
    relevant_and_retrieved = relevant_set & retrieved_at_k

    # Precision@K = releváns ÉS lekért / K
    precision = len(relevant_and_retrieved) / k

    return float(precision)


def calculate_recall_at_k(
    relevant_chunk_ids: List[str],
    retrieved_chunk_ids: List[str],
    k: int
) -> float:
    """
    Recall@K számítása - Klasszikus Information Retrieval metrika.

    Recall@K = (releváns ÉS lekért top-K) / (összes releváns)

    Azt méri, hogy az összes releváns chunk közül hányat találtunk meg a top-K-ban.

    Single-relevancia esetén (jelenlegi RAG evaluation):
    - relevant_chunk_ids = [original_chunk_id]  # Csak 1 releváns chunk
    - Összes releváns = 1
    - Recall@K lehetséges értékei:
      * 0.0 (0/1): Nincs találat
      * 1.0 (1/1): Megtaláltuk az 1 releváns chunk-ot
    - Ez MEGEGYEZIK a hit_rate_at_k (régebben top_k_recall) metrikával!

    Multi-relevancia esetén (ha a jövőben több chunk is releváns):
    - relevant_chunk_ids = [chunk1, chunk2, chunk3, chunk4, chunk5]  # 5 releváns chunk
    - Ha 3-at megtaláltunk a top-10-ben: Recall@10 = 3/5 = 0.6
    - Ha mind az 5-öt megtaláltuk: Recall@10 = 5/5 = 1.0

    Args:
        relevant_chunk_ids: Releváns chunk ID-k listája (ground truth)
        retrieved_chunk_ids: Lekért chunk ID-k listája (retrieval output)
        k: Hány top chunk-ot veszünk figyelembe (pl. 5)

    Returns:
        Recall@K érték 0.0 és 1.0 között

    Példa (single-relevancia):
        >>> recall_at_5 = calculate_recall_at_k(
        ...     relevant_chunk_ids=['abc-123'],
        ...     retrieved_chunk_ids=['def-456', 'abc-123', 'ghi-789'],
        ...     k=5
        ... )
        >>> print(f"Recall@5: {recall_at_5:.2f}")
        Recall@5: 1.00  # 1 megtalált / 1 releváns = 1.00

    Különbség a Precision@K-tól:
    - Precision@K: Lekért chunk-ok közül hány releváns? (lekérésorientált)
    - Recall@K: Releváns chunk-ok közül hányat találtunk? (teljességorientált)
    """
    if len(relevant_chunk_ids) == 0:
        # Nincs releváns chunk → Recall nincs definiálva, 0.0-t adunk vissza
        return 0.0

    # Top-K lekért chunk-ok
    retrieved_at_k = set(retrieved_chunk_ids[:k])

    # Releváns chunk-ok
    relevant_set = set(relevant_chunk_ids)

    # Hány releváns chunk van a top-K között?
    relevant_and_retrieved = relevant_set & retrieved_at_k

    # Recall@K = releváns ÉS lekért / összes releváns
    recall = len(relevant_and_retrieved) / len(relevant_set)

    return float(recall)


def calculate_metrics_at_multiple_k(
    relevant_chunk_ids: List[str],
    retrieved_chunk_ids: List[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Precision@K és Recall@K számítása különböző K értékekre.

    Ez a helper függvény több K értékre is kiszámítja a Precision és Recall metrikákat,
    így láthatjuk, hogyan változnak ezek a metrikák a retrieved chunks számának
    növelésével.

    Args:
        relevant_chunk_ids: Releváns chunk ID-k listája
        retrieved_chunk_ids: Lekért chunk ID-k listája
        k_values: K értékek listája (default: [1, 3, 5, 10])

    Returns:
        Dictionary a metrikákkal minden K értékre:
        {
            'precision_at_1': float,
            'precision_at_3': float,
            'precision_at_5': float,
            'precision_at_10': float,
            'recall_at_1': float,
            'recall_at_3': float,
            'recall_at_5': float,
            'recall_at_10': float,
        }

    Példa:
        >>> metrics = calculate_metrics_at_multiple_k(
        ...     relevant_chunk_ids=['abc-123'],
        ...     retrieved_chunk_ids=['def-456', 'abc-123', 'ghi-789', 'jkl-012', 'mno-345'],
        ...     k_values=[1, 3, 5]
        ... )
        >>> print(metrics)
        {
            'precision_at_1': 0.0,   # 0/1: nincs találat az első helyen
            'precision_at_3': 0.333, # 1/3: a 3-ból 1 releváns (2. helyen)
            'precision_at_5': 0.2,   # 1/5: az 5-ből 1 releváns
            'recall_at_1': 0.0,      # 0/1: nincs találat
            'recall_at_3': 1.0,      # 1/1: megtaláltuk az 1 releváns chunk-ot
            'recall_at_5': 1.0,      # 1/1: megtaláltuk
        }
    """
    metrics = {}

    for k in k_values:
        # Precision@K
        precision = calculate_precision_at_k(
            relevant_chunk_ids,
            retrieved_chunk_ids,
            k
        )
        metrics[f'precision_at_{k}'] = precision

        # Recall@K
        recall = calculate_recall_at_k(
            relevant_chunk_ids,
            retrieved_chunk_ids,
            k
        )
        metrics[f'recall_at_{k}'] = recall

    return metrics


def get_similarity_distributions(
    evaluation_results: List[Dict[str, Any]]
) -> Tuple[List[float], List[float]]:
    """
    Releváns és irreleváns chunk-ok similarity score listáinak kinyerése.

    Ez a függvény az értékelési eredményekből kigyűjti az összes similarity
    score-t, és két listába szétválogatja őket aszerint, hogy a chunk releváns
    volt-e (az eredeti chunk, amiből a kérdést generáltuk) vagy irreleváns.

    Ez az adatstruktúra hasznos:
    - Embedding quality vizualizációhoz (histogram)
    - Distribution analysis-hez
    - Statistical tests-ekhez

    Args:
        evaluation_results: Az értékelési eredmények listája

    Returns:
        Tuple két listával:
        - relevant_similarities: Releváns chunk-ok similarity score-jai
        - irrelevant_similarities: Irreleváns chunk-ok similarity score-jai

    Példa:
        >>> relevant, irrelevant = get_similarity_distributions(results)
        >>> print(f"Relevant avg: {np.mean(relevant):.4f}")
        Relevant avg: 0.5716
        >>> print(f"Irrelevant avg: {np.mean(irrelevant):.4f}")
        Irrelevant avg: 0.3542
    """
    relevant_similarities = []
    irrelevant_similarities = []

    for result in evaluation_results:
        # Eredeti chunk ID (amiből a kérdést generáltuk)
        original_chunk_id = result['chunk_id']

        # Lekért chunk-ok és similarity score-ok
        retrieved_chunk_ids = result['retrieved_chunk_ids']
        retrieved_similarities = result['retrieved_similarities']

        # Végigmegyünk minden lekért chunk-on
        for i, retrieved_id in enumerate(retrieved_chunk_ids):
            similarity = retrieved_similarities[i]

            if retrieved_id == original_chunk_id:
                # Ez a releváns chunk
                relevant_similarities.append(similarity)
            else:
                # Ez egy irreleváns chunk
                irrelevant_similarities.append(similarity)

    return relevant_similarities, irrelevant_similarities


def analyze_embedding_separation(
    evaluation_results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Embedding model minőségi metrikák: Relevant vs Irrelevant separation analysis.

    Ez a függvény azt méri, hogy mennyire jól különíti el az embedding model
    a releváns chunk-okat az irreleváns chunk-októl a similarity score alapján.

    A jó embedding model tulajdonságai:
    - Magas relevant_similarity_mean (pl. > 0.6)
    - Alacsony irrelevant_similarity_mean (pl. < 0.4)
    - Nagy separation_margin (pl. > 0.2)
    - Alacsony std dev-ek (konzisztens score-ok)

    Args:
        evaluation_results: Az értékelési eredmények listája

    Returns:
        Dictionary az embedding quality metrikákkal:
        - relevant_similarity_mean: Releváns chunk-ok átlagos similarity
        - irrelevant_similarity_mean: Irreleváns chunk-ok átlagos similarity
        - separation_margin: Elválasztás mértéke (relevant mean - irrelevant mean)
        - relevant_similarity_std: Releváns chunk-ok standard deviációja
        - irrelevant_similarity_std: Irreleváns chunk-ok standard deviációja
        - relevant_count: Releváns chunk-ok száma
        - irrelevant_count: Irreleváns chunk-ok száma

    Példa:
        >>> metrics = analyze_embedding_separation(results)
        >>> print(f"Separation margin: {metrics['separation_margin']:.4f}")
        Separation margin: 0.2174
        >>> print(f"Relevant mean: {metrics['relevant_similarity_mean']:.4f}")
        Relevant mean: 0.5716

    Interpretáció:
        - separation_margin > 0.3: Kiváló elválasztás
        - separation_margin > 0.2: Jó elválasztás
        - separation_margin > 0.1: Közepes elválasztás
        - separation_margin < 0.1: Gyenge elválasztás (embedding problémák)
    """
    # Similarity score-ok kinyerése
    relevant_sims, irrelevant_sims = get_similarity_distributions(evaluation_results)

    # Ha nincs adat, return default values
    if not relevant_sims:
        return {
            'relevant_similarity_mean': 0.0,
            'irrelevant_similarity_mean': 0.0,
            'separation_margin': 0.0,
            'relevant_similarity_std': 0.0,
            'irrelevant_similarity_std': 0.0,
            'relevant_count': 0,
            'irrelevant_count': 0
        }

    # Metrikák számítása
    return {
        'relevant_similarity_mean': float(np.mean(relevant_sims)),
        'irrelevant_similarity_mean': float(np.mean(irrelevant_sims)) if irrelevant_sims else 0.0,
        'separation_margin': float(np.mean(relevant_sims) - np.mean(irrelevant_sims)) if irrelevant_sims else float(np.mean(relevant_sims)),
        'relevant_similarity_std': float(np.std(relevant_sims)),
        'irrelevant_similarity_std': float(np.std(irrelevant_sims)) if irrelevant_sims else 0.0,
        'relevant_count': len(relevant_sims),
        'irrelevant_count': len(irrelevant_sims)
    }


def calculate_embedding_roc_auc(
    evaluation_results: List[Dict[str, Any]]
) -> float:
    """
    ROC-AUC score számítása az embedding model minőségének mérésére.

    Ez a metrika binary classification-ként kezeli a problémát:
    - Pozitív osztály (1): Releváns chunk
    - Negatív osztály (0): Irreleváns chunk
    - Score: Similarity score az embedding model-től

    A ROC-AUC (Receiver Operating Characteristic - Area Under Curve) azt méri,
    hogy mennyire jól tudja az embedding model a similarity score alapján
    elválasztani a releváns chunk-okat az irreleváns chunk-októl.

    Args:
        evaluation_results: Az értékelési eredmények listája

    Returns:
        ROC-AUC score (0.0-1.0 tartomány):
        - 1.0: Tökéletes elválasztás (minden releváns > minden irreleváns)
        - 0.9-1.0: Kiváló embedding model
        - 0.8-0.9: Jó embedding model
        - 0.7-0.8: Közepes embedding model
        - 0.5-0.7: Gyenge embedding model
        - 0.5: Random guess (embedding nem működik)
        - < 0.5: Rosszabb mint random (valami hiba van)

    Példa:
        >>> roc_auc = calculate_embedding_roc_auc(results)
        >>> print(f"ROC-AUC: {roc_auc:.4f}")
        ROC-AUC: 0.8234

    Note:
        Ha nincs mindkét osztály (pl. minden chunk releváns vagy minden irreleváns),
        akkor 0.0-t ad vissza, mert nem lehet ROC-AUC-t számolni.
    """
    y_true = []  # Binary labels: 1 = relevant, 0 = irrelevant
    y_score = []  # Similarity scores from embedding model

    for result in evaluation_results:
        # Eredeti chunk ID (amiből a kérdést generáltuk)
        original_chunk_id = result['chunk_id']

        # Lekért chunk-ok és similarity score-ok
        retrieved_chunk_ids = result['retrieved_chunk_ids']
        retrieved_similarities = result['retrieved_similarities']

        # Végigmegyünk minden lekért chunk-on
        for i, retrieved_id in enumerate(retrieved_chunk_ids):
            # Label: 1 ha releváns, 0 ha irreleváns
            if retrieved_id == original_chunk_id:
                y_true.append(1)  # Relevant
            else:
                y_true.append(0)  # Irrelevant

            # Score: similarity score
            y_score.append(retrieved_similarities[i])

    # Ha nincs mindkét osztály, nem lehet ROC-AUC-t számolni
    if len(set(y_true)) < 2:
        return 0.0

    # ROC-AUC számítás sklearn-kel
    try:
        roc_auc = roc_auc_score(y_true, y_score)
        return float(roc_auc)
    except Exception:
        # Ha bármilyen hiba van (pl. NaN értékek), return 0.0
        return 0.0


# ============================================================================
# CHUNK QUALITY METRICS
# ============================================================================

def calculate_chunk_size_consistency(
    evaluation_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Chunk Size Consistency Index (CSCI) számítása.

    A CSCI méri, hogy mennyire konzisztens a chunk méretek eloszlása.
    Magasabb érték = konzisztensebb chunk size-ok.

    Formula: CSCI = 1 - (std_dev_chunk_size / mean_chunk_size)

    Args:
        evaluation_results: Lista evaluation result dict-ekből

    Returns:
        Dictionary a következő mezőkkel:
        - csci: Chunk Size Consistency Index (0.0-1.0)
        - mean_chunk_size: Átlagos chunk méret (tokenekben)
        - std_chunk_size: Chunk méret szórása
        - min_chunk_size: Legkisebb chunk méret
        - max_chunk_size: Legnagyobb chunk méret
        - chunk_count: Chunk-ok száma

    Értelmezés:
        - CSCI >= 0.8: Nagyon konzisztens méretek
        - 0.6 <= CSCI < 0.8: Jó konzisztencia
        - 0.4 <= CSCI < 0.6: Közepes konzisztencia
        - CSCI < 0.4: Alacsony konzisztencia (nagy variancia)
    """
    if not evaluation_results:
        return {
            'csci': 0.0,
            'mean_chunk_size': 0.0,
            'std_chunk_size': 0.0,
            'min_chunk_size': 0,
            'max_chunk_size': 0,
            'chunk_count': 0
        }

    # Token count-ok összegyűjtése
    token_counts = [result['token_count'] for result in evaluation_results]

    # Statisztikák számítása
    mean_size = float(np.mean(token_counts))
    std_size = float(np.std(token_counts))

    # CSCI számítás
    if mean_size > 0:
        csci = 1.0 - (std_size / mean_size)
    else:
        csci = 0.0

    return {
        'csci': float(csci),
        'mean_chunk_size': mean_size,
        'std_chunk_size': std_size,
        'min_chunk_size': int(min(token_counts)),
        'max_chunk_size': int(max(token_counts)),
        'chunk_count': len(token_counts)
    }


def calculate_retrieval_by_chunk_size(
    evaluation_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Retrieval Quality per Chunk Size Bucket (RQCSB) analízis.

    Chunk-okat méret szerint bucket-ekbe osztja és minden bucket-re
    kiszámítja a retrieval metrikákat. Így megtalálható az optimális
    chunk size tartomány.

    Bucket definíciók:
        - tiny: 0-100 tokens
        - small: 100-300 tokens
        - medium: 300-600 tokens
        - large: 600-1000 tokens
        - xlarge: 1000+ tokens

    Args:
        evaluation_results: Lista evaluation result dict-ekből

    Returns:
        Dictionary bucket nevekkel, mindegyik tartalmazza:
        - hit_rate: Találati arány a bucket-ben
        - avg_rank: Átlagos rank
        - avg_similarity: Átlagos similarity score
        - first_position_accuracy: Első helyes találat aránya
        - chunk_count: Chunk-ok száma a bucket-ben
        - token_count_mean: Átlagos token szám
        - token_count_std: Token szám szórása
    """
    if not evaluation_results:
        return {}

    # Bucket határok definiálása
    buckets = {
        'tiny': (0, 100),
        'small': (100, 300),
        'medium': (300, 600),
        'large': (600, 1000),
        'xlarge': (1000, float('inf'))
    }

    # Eredmények bucket-ek szerint csoportosítása
    bucket_data = {name: [] for name in buckets.keys()}

    for result in evaluation_results:
        token_count = result['token_count']

        # Megkeressük a megfelelő bucket-et
        for bucket_name, (min_tokens, max_tokens) in buckets.items():
            if min_tokens <= token_count < max_tokens:
                bucket_data[bucket_name].append(result)
                break

    # Metrikák számítása bucket-enként
    bucket_metrics = {}

    for bucket_name, results in bucket_data.items():
        if not results:
            # Üres bucket - skip
            continue

        # Token statisztikák
        token_counts = [r['token_count'] for r in results]

        # Retrieval metrikák (metrics dict-ből)
        hit_rates = [r['metrics']['hit_rate_at_k'] for r in results]
        ranks = [r['metrics']['rank'] for r in results if r['metrics']['rank'] is not None]
        similarities = [r['metrics']['similarity_score'] for r in results if r['metrics']['similarity_score'] is not None]
        first_pos = [r['metrics']['first_position_accuracy'] for r in results]

        bucket_metrics[bucket_name] = {
            'hit_rate': float(np.mean(hit_rates)) if hit_rates else 0.0,
            'avg_rank': float(np.mean(ranks)) if ranks else None,
            'avg_similarity': float(np.mean(similarities)) if similarities else None,
            'first_position_accuracy': float(np.mean(first_pos)) if first_pos else 0.0,
            'chunk_count': len(results),
            'token_count_mean': float(np.mean(token_counts)),
            'token_count_std': float(np.std(token_counts))
        }

    return bucket_metrics


def calculate_position_stability(
    evaluation_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Position Stability Score (PSS) számítása chunk-onként.

    A PSS méri, hogy egy adott chunk mennyire "stabilán" jelenik meg
    különböző query-k esetén. Ha egy chunk mindig ugyanazon a pozíción
    van a ranking-ben, akkor stabil (magas PSS). Ha a pozíciója nagyon
    változó, akkor instabil (alacsony PSS).

    Formula: PSS[chunk_id] = 1 - (std_dev_rank / mean_rank)

    Args:
        evaluation_results: Lista evaluation result dict-ekből

    Returns:
        Dictionary a következő mezőkkel:
        - top_stable_chunks: Top 10 legstabilabb chunk (magas PSS)
        - top_unstable_chunks: Top 10 leginstabilabb chunk (alacsony PSS)
        - overall_stability_mean: Összes chunk PSS átlaga
        - overall_stability_std: PSS szórása
        - chunks_analyzed: Vizsgált chunk-ok száma (min 2 appearance)

    Értelmezés:
        - PSS >= 0.8: Nagyon stabil chunk (megbízható retrieval)
        - 0.6 <= PSS < 0.8: Stabil chunk
        - 0.4 <= PSS < 0.6: Közepes stabilitás
        - PSS < 0.4: Instabil chunk (változó ranking)
    """
    # Chunk megjelenések gyűjtése
    chunk_appearances = {}

    for result in evaluation_results:
        retrieved_chunk_ids = result['retrieved_chunk_ids']

        # Végigmegyünk a lekért chunk-okon és rögzítjük a ranket
        for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
            if chunk_id not in chunk_appearances:
                chunk_appearances[chunk_id] = []
            chunk_appearances[chunk_id].append(rank)

    # PSS számítása chunk-onként
    stability_scores = []

    for chunk_id, ranks in chunk_appearances.items():
        # Csak olyan chunk-okat vizsgálunk, amelyek legalább 2x megjelentek
        if len(ranks) >= 2:
            mean_rank = float(np.mean(ranks))
            std_rank = float(np.std(ranks))

            # PSS formula
            if mean_rank > 0:
                pss = 1.0 - (std_rank / mean_rank)
            else:
                pss = 0.0

            stability_scores.append({
                'chunk_id': chunk_id,
                'appearances': len(ranks),
                'mean_rank': mean_rank,
                'std_rank': std_rank,
                'pss': pss
            })

    # Rendezés PSS szerint
    stability_scores.sort(key=lambda x: x['pss'], reverse=True)

    # Top 10 legstabilabb és leginstabilabb
    top_stable = stability_scores[:10] if len(stability_scores) >= 10 else stability_scores
    top_unstable = stability_scores[-10:][::-1] if len(stability_scores) >= 10 else []

    # Overall statisztikák
    if stability_scores:
        all_pss = [s['pss'] for s in stability_scores]
        overall_mean = float(np.mean(all_pss))
        overall_std = float(np.std(all_pss))
    else:
        overall_mean = 0.0
        overall_std = 0.0

    return {
        'top_stable_chunks': top_stable,
        'top_unstable_chunks': top_unstable,
        'overall_stability_mean': overall_mean,
        'overall_stability_std': overall_std,
        'chunks_analyzed': len(stability_scores)
    }
