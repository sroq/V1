"""
RAG értékelési eredmények elemzése és vizualizálása.

Ez a script az értékelési folyamat HARMADIK LÉPÉSE:
1. Betölti az értékelési eredményeket JSON fájlból
2. Kiszámítja az aggregált metrikákat
3. Chunking stratégia szerint bontott elemzést készít
4. Vizualizációkat hoz létre (matplotlib)
5. Részletes riportokat ment (JSON, CSV, PNG)

Futtatás:
    python3 analyze_results.py

Input:
    data/evaluation_results.json (evaluate_rag.py output-ja)

Output:
    results/YYYYMMDD_HHMMSS/
    ├── summary.json                 # Aggregált metrikák
    ├── detailed_results.csv         # Minden kérdés részletei
    ├── metrics_by_strategy.csv      # Stratégiánként bontva
    └── plots/                       # Vizualizációk
        ├── overall_metrics.png
        ├── rank_distribution.png
        ├── similarity_distribution.png
        └── metrics_by_strategy.png
"""
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    aggregate_metrics,
    format_results,
    results_to_dataframe,
    calculate_metrics_by_strategy,
    get_similarity_distributions
)
from config import EVALUATION_RESULTS_FILE, RESULTS_DIR
import sys


def create_visualizations(df: pd.DataFrame, output_dir: str, evaluation_results: list, aggregated_metrics: dict):
    """
    Vizualizációk létrehozása az értékelési eredményekből.

    7 különböző plot készül:
    1. Overall binary metrics: Hit Rate@K és First Position Accuracy bar chart
    2. Rank distribution: Hányadik helyen volt az eredeti chunk
    3. Similarity distribution: Similarity score-ok eloszlása
    4. Metrics by strategy: Teljesítmény chunking stratégiánként
    5. Precision/Recall curves: Precision@K és Recall@K különböző K értékekre
    6. MRR comparison: MRR összehasonlítás stratégiánként (ha van több stratégia)
    7. Embedding quality distribution: Relevant vs Irrelevant similarity distributions (ÚJ)

    Args:
        df: DataFrame az értékelési eredményekkel
        output_dir: Könyvtár, ahova a PNG fájlokat mentjük
        evaluation_results: Az evaluation results listája (similarity distributions-hez szükséges)
    """
    # Matplotlib/seaborn stílus beállítása
    sns.set_style("whitegrid")

    # ========================================================================
    # 1. OVERALL BINARY METRICS BAR CHART (frissített nevek)
    # ========================================================================
    # Hit Rate@K és First Position Accuracy összehasonlítása
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_summary = pd.DataFrame({
        'Metric': ['Hit Rate@K', 'First Position Accuracy'],
        'Percentage': [
            df['hit_rate_at_k'].mean() * 100,           # Új név
            df['first_position_accuracy'].mean() * 100   # Új név
        ]
    })
    sns.barplot(data=metrics_summary, x='Metric', y='Percentage', ax=ax, hue='Metric', palette='viridis', legend=False)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('RAG Retrieval Performance - Binary Metrics')
    ax.set_ylim(0, 100)
    # Értékek kiírása a bárok tetejére
    for i, v in enumerate(metrics_summary['Percentage']):
        ax.text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_metrics.png'), dpi=300)
    plt.close()

    # ========================================================================
    # 2. RANK DISTRIBUTION
    # ========================================================================
    # Ha az eredeti chunk megtalálható volt, hányadik helyen volt?
    if df['rank'].notna().any():  # Van legalább egy nem-None rank
        fig, ax = plt.subplots(figsize=(10, 6))
        rank_counts = df[df['rank'].notna()]['rank'].value_counts().sort_index()
        ax.bar(rank_counts.index, rank_counts.values, color=plt.cm.coolwarm(np.linspace(0, 1, len(rank_counts))))
        ax.set_xlabel('Rank Position')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Original Chunk Rank in Retrieved Results')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rank_distribution.png'), dpi=300)
        plt.close()

    # ========================================================================
    # 3. SIMILARITY SCORE DISTRIBUTION
    # ========================================================================
    # Similarity score-ok eloszlása (csak ahol megtalálható volt)
    if df['similarity_score'].notna().any():
        fig, ax = plt.subplots(figsize=(10, 6))
        df[df['similarity_score'].notna()]['similarity_score'].hist(bins=30, ax=ax, edgecolor='black')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Similarity Scores for Original Chunks')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'), dpi=300)
        plt.close()

    # ========================================================================
    # 4. METRICS BY CHUNKING STRATEGY (frissített nevek)
    # ========================================================================
    # Ha több chunking stratégia van, összehasonlítjuk őket
    if 'chunking_strategy' in df.columns and df['chunking_strategy'].nunique() > 1:
        strategy_metrics = df.groupby('chunking_strategy').agg({
            'hit_rate_at_k': 'mean',
            'first_position_accuracy': 'mean'
        }).reset_index()

        # Wide format -> long format (melt) a seaborn számára
        strategy_metrics_melted = strategy_metrics.melt(
            id_vars='chunking_strategy',
            value_vars=['hit_rate_at_k', 'first_position_accuracy'],
            var_name='Metric',
            value_name='Score'
        )
        strategy_metrics_melted['Score'] *= 100  # Százalékra váltás

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=strategy_metrics_melted,
            x='chunking_strategy',
            y='Score',
            hue='Metric',
            ax=ax,
            palette='Set2'
        )
        ax.set_xlabel('Chunking Strategy')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Binary Metrics by Chunking Strategy')
        ax.set_ylim(0, 100)
        ax.legend(title='Metric', labels=['Hit Rate@K', 'First Position Accuracy'])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_by_strategy.png'), dpi=300)
        plt.close()

    # ========================================================================
    # 5. PRECISION/RECALL CURVES @K (ÚJ)
    # ========================================================================
    # Precision@K és Recall@K vizualizáció K=1,3,5,10 értékekre
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Precision@K értékek
    k_values = [1, 3, 5, 10]
    precision_values = [
        df['precision_at_1'].mean(),
        df['precision_at_3'].mean(),
        df['precision_at_5'].mean(),
        df['precision_at_10'].mean()
    ]
    recall_values = [
        df['recall_at_1'].mean(),
        df['recall_at_3'].mean(),
        df['recall_at_5'].mean(),
        df['recall_at_10'].mean()
    ]

    # Bal oldali plot: Precision@K
    ax1.plot(k_values, precision_values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('K (Top-K Results)', fontsize=12)
    ax1.set_ylabel('Precision@K', fontsize=12)
    ax1.set_title('Precision@K - Classical IR Metric', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    # Értékek kiírása
    for k, p in zip(k_values, precision_values):
        ax1.text(k, p + 0.03, f'{p:.3f}', ha='center', fontsize=10)

    # Jobb oldali plot: Recall@K
    ax2.plot(k_values, recall_values, marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('K (Top-K Results)', fontsize=12)
    ax2.set_ylabel('Recall@K', fontsize=12)
    ax2.set_title('Recall@K - Classical IR Metric', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    # Értékek kiírása
    for k, r in zip(k_values, recall_values):
        ax2.text(k, r + 0.03, f'{r:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), dpi=300)
    plt.close()

    # ========================================================================
    # 6. MRR COMPARISON BY STRATEGY (ÚJ) - csak ha van több stratégia
    # ========================================================================
    if 'chunking_strategy' in df.columns and df['chunking_strategy'].nunique() > 1:
        # MRR kiszámítása stratégiánként
        # MRR = átlag(1/rank) minden kérdésre stratégiánként
        def calculate_mrr_per_strategy(group):
            """Reciprocal Rank számítása egy stratégia csoporthoz"""
            reciprocal_ranks = []
            for rank in group['rank']:
                if rank is not None:
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)
            return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

        strategy_mrr = df.groupby('chunking_strategy').apply(
            calculate_mrr_per_strategy
        ).reset_index()
        strategy_mrr.columns = ['chunking_strategy', 'MRR']

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=strategy_mrr,
            x='chunking_strategy',
            y='MRR',
            ax=ax,
            palette='rocket'
        )
        ax.set_xlabel('Chunking Strategy', fontsize=12)
        ax.set_ylabel('Mean Reciprocal Rank (MRR)', fontsize=12)
        ax.set_title('MRR Comparison by Chunking Strategy', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        # Értékek kiírása a bárok tetejére
        for i, row in strategy_mrr.iterrows():
            ax.text(i, row['MRR'] + 0.02, f'{row["MRR"]:.3f}', ha='center', fontsize=10, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mrr_comparison.png'), dpi=300)
        plt.close()

    # ========================================================================
    # 7. EMBEDDING QUALITY DISTRIBUTION (ÚJ)
    # ========================================================================
    # Relevant vs Irrelevant similarity score distributions
    relevant_sims, irrelevant_sims = get_similarity_distributions(evaluation_results)

    if relevant_sims and irrelevant_sims:  # Ha van adat
        fig, ax = plt.subplots(figsize=(12, 6))

        # Histogram - Relevant chunks (kék)
        ax.hist(relevant_sims, bins=30, alpha=0.6, label='Relevant Chunks',
                color='#2E86AB', edgecolor='black')

        # Histogram - Irrelevant chunks (lila)
        ax.hist(irrelevant_sims, bins=30, alpha=0.6, label='Irrelevant Chunks',
                color='#A23B72', edgecolor='black')

        # Átlagok kijelölése függőleges vonalakkal
        relevant_mean = sum(relevant_sims) / len(relevant_sims)
        irrelevant_mean = sum(irrelevant_sims) / len(irrelevant_sims)

        ax.axvline(relevant_mean, color='#2E86AB', linestyle='--', linewidth=2,
                   label=f'Relevant Mean: {relevant_mean:.4f}')
        ax.axvline(irrelevant_mean, color='#A23B72', linestyle='--', linewidth=2,
                   label=f'Irrelevant Mean: {irrelevant_mean:.4f}')

        # Separation margin annotation
        separation = relevant_mean - irrelevant_mean
        mid_point = (relevant_mean + irrelevant_mean) / 2
        ax.annotate(f'Separation: {separation:.4f}',
                    xy=(mid_point, ax.get_ylim()[1] * 0.9),
                    ha='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

        ax.set_xlabel('Cosine Similarity Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Embedding Quality: Relevant vs Irrelevant Similarity Distribution',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'embedding_quality_distribution.png'), dpi=300)
        plt.close()

    # ========================================================================
    # 8. CHUNK SIZE DISTRIBUTION (ÚJ)
    # ========================================================================
    # Token count distribution vizualizáció
    token_counts = df['token_count'].values

    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogram
    n, bins, patches = ax.hist(token_counts, bins=30, alpha=0.7, color='#3A86FF', edgecolor='black')

    # Mean és median vonalak
    mean_tokens = token_counts.mean()
    median_tokens = np.median(token_counts)

    ax.axvline(mean_tokens, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_tokens:.1f} tokens')
    ax.axvline(median_tokens, color='green', linestyle='--', linewidth=2,
               label=f'Median: {median_tokens:.1f} tokens')

    # Size bucket határok kijelölése
    bucket_boundaries = [100, 300, 600, 1000]
    for boundary in bucket_boundaries:
        ax.axvline(boundary, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Bucket labels hozzáadása
    ax.text(50, ax.get_ylim()[1] * 0.95, 'Tiny', ha='center', fontsize=9, alpha=0.7)
    ax.text(200, ax.get_ylim()[1] * 0.95, 'Small', ha='center', fontsize=9, alpha=0.7)
    ax.text(450, ax.get_ylim()[1] * 0.95, 'Medium', ha='center', fontsize=9, alpha=0.7)
    ax.text(800, ax.get_ylim()[1] * 0.95, 'Large', ha='center', fontsize=9, alpha=0.7)
    if token_counts.max() > 1000:
        ax.text(min(1200, token_counts.max() - 100), ax.get_ylim()[1] * 0.95, 'XLarge',
                ha='center', fontsize=9, alpha=0.7)

    # Stats annotation
    stats_text = f'Std: {token_counts.std():.1f}\nMin: {token_counts.min()}\nMax: {token_counts.max()}'
    ax.text(0.98, 0.75, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Chunk Size (tokens)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Chunk Size Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chunk_size_distribution.png'), dpi=300)
    plt.close()

    # ========================================================================
    # 9. RETRIEVAL QUALITY BY CHUNK SIZE BUCKET (RQCSB) HEATMAP (ÚJ)
    # ========================================================================
    # Ha van chunk_quality metrics az aggregated_metrics-ben
    if 'chunk_quality' in aggregated_metrics:
        rbs = aggregated_metrics['chunk_quality']['retrieval_by_size_bucket']

        if rbs:
            # Adatok előkészítése heatmap-hez
            bucket_order = ['tiny', 'small', 'medium', 'large', 'xlarge']
            metric_names = ['Hit Rate', 'Avg Rank', 'Avg Similarity', 'First Pos Acc']

            # Mátrix létrehozása
            data_matrix = []
            available_buckets = []

            for bucket_name in bucket_order:
                if bucket_name in rbs:
                    bucket = rbs[bucket_name]
                    row = [
                        bucket['hit_rate'],
                        1.0 / bucket['avg_rank'] if bucket['avg_rank'] else 0.0,  # Normalizált rank (magasabb=jobb)
                        bucket['avg_similarity'] if bucket['avg_similarity'] else 0.0,
                        bucket['first_position_accuracy']
                    ]
                    data_matrix.append(row)
                    available_buckets.append(bucket_name.capitalize())

            if data_matrix:
                data_matrix = np.array(data_matrix)

                # Heatmap készítése
                fig, ax = plt.subplots(figsize=(10, 6))

                # Normalizáljuk az adatokat oszloponként (0-1 skálára)
                data_normalized = data_matrix.copy()
                for col in range(data_matrix.shape[1]):
                    col_data = data_matrix[:, col]
                    min_val = col_data.min()
                    max_val = col_data.max()
                    if max_val > min_val:
                        data_normalized[:, col] = (col_data - min_val) / (max_val - min_val)

                # Heatmap
                im = ax.imshow(data_normalized.T, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)

                # Tengelyek beállítása
                ax.set_xticks(np.arange(len(available_buckets)))
                ax.set_yticks(np.arange(len(metric_names)))
                ax.set_xticklabels(available_buckets, fontsize=11)
                ax.set_yticklabels(metric_names, fontsize=11)

                # Értékek kiírása cellákba
                for i in range(len(available_buckets)):
                    for j in range(len(metric_names)):
                        # Original érték (nem normalizált)
                        original_val = data_matrix[i, j]

                        # Formázás
                        if j == 0 or j == 3:  # Hit Rate, First Pos Acc - százalék
                            text_val = f'{original_val:.1%}'
                        elif j == 1:  # Avg Rank (inverted)
                            actual_rank = 1.0 / original_val if original_val > 0 else float('inf')
                            text_val = f'{actual_rank:.2f}' if actual_rank != float('inf') else 'N/A'
                        else:  # Avg Similarity
                            text_val = f'{original_val:.3f}'

                        # Text color based on background
                        text_color = 'white' if data_normalized[i, j] > 0.5 else 'black'
                        ax.text(i, j, text_val, ha='center', va='center',
                                color=text_color, fontsize=10, fontweight='bold')

                # Colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Normalized Score (0-1)', rotation=270, labelpad=20, fontsize=11)

                ax.set_title('Retrieval Quality by Chunk Size Bucket (RQCSB)',
                             fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel('Chunk Size Bucket', fontsize=12)
                ax.set_ylabel('Metric', fontsize=12)

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'rqcsb_heatmap.png'), dpi=300)
                plt.close()

    print(f"✓ Saved visualizations to {output_dir}/")


def main():
    """Fő függvény: eredmények elemzése és vizualizálása."""

    # ========================================================================
    # 1. KEZDŐ BANNER
    # ========================================================================
    print("=" * 80)
    print("RAG EVALUATION - RESULTS ANALYSIS")
    print("=" * 80)
    print()

    # ========================================================================
    # 2. ÉRTÉKELÉSI EREDMÉNYEK BETÖLTÉSE
    # ========================================================================
    # Ellenőrizzük, hogy létezik-e az eredmények fájlja
    if not os.path.exists(EVALUATION_RESULTS_FILE):
        print(f"✗ Evaluation results file not found: {EVALUATION_RESULTS_FILE}")
        print("  Please run evaluate_rag.py first.")
        sys.exit(1)

    # Eredmények betöltése JSON fájlból
    print(f"Loading evaluation results from {EVALUATION_RESULTS_FILE}...")
    try:
        with open(EVALUATION_RESULTS_FILE, 'r', encoding='utf-8') as f:
            evaluation_results = json.load(f)
        print(f"✓ Loaded {len(evaluation_results)} evaluation results")
    except Exception as e:
        print(f"✗ Failed to load evaluation results: {e}")
        sys.exit(1)

    # Ha nincs eredmény, kilépünk
    if len(evaluation_results) == 0:
        print("\n⚠ No evaluation results found.")
        sys.exit(0)

    # ========================================================================
    # 3. AGGREGÁLT METRIKÁK SZÁMÍTÁSA
    # ========================================================================
    print("\nCalculating aggregated metrics...")
    # aggregate_metrics() kiszámítja:
    # - Top-K Recall %
    # - Top-1 Precision %
    # - Average rank
    # - Average similarity
    # - Found count és %
    aggregated = aggregate_metrics(evaluation_results)

    # ========================================================================
    # 4. EREDMÉNYEK MEGJELENÍTÉSE KONZOLRA
    # ========================================================================
    # format_results() szépen formázott string-et készít
    print("\n" + format_results(evaluation_results, aggregated))

    # ========================================================================
    # 5. DATAFRAME KONVERZIÓ ÉS RÉSZLETES ELEMZÉS
    # ========================================================================
    print("\nGenerating detailed analysis...")
    # results_to_dataframe() átalakítja pandas DataFrame-mé
    # Ez lehetővé teszi a pandas-os elemzést (groupby, filtering, stb.)
    df = results_to_dataframe(evaluation_results)

    # Ha több chunking stratégia van, stratégiánként is elemzünk
    if 'chunking_strategy' in df.columns and df['chunking_strategy'].nunique() > 1:
        print("\nMetrics by Chunking Strategy:")
        print("=" * 80)
        strategy_metrics = calculate_metrics_by_strategy(df)
        print(strategy_metrics.to_string())
        print()

    # ========================================================================
    # 6. TIMESTAMPED OUTPUT KÖNYVTÁR LÉTREHOZÁSA
    # ========================================================================
    # Minden futtatáshoz egy külön timestamped könyvtárat hozunk létre
    # Formátum: YYYYMMDD_HHMMSS (pl. 20250108_143025)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # ========================================================================
    # 7. SUMMARY JSON MENTÉSE
    # ========================================================================
    summary_file = os.path.join(output_dir, 'summary.json')
    print(f"\nSaving summary to {summary_file}...")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    print(f"✓ Summary saved")

    # ========================================================================
    # 8. DETAILED RESULTS CSV MENTÉSE
    # ========================================================================
    csv_file = os.path.join(output_dir, 'detailed_results.csv')
    print(f"Saving detailed results to {csv_file}...")
    df.to_csv(csv_file, index=False)
    print(f"✓ Detailed results saved")

    # ========================================================================
    # 9. STRATEGY METRICS CSV MENTÉSE (HA VAN)
    # ========================================================================
    if 'chunking_strategy' in df.columns and df['chunking_strategy'].nunique() > 1:
        strategy_file = os.path.join(output_dir, 'metrics_by_strategy.csv')
        print(f"Saving strategy metrics to {strategy_file}...")
        strategy_metrics = calculate_metrics_by_strategy(df)
        strategy_metrics.to_csv(strategy_file)
        print(f"✓ Strategy metrics saved")

    # ========================================================================
    # 10. VIZUALIZÁCIÓK KÉSZÍTÉSE
    # ========================================================================
    print("\nCreating visualizations...")
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    create_visualizations(df, plots_dir, evaluation_results, aggregated)

    # ========================================================================
    # 11. ZÁRÓ ÖSSZEFOGLALÓ
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"  - summary.json: Aggregated metrics")
    print(f"  - detailed_results.csv: All evaluation results")
    if 'chunking_strategy' in df.columns and df['chunking_strategy'].nunique() > 1:
        print(f"  - metrics_by_strategy.csv: Metrics by chunking strategy")
    print(f"  - plots/: Visualization plots")
    print()


# ============================================================================
# SCRIPT BELÉPÉSI PONT
# ============================================================================
if __name__ == '__main__':
    main()
