"""
Multi-Turn Evaluation Results Visualization

Ez a modul felelős a multi-turn evaluation eredmények vizualizációjáért.
Különböző grafikonokat és elemzéseket készít a batch eredményekből.

Generált grafikonok:
1. Overall Scores (5 dimenzió összehasonlítása)
2. Goal Achievement Rate by Persona
3. Goal Achievement Rate by Difficulty
4. Turn Count Distribution
5. Duration Distribution
6. Score Distribution by Dimension (box plot)
7. Success Rate Heatmap (persona x goal)
8. User Experience Breakdown
9. Efficiency vs Quality Scatter
10. Dimension Correlation Heatmap
11. API Latency Distribution (performance)
12. Time To First Token (TTFT) Distribution (performance)
13. Tokens per Second Distribution (performance)
14. Performance Metrics Heatmap (persona x goal)

Használat:
    python3 visualize_results.py results/batch_summary_TIMESTAMP.json
"""

import json
import os
import argparse
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn stílus beállítása
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class MultiTurnVisualizer:
    """
    Multi-turn evaluation eredmények vizualizálója.
    """

    def __init__(self, batch_summary: Dict[str, Any], output_dir: str):
        """
        Inicializálás.

        Args:
            batch_summary: Batch summary JSON (loaded dict)
            output_dir: Kimeneti könyvtár grafikonoknak
        """
        self.batch_summary = batch_summary
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # DataFrame készítése az eredményekből
        self.df = self._prepare_dataframe()

        # Per-turn performance data for distributions
        self.per_turn_perf_data = self._extract_per_turn_performance()

    def _prepare_dataframe(self) -> pd.DataFrame:
        """
        Pandas DataFrame készítése a batch eredményekből.

        Returns:
            DataFrame az összes eredménnyel
        """
        rows = []

        for result in self.batch_summary.get("results", []):
            if not result.get("success"):
                continue

            evaluation = result.get("evaluation", {})
            metadata = result.get("metadata", {})

            # Performance metrics extraction
            performance = metadata.get("performance", {})

            row = {
                "persona": result["persona"],
                "goal": result["goal"],
                "overall_score": evaluation.get("overall_score", 0),
                "goal_achievement_score": evaluation.get("goal_achievement", {}).get("achievement_score", 0),
                "conversation_quality_score": evaluation.get("conversation_quality", {}).get("overall_quality_score", 0),
                "response_relevance_score": evaluation.get("response_relevance", {}).get("overall_relevance_score", 0),
                "user_experience_score": evaluation.get("user_experience", {}).get("ux_score", 0),
                "efficiency_score": evaluation.get("efficiency", {}).get("efficiency_score", 0),
                "goal_reached": evaluation.get("goal_achievement", {}).get("goal_reached", False),
                "turns": metadata.get("turns", 0),
                "duration_s": metadata.get("duration_s", 0),
                "frustration_level": metadata.get("user_progress", {}).get("frustration_level", 0),
                "progress_percentage": metadata.get("user_progress", {}).get("progress_percentage", 0),
                "ux_overall": evaluation.get("user_experience", {}).get("overall_experience", "neutral"),
                # Performance metrics
                "avg_api_latency_ms": performance.get("avg_api_latency_ms"),
                "avg_ttft_ms": performance.get("avg_ttft_ms"),
                "avg_tokens_per_second": performance.get("avg_tokens_per_second"),
                "total_tokens": performance.get("total_tokens", 0),
                "total_retries": performance.get("total_retries", 0),
            }

            # Goal difficulty kinyerése (easy, medium, hard)
            goal_name = result["goal"]
            if "identity" in goal_name or "baloo" in goal_name:
                row["difficulty"] = "easy"
            elif "journey" in goal_name or "comparison" in goal_name:
                row["difficulty"] = "hard"
            else:
                row["difficulty"] = "medium"

            rows.append(row)

        return pd.DataFrame(rows)

    def _extract_per_turn_performance(self) -> Dict[str, List[float]]:
        """
        Per-turn performance adatok kinyerése minden beszélgetésből.

        Returns:
            Dict with lists of performance metrics across all turns
        """
        api_latencies = []
        ttfts = []
        tokens_per_sec = []

        for result in self.batch_summary.get("results", []):
            if not result.get("success"):
                continue

            per_turn_perf = result.get("metadata", {}).get("per_turn_performance", [])

            for perf in per_turn_perf:
                if perf.get("api_latency_ms") is not None:
                    api_latencies.append(perf["api_latency_ms"])
                if perf.get("ttft_ms") is not None:
                    ttfts.append(perf["ttft_ms"])
                if perf.get("tokens_per_second") is not None:
                    tokens_per_sec.append(perf["tokens_per_second"])

        return {
            "api_latency_ms": api_latencies,
            "ttft_ms": ttfts,
            "tokens_per_second": tokens_per_sec
        }

    def visualize_all(self):
        """
        Összes grafikon generálása.
        """
        print("Generating visualizations...")

        # 1. Overall Scores
        print("  - Overall scores by dimension...")
        self.plot_overall_scores()

        # 2. Goal Achievement by Persona
        print("  - Goal achievement by persona...")
        self.plot_goal_achievement_by_persona()

        # 3. Goal Achievement by Difficulty
        print("  - Goal achievement by difficulty...")
        self.plot_goal_achievement_by_difficulty()

        # 4. Turn Count Distribution
        print("  - Turn count distribution...")
        self.plot_turn_distribution()

        # 5. Duration Distribution
        print("  - Duration distribution...")
        self.plot_duration_distribution()

        # 6. Score Distribution by Dimension
        print("  - Score distribution by dimension...")
        self.plot_score_distributions()

        # 7. Success Rate Heatmap
        print("  - Success rate heatmap...")
        self.plot_success_heatmap()

        # 8. User Experience Breakdown
        print("  - User experience breakdown...")
        self.plot_ux_breakdown()

        # 9. Efficiency vs Quality
        print("  - Efficiency vs quality scatter...")
        self.plot_efficiency_vs_quality()

        # 10. Dimension Correlation
        print("  - Dimension correlation heatmap...")
        self.plot_dimension_correlation()

        # 11. API Latency Distribution (Performance)
        print("  - API latency distribution...")
        self.plot_api_latency_distribution()

        # 12. TTFT Distribution (Performance)
        print("  - TTFT distribution...")
        self.plot_ttft_distribution()

        # 13. Tokens/sec Distribution (Performance)
        print("  - Tokens per second distribution...")
        self.plot_tokens_per_second_distribution()

        # 14. Performance Metrics Heatmap
        print("  - Performance metrics heatmap...")
        self.plot_performance_heatmap()

        print(f"\nAll visualizations saved to: {self.output_dir}")

    def plot_overall_scores(self):
        """
        1. Overall Scores - 5 dimenzió átlaga bar chart.
        """
        dimensions = [
            "goal_achievement_score",
            "conversation_quality_score",
            "response_relevance_score",
            "user_experience_score",
            "efficiency_score"
        ]

        labels = [
            "Goal Achievement",
            "Conversation Quality",
            "Response Relevance",
            "User Experience",
            "Efficiency"
        ]

        scores = [self.df[dim].mean() for dim in dimensions]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, scores, color=sns.color_palette("husl", 5))

        # Értékek a bar-ok tetején
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.1f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Score (0-100)')
        ax.set_title('Overall Scores by Evaluation Dimension', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='50% threshold')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_overall_scores.png'), dpi=300)
        plt.close()

    def plot_goal_achievement_by_persona(self):
        """
        2. Goal Achievement Rate by Persona.
        """
        persona_stats = self.df.groupby('persona').agg({
            'goal_reached': 'mean',
            'goal_achievement_score': 'mean'
        }).reset_index()

        persona_stats['achievement_rate'] = persona_stats['goal_reached'] * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(persona_stats['persona'], persona_stats['achievement_rate'],
                     color=sns.color_palette("Set2", len(persona_stats)))

        for bar, rate in zip(bars, persona_stats['achievement_rate']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Goal Achievement Rate (%)')
        ax.set_title('Goal Achievement Rate by Persona', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 110)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '02_goal_achievement_by_persona.png'), dpi=300)
        plt.close()

    def plot_goal_achievement_by_difficulty(self):
        """
        3. Goal Achievement Rate by Difficulty.
        """
        difficulty_stats = self.df.groupby('difficulty').agg({
            'goal_reached': 'mean',
            'goal_achievement_score': 'mean',
            'turns': 'mean'
        }).reset_index()

        difficulty_stats['achievement_rate'] = difficulty_stats['goal_reached'] * 100

        # Rendezés: easy, medium, hard
        difficulty_order = ['easy', 'medium', 'hard']
        difficulty_stats['difficulty'] = pd.Categorical(
            difficulty_stats['difficulty'],
            categories=difficulty_order,
            ordered=True
        )
        difficulty_stats = difficulty_stats.sort_values('difficulty')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Achievement rate
        bars1 = ax1.bar(difficulty_stats['difficulty'], difficulty_stats['achievement_rate'],
                       color=['green', 'orange', 'red'])
        for bar, rate in zip(bars1, difficulty_stats['achievement_rate']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax1.set_ylabel('Goal Achievement Rate (%)')
        ax1.set_title('Achievement Rate by Difficulty')
        ax1.set_ylim(0, 110)

        # Average turns
        bars2 = ax2.bar(difficulty_stats['difficulty'], difficulty_stats['turns'],
                       color=['green', 'orange', 'red'])
        for bar, turns in zip(bars2, difficulty_stats['turns']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{turns:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax2.set_ylabel('Average Turns')
        ax2.set_title('Average Turn Count by Difficulty')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_goal_achievement_by_difficulty.png'), dpi=300)
        plt.close()

    def plot_turn_distribution(self):
        """
        4. Turn Count Distribution - Histogram.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(self.df['turns'], bins=range(1, int(self.df['turns'].max()) + 2),
               color='skyblue', edgecolor='black', alpha=0.7)

        ax.axvline(self.df['turns'].mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {self.df["turns"].mean():.1f}')
        ax.axvline(self.df['turns'].median(), color='green', linestyle='--',
                  linewidth=2, label=f'Median: {self.df["turns"].median():.1f}')

        ax.set_xlabel('Number of Turns')
        ax.set_ylabel('Frequency')
        ax.set_title('Turn Count Distribution', fontsize=14, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '04_turn_distribution.png'), dpi=300)
        plt.close()

    def plot_duration_distribution(self):
        """
        5. Duration Distribution - Histogram.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(self.df['duration_s'], bins=20, color='coral', edgecolor='black', alpha=0.7)

        ax.axvline(self.df['duration_s'].mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {self.df["duration_s"].mean():.1f}s')
        ax.axvline(self.df['duration_s'].median(), color='green', linestyle='--',
                  linewidth=2, label=f'Median: {self.df["duration_s"].median():.1f}s')

        ax.set_xlabel('Duration (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Conversation Duration Distribution', fontsize=14, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '05_duration_distribution.png'), dpi=300)
        plt.close()

    def plot_score_distributions(self):
        """
        6. Score Distribution by Dimension - Box Plot.
        """
        dimensions = [
            "goal_achievement_score",
            "conversation_quality_score",
            "response_relevance_score",
            "user_experience_score",
            "efficiency_score"
        ]

        labels = [
            "Goal\nAchievement",
            "Conversation\nQuality",
            "Response\nRelevance",
            "User\nExperience",
            "Efficiency"
        ]

        data = [self.df[dim].dropna() for dim in dimensions]

        fig, ax = plt.subplots(figsize=(12, 7))

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        ax.set_ylabel('Score (0-100)')
        ax.set_title('Score Distribution by Evaluation Dimension', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '06_score_distributions.png'), dpi=300)
        plt.close()

    def plot_success_heatmap(self):
        """
        7. Success Rate Heatmap (persona x goal).
        """
        # Pivot table: persona vs goal
        heatmap_data = self.df.pivot_table(
            values='overall_score',
            index='persona',
            columns='goal',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(14, 8))

        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
                   vmin=0, vmax=100, cbar_kws={'label': 'Overall Score'},
                   linewidths=0.5, ax=ax)

        ax.set_title('Overall Score Heatmap (Persona × Goal)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Goal')
        ax.set_ylabel('Persona')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '07_success_heatmap.png'), dpi=300)
        plt.close()

    def plot_ux_breakdown(self):
        """
        8. User Experience Breakdown - Pie Chart.
        """
        ux_counts = self.df['ux_overall'].value_counts()

        fig, ax = plt.subplots(figsize=(8, 8))

        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        pie_colors = [colors.get(label, 'blue') for label in ux_counts.index]

        wedges, texts, autotexts = ax.pie(
            ux_counts.values,
            labels=ux_counts.index,
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )

        ax.set_title('User Experience Breakdown', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '08_ux_breakdown.png'), dpi=300)
        plt.close()

    def plot_efficiency_vs_quality(self):
        """
        9. Efficiency vs Quality Scatter Plot.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot: Efficiency vs Overall Score
        scatter = ax.scatter(
            self.df['efficiency_score'],
            self.df['overall_score'],
            c=self.df['turns'],
            s=100,
            cmap='viridis',
            alpha=0.6,
            edgecolors='black'
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Turn Count', rotation=270, labelpad=20)

        # Trendline
        z = np.polyfit(self.df['efficiency_score'], self.df['overall_score'], 1)
        p = np.poly1d(z)
        ax.plot(self.df['efficiency_score'].sort_values(),
               p(self.df['efficiency_score'].sort_values()),
               "r--", alpha=0.5, linewidth=2, label='Trendline')

        ax.set_xlabel('Efficiency Score')
        ax.set_ylabel('Overall Score')
        ax.set_title('Efficiency vs Overall Quality', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '09_efficiency_vs_quality.png'), dpi=300)
        plt.close()

    def plot_dimension_correlation(self):
        """
        10. Dimension Correlation Heatmap.
        """
        dimensions = [
            "goal_achievement_score",
            "conversation_quality_score",
            "response_relevance_score",
            "user_experience_score",
            "efficiency_score",
            "overall_score"
        ]

        labels = [
            "Goal\nAchievement",
            "Conversation\nQuality",
            "Response\nRelevance",
            "User\nExperience",
            "Efficiency",
            "Overall\nScore"
        ]

        corr_matrix = self.df[dimensions].corr()

        # Rename columns for better display
        corr_matrix.columns = labels
        corr_matrix.index = labels

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   vmin=-1, vmax=1, center=0, square=True,
                   linewidths=0.5, cbar_kws={'label': 'Correlation'}, ax=ax)

        ax.set_title('Evaluation Dimension Correlation Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '10_dimension_correlation.png'), dpi=300)
        plt.close()

    def plot_api_latency_distribution(self):
        """
        11. API Latency Distribution - Histogram of API latencies across all turns.
        """
        latencies = self.per_turn_perf_data.get("api_latency_ms", [])

        if not latencies:
            print("    WARNING: No API latency data available. Skipping chart.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(latencies, bins=30, color='steelblue', edgecolor='black', alpha=0.7)

        # Statistics lines
        mean_val = np.mean(latencies)
        median_val = np.median(latencies)
        p95_val = np.percentile(latencies, 95)

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.1f}ms')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                  label=f'Median: {median_val:.1f}ms')
        ax.axvline(p95_val, color='orange', linestyle='--', linewidth=2,
                  label=f'P95: {p95_val:.1f}ms')

        ax.set_xlabel('API Latency (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('API Latency Distribution (All Turns)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '11_api_latency_distribution.png'), dpi=300)
        plt.close()

    def plot_ttft_distribution(self):
        """
        12. TTFT Distribution - Histogram of Time To First Token across all turns.
        """
        ttfts = self.per_turn_perf_data.get("ttft_ms", [])

        if not ttfts:
            print("    WARNING: No TTFT data available. Skipping chart.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(ttfts, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)

        # Statistics lines
        mean_val = np.mean(ttfts)
        median_val = np.median(ttfts)
        p95_val = np.percentile(ttfts, 95)

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.1f}ms')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                  label=f'Median: {median_val:.1f}ms')
        ax.axvline(p95_val, color='orange', linestyle='--', linewidth=2,
                  label=f'P95: {p95_val:.1f}ms')

        ax.set_xlabel('Time To First Token (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('TTFT Distribution (All Turns)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '12_ttft_distribution.png'), dpi=300)
        plt.close()

    def plot_tokens_per_second_distribution(self):
        """
        13. Tokens per Second Distribution - Histogram of throughput across all turns.
        """
        tokens_per_sec = self.per_turn_perf_data.get("tokens_per_second", [])

        if not tokens_per_sec:
            print("    WARNING: No tokens/sec data available. Skipping chart.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(tokens_per_sec, bins=30, color='coral', edgecolor='black', alpha=0.7)

        # Statistics lines
        mean_val = np.mean(tokens_per_sec)
        median_val = np.median(tokens_per_sec)
        p95_val = np.percentile(tokens_per_sec, 95)

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.1f} tok/s')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                  label=f'Median: {median_val:.1f} tok/s')
        ax.axvline(p95_val, color='orange', linestyle='--', linewidth=2,
                  label=f'P95: {p95_val:.1f} tok/s')

        ax.set_xlabel('Tokens per Second')
        ax.set_ylabel('Frequency')
        ax.set_title('Throughput Distribution (All Turns)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '13_tokens_per_second_distribution.png'), dpi=300)
        plt.close()

    def plot_performance_heatmap(self):
        """
        14. Performance Metrics Heatmap - Average performance by persona × goal.
        """
        # Create pivot tables for each performance metric
        metrics = {
            'API Latency (ms)': 'avg_api_latency_ms',
            'TTFT (ms)': 'avg_ttft_ms',
            'Tokens/sec': 'avg_tokens_per_second'
        }

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, (metric_name, column) in enumerate(metrics.items()):
            # Check if data exists
            if column not in self.df.columns or self.df[column].isna().all():
                axes[idx].text(0.5, 0.5, f'No {metric_name} data',
                             ha='center', va='center', fontsize=14)
                axes[idx].set_title(metric_name, fontsize=12, fontweight='bold')
                axes[idx].axis('off')
                continue

            # Pivot table
            heatmap_data = self.df.pivot_table(
                values=column,
                index='persona',
                columns='goal',
                aggfunc='mean'
            )

            # Determine colormap (lower is better for latency, higher is better for tokens/sec)
            if 'Tokens/sec' in metric_name:
                cmap = 'RdYlGn'  # Red = low (bad), Green = high (good)
            else:
                cmap = 'RdYlGn_r'  # Red = high (bad), Green = low (good)

            sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap=cmap,
                       linewidths=0.5, ax=axes[idx], cbar_kws={'label': metric_name})

            axes[idx].set_title(metric_name, fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('')
            axes[idx].set_ylabel('Persona' if idx == 0 else '')
            axes[idx].tick_params(axis='x', rotation=45)

        plt.suptitle('Performance Metrics by Persona × Goal', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '14_performance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize multi-turn evaluation results"
    )
    parser.add_argument(
        "batch_summary_path",
        type=str,
        help="Path to batch_summary_*.json file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same dir as input with /plots suffix)"
    )

    args = parser.parse_args()

    # Load batch summary
    print(f"Loading batch summary: {args.batch_summary_path}")
    with open(args.batch_summary_path, 'r', encoding='utf-8') as f:
        batch_summary = json.load(f)

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Same directory as input file + /plots
        input_dir = os.path.dirname(args.batch_summary_path)
        output_dir = os.path.join(input_dir, "plots")

    # Create visualizer
    visualizer = MultiTurnVisualizer(batch_summary, output_dir)

    # Generate all plots
    visualizer.visualize_all()

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"Generated 14 plots in: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
