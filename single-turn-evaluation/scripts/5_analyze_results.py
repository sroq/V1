#!/usr/bin/env python3
"""
Analyze Results and Generate Report.

This script analyzes all evaluation results and generates:
- Aggregated metrics (correctness rate, relevance rate)
- Breakdowns by category and difficulty
- Visualizations (charts)
- Summary report

Process:
1. Load correctness_evaluations.json and relevance_evaluations.json
2. Aggregate metrics overall and by category/difficulty
3. Generate visualizations
4. Create summary report
5. Save final_report.json

Usage:
    python scripts/5_analyze_results.py
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import visualization libraries (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    logger.info("Visualization libraries available")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("Matplotlib/seaborn not installed, skipping visualizations")


def load_evaluations(data_dir: Path) -> Dict[str, Any]:
    """
    Load correctness and relevance evaluations.

    Args:
        data_dir: Data directory path

    Returns:
        Dictionary with correctness and relevance evaluations

    Raises:
        FileNotFoundError: If evaluation files not found
    """
    correctness_path = data_dir / 'correctness_evaluations.json'
    relevance_path = data_dir / 'relevance_evaluations.json'

    if not correctness_path.exists():
        raise FileNotFoundError(
            f"Correctness evaluations not found: {correctness_path}\n"
            "Please run scripts/3_evaluate_correctness.py first"
        )

    if not relevance_path.exists():
        raise FileNotFoundError(
            f"Relevance evaluations not found: {relevance_path}\n"
            "Please run scripts/4_evaluate_relevance.py first"
        )

    with open(correctness_path, 'r', encoding='utf-8') as f:
        correctness_data = json.load(f)

    with open(relevance_path, 'r', encoding='utf-8') as f:
        relevance_data = json.load(f)

    logger.info(f"Loaded correctness evaluations: {len(correctness_data['evaluations'])} entries")
    logger.info(f"Loaded relevance evaluations: {len(relevance_data['evaluations'])} entries")

    return {
        'correctness': correctness_data,
        'relevance': relevance_data
    }


def aggregate_metrics(evaluations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate metrics from evaluations.

    Args:
        evaluations: Dictionary with correctness and relevance data

    Returns:
        Aggregated metrics dictionary
    """
    correctness_evals = evaluations['correctness']['evaluations']
    relevance_evals = evaluations['relevance']['evaluations']

    # Overall metrics
    total = len(correctness_evals)
    correct_count = sum(1 for e in correctness_evals if e['is_correct'])
    relevant_count = sum(1 for e in relevance_evals if e['is_relevant'])

    correctness_rate = correct_count / total * 100 if total > 0 else 0
    relevance_rate = relevant_count / total * 100 if total > 0 else 0

    # Metrics by category
    category_metrics = {}
    for e in correctness_evals:
        cat = e.get('category', 'unknown')
        if cat not in category_metrics:
            category_metrics[cat] = {
                'total': 0,
                'correct': 0,
                'relevant': 0
            }
        category_metrics[cat]['total'] += 1
        if e['is_correct']:
            category_metrics[cat]['correct'] += 1

    for e in relevance_evals:
        cat = e.get('category', 'unknown')
        if cat in category_metrics and e['is_relevant']:
            category_metrics[cat]['relevant'] += 1

    # Calculate rates
    for cat, stats in category_metrics.items():
        stats['correctness_rate'] = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        stats['relevance_rate'] = stats['relevant'] / stats['total'] * 100 if stats['total'] > 0 else 0

    # Metrics by difficulty
    difficulty_metrics = {}
    for e in correctness_evals:
        diff = e.get('difficulty', 'unknown')
        if diff not in difficulty_metrics:
            difficulty_metrics[diff] = {
                'total': 0,
                'correct': 0,
                'relevant': 0
            }
        difficulty_metrics[diff]['total'] += 1
        if e['is_correct']:
            difficulty_metrics[diff]['correct'] += 1

    for e in relevance_evals:
        diff = e.get('difficulty', 'unknown')
        if diff in difficulty_metrics and e['is_relevant']:
            difficulty_metrics[diff]['relevant'] += 1

    # Calculate rates
    for diff, stats in difficulty_metrics.items():
        stats['correctness_rate'] = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        stats['relevance_rate'] = stats['relevant'] / stats['total'] * 100 if stats['total'] > 0 else 0

    # Both correct AND relevant
    both_correct_relevant = sum(
        1 for c, r in zip(correctness_evals, relevance_evals)
        if c['is_correct'] and r['is_relevant']
    )
    both_rate = both_correct_relevant / total * 100 if total > 0 else 0

    metrics = {
        'overall': {
            'total_evaluations': total,
            'correctness_rate': correctness_rate,
            'relevance_rate': relevance_rate,
            'both_correct_and_relevant': both_correct_relevant,
            'both_rate': both_rate,
            'correct_count': correct_count,
            'relevant_count': relevant_count
        },
        'by_category': category_metrics,
        'by_difficulty': difficulty_metrics
    }

    return metrics


def create_visualizations(
    metrics: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Create visualization charts.

    Args:
        metrics: Aggregated metrics dictionary
        output_dir: Output directory for charts
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Skipping visualizations (libraries not available)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

    # 1. Overall Metrics Bar Chart
    logger.info("Creating overall metrics chart...")
    fig, ax = plt.subplots()
    metrics_data = {
        'Correctness': metrics['overall']['correctness_rate'],
        'Relevance': metrics['overall']['relevance_rate'],
        'Both': metrics['overall']['both_rate']
    }
    bars = ax.bar(metrics_data.keys(), metrics_data.values(), color=['#2ecc71', '#3498db', '#9b59b6'])
    ax.set_ylabel('Rate (%)')
    ax.set_title('Overall Evaluation Metrics')
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'overall_metrics.png', dpi=150)
    plt.close()
    logger.info(f"  ✓ Saved: {output_dir / 'overall_metrics.png'}")

    # 2. Metrics by Category
    logger.info("Creating category breakdown chart...")
    categories = sorted(metrics['by_category'].keys())
    correctness_rates = [metrics['by_category'][cat]['correctness_rate'] for cat in categories]
    relevance_rates = [metrics['by_category'][cat]['relevance_rate'] for cat in categories]

    fig, ax = plt.subplots()
    x = range(len(categories))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], correctness_rates, width, label='Correctness', color='#2ecc71')
    bars2 = ax.bar([i + width/2 for i in x], relevance_rates, width, label='Relevance', color='#3498db')

    ax.set_ylabel('Rate (%)')
    ax.set_title('Metrics by Question Category')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_category.png', dpi=150)
    plt.close()
    logger.info(f"  ✓ Saved: {output_dir / 'metrics_by_category.png'}")

    # 3. Metrics by Difficulty
    logger.info("Creating difficulty breakdown chart...")
    difficulties = sorted(metrics['by_difficulty'].keys())
    correctness_rates_diff = [metrics['by_difficulty'][diff]['correctness_rate'] for diff in difficulties]
    relevance_rates_diff = [metrics['by_difficulty'][diff]['relevance_rate'] for diff in difficulties]

    fig, ax = plt.subplots()
    x = range(len(difficulties))

    bars1 = ax.bar([i - width/2 for i in x], correctness_rates_diff, width, label='Correctness', color='#2ecc71')
    bars2 = ax.bar([i + width/2 for i in x], relevance_rates_diff, width, label='Relevance', color='#3498db')

    ax.set_ylabel('Rate (%)')
    ax.set_title('Metrics by Question Difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_difficulty.png', dpi=150)
    plt.close()
    logger.info(f"  ✓ Saved: {output_dir / 'metrics_by_difficulty.png'}")


def generate_summary_report(
    metrics: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Generate human-readable summary report.

    Args:
        metrics: Aggregated metrics dictionary
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SINGLE-TURN EVALUATION SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        overall = metrics['overall']
        f.write(f"Total Evaluations: {overall['total_evaluations']}\n")
        f.write(f"Correctness Rate: {overall['correctness_rate']:.1f}% ({overall['correct_count']}/{overall['total_evaluations']})\n")
        f.write(f"Relevance Rate: {overall['relevance_rate']:.1f}% ({overall['relevant_count']}/{overall['total_evaluations']})\n")
        f.write(f"Both Correct & Relevant: {overall['both_rate']:.1f}% ({overall['both_correct_and_relevant']}/{overall['total_evaluations']})\n\n")

        # By category
        f.write("METRICS BY CATEGORY\n")
        f.write("-" * 80 + "\n")
        for cat, stats in sorted(metrics['by_category'].items()):
            f.write(f"\n{cat.upper()}:\n")
            f.write(f"  Total Questions: {stats['total']}\n")
            f.write(f"  Correctness: {stats['correctness_rate']:.1f}% ({stats['correct']}/{stats['total']})\n")
            f.write(f"  Relevance: {stats['relevance_rate']:.1f}% ({stats['relevant']}/{stats['total']})\n")

        # By difficulty
        f.write("\n\nMETRICS BY DIFFICULTY\n")
        f.write("-" * 80 + "\n")
        for diff, stats in sorted(metrics['by_difficulty'].items()):
            f.write(f"\n{diff.upper()}:\n")
            f.write(f"  Total Questions: {stats['total']}\n")
            f.write(f"  Correctness: {stats['correctness_rate']:.1f}% ({stats['correct']}/{stats['total']})\n")
            f.write(f"  Relevance: {stats['relevance_rate']:.1f}% ({stats['relevant']}/{stats['total']})\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    logger.info(f"✓ Summary report saved to: {output_path}")


def main():
    """
    Main script entry point.
    """
    logger.info("=" * 80)
    logger.info("ANALYZE RESULTS AND GENERATE REPORT")
    logger.info("=" * 80)

    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    results_dir = Path(__file__).parent.parent / 'results'
    charts_dir = results_dir / 'charts'

    # Load evaluations
    try:
        evaluations = load_evaluations(data_dir)
    except FileNotFoundError as e:
        logger.error(f"✗ {e}")
        sys.exit(1)

    # Aggregate metrics
    logger.info("\nAggregating metrics...")
    metrics = aggregate_metrics(evaluations)

    # Display metrics
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL METRICS")
    logger.info("=" * 80)
    overall = metrics['overall']
    logger.info(f"Total Evaluations: {overall['total_evaluations']}")
    logger.info(f"Correctness Rate: {overall['correctness_rate']:.1f}%")
    logger.info(f"Relevance Rate: {overall['relevance_rate']:.1f}%")
    logger.info(f"Both Correct & Relevant: {overall['both_rate']:.1f}%")

    logger.info("\nBy Category:")
    for cat, stats in sorted(metrics['by_category'].items()):
        logger.info(f"  {cat}: Correctness={stats['correctness_rate']:.1f}%, Relevance={stats['relevance_rate']:.1f}%")

    logger.info("\nBy Difficulty:")
    for diff, stats in sorted(metrics['by_difficulty'].items()):
        logger.info(f"  {diff}: Correctness={stats['correctness_rate']:.1f}%, Relevance={stats['relevance_rate']:.1f}%")

    # Create visualizations
    if VISUALIZATION_AVAILABLE:
        logger.info("\nCreating visualizations...")
        create_visualizations(metrics, charts_dir)
    else:
        logger.info("\nSkipping visualizations (install matplotlib and seaborn to enable)")

    # Generate summary report
    logger.info("\nGenerating summary report...")
    summary_path = results_dir / 'summary_report.txt'
    results_dir.mkdir(parents=True, exist_ok=True)
    generate_summary_report(metrics, summary_path)

    # Save final report JSON
    logger.info("\nSaving final report...")
    final_report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'dataset_id': evaluations['correctness']['metadata']['dataset_id']
        },
        'metrics': metrics
    }

    final_report_path = data_dir / 'final_report.json'
    with open(final_report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Final report saved to: {final_report_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults summary:")
    logger.info(f"  - JSON report: {final_report_path}")
    logger.info(f"  - Text summary: {summary_path}")
    if VISUALIZATION_AVAILABLE:
        logger.info(f"  - Charts: {charts_dir}/")


if __name__ == '__main__':
    main()
