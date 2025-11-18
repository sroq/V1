"""
Multi-Turn Evaluation - Main Orchestrator Script

Ez a script összefogja az összes komponenst és végrehajt egy teljes
multi-turn evaluation-t különböző persona-goal kombinációkkal.

Pipeline:
1. Persona és Goal kiválasztása
2. User Simulator inicializálása
3. Assistant Client inicializálása
4. Multi-turn beszélgetés szimulálása (loop)
5. Beszélgetés értékelése (Evaluator)
6. Eredmények mentése

Használat:
    python run_multi_turn_evaluation.py

Vagy specifikus persona-goal párokkal:
    python run_multi_turn_evaluation.py --persona patient_intermediate --goal mowgli_identity
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import time

from personas import get_persona, list_personas, Persona
from goals import get_goal, list_goals, Goal
from user_simulator import UserSimulator
from assistant_client import AssistantClient, create_message
from evaluator import MultiTurnEvaluator
from visualize_results import MultiTurnVisualizer


class MultiTurnEvaluationRunner:
    """
    Multi-Turn Evaluation orchestrator.

    Ez az osztály koordinálja a teljes evaluation folyamatot.
    """

    def __init__(
        self,
        max_turns: int = 15,
        output_dir: str = "results"
    ):
        """
        Inicializálás.

        Args:
            max_turns: Maximum turns száma (biztonsági korlát)
            output_dir: Eredmények mentési könyvtára
        """
        self.max_turns = max_turns
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Komponensek
        self.assistant_client = AssistantClient()
        self.evaluator = MultiTurnEvaluator()

    def run_single_conversation(
        self,
        persona: Persona,
        goal: Goal,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Egyetlen beszélgetés futtatása és értékelése.

        Args:
            persona: Felhasználói persona
            goal: Beszélgetési cél
            verbose: Részletes kimenet

        Returns:
            Dict az eredményekkel
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Running conversation:")
            print(f"  Persona: {persona.name}")
            print(f"  Goal: {goal.name} ({goal.difficulty})")
            print(f"{'='*70}\n")

        # User Simulator
        simulator = UserSimulator(persona, goal)

        # Kezdő kérdés
        initial_query = simulator.generate_initial_query()
        if verbose:
            print(f"[Turn 1] USER: {initial_query}")

        conversation = []
        start_time = time.time()

        # Performance tracking per-turn
        per_turn_performance = []

        # Kezdő kérdés küldése
        conversation.append(create_message("user", initial_query))
        result = self.assistant_client.send_message(conversation)

        if not result["success"]:
            print(f"ERROR: {result['error']}")
            return {
                "success": False,
                "error": result["error"],
                "persona": persona.name,
                "goal": goal.name
            }

        assistant_response = result["response"]
        conversation.append(create_message("assistant", assistant_response))

        # Save turn performance
        per_turn_performance.append(result.get("performance", {}))

        if verbose:
            print(f"[Turn 1] ASSISTANT: {assistant_response[:200]}...")
            perf = result.get("performance", {})
            print(f"  Performance: {perf.get('api_latency_ms')}ms latency, "
                  f"TTFT: {perf.get('ttft_ms')}ms, "
                  f"{perf.get('tokens_per_second')} tok/s")
            print()

        # Multi-turn loop
        turn = 1
        while turn < self.max_turns:
            # Következő kérdés generálása
            next_query = simulator.generate_next_query(conversation, assistant_response)

            if next_query is None:
                if verbose:
                    print(f"[Turn {turn+1}] Conversation ended (goal reached or user gave up)\n")
                break

            turn += 1
            if verbose:
                print(f"[Turn {turn}] USER: {next_query}")

            # Kérdés küldése
            conversation.append(create_message("user", next_query))
            result = self.assistant_client.send_message(conversation)

            if not result["success"]:
                print(f"ERROR at turn {turn}: {result['error']}")
                break

            assistant_response = result["response"]
            conversation.append(create_message("assistant", assistant_response))

            # Save turn performance
            per_turn_performance.append(result.get("performance", {}))

            if verbose:
                print(f"[Turn {turn}] ASSISTANT: {assistant_response[:200]}...")
                perf = result.get("performance", {})
                print(f"  Performance: {perf.get('api_latency_ms')}ms latency, "
                      f"TTFT: {perf.get('ttft_ms')}ms, "
                      f"{perf.get('tokens_per_second')} tok/s")
                print()

        end_time = time.time()
        duration_s = end_time - start_time

        # Goal progress kiértékelés
        progress = simulator.evaluate_progress(conversation)

        # Aggregate performance metrics
        aggregated_performance = self._aggregate_performance_metrics(per_turn_performance)

        if verbose:
            print(f"Conversation completed:")
            print(f"  Total turns: {turn}")
            print(f"  Duration: {duration_s:.1f}s")
            print(f"  Progress: {progress['progress_percentage']:.1f}%")
            print(f"  Frustration level: {progress['frustration_level']}")
            print(f"\nPerformance Summary:")
            print(f"  Avg API Latency: {aggregated_performance.get('avg_api_latency_ms', 'N/A')}ms")
            print(f"  Avg TTFT: {aggregated_performance.get('avg_ttft_ms', 'N/A')}ms")
            print(f"  Avg Tokens/sec: {aggregated_performance.get('avg_tokens_per_second', 'N/A')}")
            print(f"  Total Tokens: {aggregated_performance.get('total_tokens', 'N/A')}")
            print()

        # LLM-based Evaluation
        if verbose:
            print("Running LLM-based evaluation...\n")

        evaluation_result = self.evaluator.evaluate_conversation(
            conversation,
            goal,
            persona,
            metadata={
                "turns": turn,
                "duration_s": duration_s,
                "user_progress": progress,
                "performance": aggregated_performance
            }
        )

        if verbose:
            print(f"\n{'='*70}")
            print(f"EVALUATION RESULTS")
            print(f"{'='*70}")
            print(f"Overall Score: {evaluation_result['overall_score']}/100")
            print(f"  - Goal Achievement: {evaluation_result['goal_achievement'].get('achievement_score', 'N/A')}/100")
            print(f"  - Conversation Quality: {evaluation_result['conversation_quality'].get('overall_quality_score', 'N/A')}/100")
            print(f"  - Response Relevance: {evaluation_result['response_relevance'].get('overall_relevance_score', 'N/A')}/100")
            print(f"  - User Experience: {evaluation_result['user_experience'].get('ux_score', 'N/A')}/100")
            print(f"  - Efficiency: {evaluation_result['efficiency'].get('efficiency_score', 'N/A')}/100")
            print(f"{'='*70}\n")

        return {
            "success": True,
            "persona": persona.name,
            "goal": goal.name,
            "conversation": conversation,
            "evaluation": evaluation_result,
            "metadata": {
                "turns": turn,
                "duration_s": duration_s,
                "user_progress": progress,
                "performance": aggregated_performance,
                "per_turn_performance": per_turn_performance
            }
        }

    def _aggregate_performance_metrics(self, per_turn_perf: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Performance metrikák aggregálása több turn-ből.

        Args:
            per_turn_perf: Per-turn performance metrikák listája

        Returns:
            Aggregált performance metrikák
        """
        if not per_turn_perf:
            return {}

        # Filter out None values for averaging
        api_latencies = [p.get("api_latency_ms") for p in per_turn_perf if p.get("api_latency_ms") is not None]
        ttfts = [p.get("ttft_ms") for p in per_turn_perf if p.get("ttft_ms") is not None]
        tokens_per_sec = [p.get("tokens_per_second") for p in per_turn_perf if p.get("tokens_per_second") is not None]
        total_tokens = sum(p.get("response_tokens", 0) for p in per_turn_perf)
        retry_count = sum(p.get("retry_count", 0) for p in per_turn_perf)

        return {
            "avg_api_latency_ms": round(sum(api_latencies) / len(api_latencies), 2) if api_latencies else None,
            "min_api_latency_ms": min(api_latencies) if api_latencies else None,
            "max_api_latency_ms": max(api_latencies) if api_latencies else None,
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
            "min_ttft_ms": min(ttfts) if ttfts else None,
            "max_ttft_ms": max(ttfts) if ttfts else None,
            "avg_tokens_per_second": round(sum(tokens_per_sec) / len(tokens_per_sec), 2) if tokens_per_sec else None,
            "min_tokens_per_second": round(min(tokens_per_sec), 2) if tokens_per_sec else None,
            "max_tokens_per_second": round(max(tokens_per_sec), 2) if tokens_per_sec else None,
            "total_tokens": total_tokens,
            "total_retries": retry_count,
            "turn_count": len(per_turn_perf)
        }

    def run_batch_evaluation(
        self,
        persona_ids: List[str] = None,
        goal_ids: List[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Több beszélgetés batch futtatása és értékelése.

        Args:
            persona_ids: Persona ID-k listája (None = összes)
            goal_ids: Goal ID-k listája (None = összes)
            verbose: Részletes kimenet

        Returns:
            Dict a batch eredményekkel
        """
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # Alapértelmezett: összes persona és goal
        if persona_ids is None:
            persona_ids = list_personas()

        if goal_ids is None:
            goal_ids = list_goals()

        # Kombinációk generálása
        combinations = [
            (pid, gid)
            for pid in persona_ids
            for gid in goal_ids
        ]

        print(f"\n{'='*70}")
        print(f"BATCH EVALUATION")
        print(f"{'='*70}")
        print(f"Run directory: {run_dir}")
        print(f"Personas: {len(persona_ids)}")
        print(f"Goals: {len(goal_ids)}")
        print(f"Total combinations: {len(combinations)}")
        print(f"{'='*70}\n")

        results = []
        for i, (persona_id, goal_id) in enumerate(combinations, 1):
            print(f"\n[{i}/{len(combinations)}] Running: {persona_id} x {goal_id}")

            persona = get_persona(persona_id)
            goal = get_goal(goal_id)

            try:
                result = self.run_single_conversation(persona, goal, verbose=verbose)
                results.append(result)

                # Mentés után minden beszélgetés
                self._save_conversation_result(result, run_dir)

            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "persona": persona_id,
                    "goal": goal_id
                })

            print("\n" + "="*70 + "\n")

        # Batch összesítő mentése
        summary = self._create_batch_summary(results)
        self._save_batch_summary(summary, run_dir)

        return summary

    def _save_conversation_result(self, result: Dict[str, Any], run_dir: str = None):
        """Egyetlen beszélgetés eredményének mentése."""
        if not result.get("success"):
            return

        # Use run_dir if provided, otherwise output_dir
        save_dir = run_dir if run_dir else self.output_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result['persona']}_{result['goal']}_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Saved: {filepath}")

    def _create_batch_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch összesítő létrehozása."""
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        # Átlagos metrikák
        avg_score = sum(r["evaluation"]["overall_score"] for r in successful) / len(successful) if successful else 0
        avg_turns = sum(r["metadata"]["turns"] for r in successful) / len(successful) if successful else 0
        avg_duration = sum(r["metadata"]["duration_s"] for r in successful) / len(successful) if successful else 0

        # Goal achievement rate
        goal_reached = sum(
            1 for r in successful
            if r["evaluation"]["goal_achievement"].get("goal_reached", False)
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "total_conversations": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "metrics": {
                "average_overall_score": round(avg_score, 2),
                "average_turns": round(avg_turns, 2),
                "average_duration_s": round(avg_duration, 2),
                "goal_achievement_rate": round(goal_reached / len(successful) * 100, 2) if successful else 0
            },
            "results": results
        }

    def _save_batch_summary(self, summary: Dict[str, Any], run_dir: str = None):
        """Batch összesítő mentése és vizualizáció generálása."""
        # Use run_dir if provided, otherwise output_dir
        save_dir = run_dir if run_dir else self.output_dir

        filename = "batch_summary.json"
        filepath = os.path.join(save_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*70}")
        print(f"BATCH SUMMARY SAVED: {filepath}")
        print(f"{'='*70}")
        print(f"Total conversations: {summary['total_conversations']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Average overall score: {summary['metrics']['average_overall_score']}/100")
        print(f"Goal achievement rate: {summary['metrics']['goal_achievement_rate']}%")
        print(f"Average turns: {summary['metrics']['average_turns']}")
        print(f"{'='*70}\n")

        # Vizualizáció generálása
        if summary['successful'] > 0:
            print("Generating visualizations...\n")
            plots_dir = os.path.join(save_dir, "plots")
            try:
                visualizer = MultiTurnVisualizer(summary, plots_dir)
                visualizer.visualize_all()
                print(f"\nVisualizations saved to: {plots_dir}")
            except Exception as e:
                print(f"Warning: Failed to generate visualizations: {e}")
                print("Continuing without visualizations...")
        else:
            print("Skipping visualization (no successful conversations)\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Turn Conversation Evaluation"
    )
    parser.add_argument(
        "--persona",
        type=str,
        help=f"Persona ID (available: {', '.join(list_personas())})"
    )
    parser.add_argument(
        "--goal",
        type=str,
        help=f"Goal ID (available: {', '.join(list_goals())})"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch evaluation with all combinations"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Maximum turns per conversation (default: 15)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    # Runner létrehozása
    runner = MultiTurnEvaluationRunner(
        max_turns=args.max_turns,
        output_dir=args.output_dir
    )

    # Health check
    print("Checking backend connectivity...")
    if not runner.assistant_client.health_check():
        print("ERROR: Backend is not running!")
        print("Please start: cd assistant && npm run dev")
        return

    print("Backend is running ✓\n")

    # Batch mode
    if args.batch:
        persona_ids = [args.persona] if args.persona else None
        goal_ids = [args.goal] if args.goal else None
        runner.run_batch_evaluation(persona_ids, goal_ids, verbose=not args.quiet)

    # Single mode
    else:
        # Create timestamped run directory for single run too
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"Run directory: {run_dir}\n")

        # Ha nincs megadva persona/goal, default
        persona_id = args.persona or "patient_intermediate"
        goal_id = args.goal or "mowgli_identity"

        persona = get_persona(persona_id)
        goal = get_goal(goal_id)

        result = runner.run_single_conversation(persona, goal, verbose=not args.quiet)

        if result.get("success"):
            runner._save_conversation_result(result, run_dir)


if __name__ == "__main__":
    main()
