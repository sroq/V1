"""
Multi-Turn Conversation Evaluation Package

Ez a csomag átfogó evaluation rendszert biztosít RAG-alapú AI asszisztensek
multi-turn beszélgetéseinek értékelésére.

Fő komponensek:
- Personas: Felhasználói típusok
- Goals: Beszélgetési célok
- UserSimulator: Felhasználó szimuláció LLM-mel
- AssistantClient: Backend API kliens
- MultiTurnEvaluator: LLM-alapú értékelő (5 dimenzió)
- MultiTurnEvaluationRunner: Pipeline orchestrator

Használat:
    from multi_turn_evaluation import MultiTurnEvaluationRunner, get_persona, get_goal

    runner = MultiTurnEvaluationRunner()
    persona = get_persona("patient_intermediate")
    goal = get_goal("mowgli_identity")
    result = runner.run_single_conversation(persona, goal)
"""

from .personas import Persona, get_persona, list_personas, PERSONAS
from .goals import Goal, get_goal, list_goals, GOALS
from .user_simulator import UserSimulator
from .assistant_client import AssistantClient, create_message
from .evaluator import MultiTurnEvaluator
from .run_multi_turn_evaluation import MultiTurnEvaluationRunner
from .visualize_results import MultiTurnVisualizer

__version__ = "1.0.0"
__all__ = [
    # Personas
    "Persona",
    "get_persona",
    "list_personas",
    "PERSONAS",

    # Goals
    "Goal",
    "get_goal",
    "list_goals",
    "GOALS",

    # Components
    "UserSimulator",
    "AssistantClient",
    "create_message",
    "MultiTurnEvaluator",
    "MultiTurnEvaluationRunner",
    "MultiTurnVisualizer",
]
