"""
Multi-Turn Conversation Evaluator - LLM as a Judge.

Ez a modul felelős a multi-turn beszélgetések kiértékeléséért.
LLM-et használ (GPT-4o mini) mint bírót különböző dimenziók mentén.

Értékelési dimenziók:
1. Goal Achievement: Elérte-e a célt?
2. Conversation Quality: Beszélgetés minősége
3. Response Relevance: Válaszok relevanciája
4. User Experience: Felhasználói élmény
5. Efficiency: Hatékonyság (turns, time)

Minden dimenzióhoz strukturált prompt és scoring rendszer.
"""

import os
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv
import json

from goals import Goal
from personas import Persona
from cost_metrics import get_cost_tracker

load_dotenv()


# ============================================================================
# EVALUATION PROMPTS
# ============================================================================

GOAL_ACHIEVEMENT_PROMPT = """
Te egy szakértő értékelő vagy. A feladatod egy multi-turn beszélgetés kiértékelése abból a szempontból, hogy a felhasználó elérte-e a célját.

CÉL INFORMÁCIÓK:
Cél neve: {goal_name}
Cél leírása: {goal_description}
Sikerkritériumok:
{success_criteria}

Mérföldkövek (várható haladás):
{milestones}

BESZÉLGETÉS:
{conversation}

Értékeld a következőket:
1. Hány sikerkritériumot teljesített a beszélgetés?
2. Hány mérföldkövet értek el?
3. Milyen mértékben érte el a felhasználó a célját?

VÁLASZ FORMÁTUM (JSON):
{{
  "goal_reached": true/false,
  "success_criteria_met": [lista a teljesített kritériumokról],
  "success_criteria_missed": [lista az el nem ért kritériumokról],
  "milestones_reached": szám,
  "total_milestones": szám,
  "achievement_score": 0-100 közötti szám,
  "reasoning": "részletes indoklás"
}}

Csak a JSON-t add vissza, semmi mást.
"""

CONVERSATION_QUALITY_PROMPT = """
Te egy szakértő értékelő vagy. A feladatod egy multi-turn beszélgetés minőségének értékelése.

BESZÉLGETÉS:
{conversation}

Értékeld a következő szempontokat:
1. Koherencia: Logikusan követik-e egymást a kérdések és válaszok?
2. Természetesség: Természetes-e a beszélgetés folyama?
3. Információ minőség: Releváns és pontos információkat kap a felhasználó?
4. Kontextus megértés: Az asszisztens megérti-e a kontextust?

VÁLASZ FORMÁTUM (JSON):
{{
  "coherence_score": 0-10,
  "naturalness_score": 0-10,
  "information_quality_score": 0-10,
  "context_understanding_score": 0-10,
  "overall_quality_score": 0-100,
  "strengths": ["erősség 1", "erősség 2", ...],
  "weaknesses": ["gyengeség 1", "gyengeség 2", ...],
  "reasoning": "részletes indoklás"
}}

Csak a JSON-t add vissza, semmi mást.
"""

RESPONSE_RELEVANCE_PROMPT = """
Te egy szakértő értékelő vagy. A feladatod az asszisztens válaszainak relevanciájának értékelése.

BESZÉLGETÉS:
{conversation}

Minden asszisztens válaszhoz értékeld:
1. Releváns-e a felhasználó kérdésére?
2. Pontos-e az információ?
3. Kielégítő-e a válasz?

VÁLASZ FORMÁTUM (JSON):
{{
  "responses": [
    {{
      "turn": 1,
      "user_query": "kérdés",
      "assistant_response": "válasz",
      "relevance_score": 0-10,
      "accuracy_score": 0-10,
      "completeness_score": 0-10,
      "reasoning": "rövid indoklás"
    }},
    ...
  ],
  "average_relevance": 0-10,
  "average_accuracy": 0-10,
  "average_completeness": 0-10,
  "overall_relevance_score": 0-100
}}

Csak a JSON-t add vissza, semmi mást.
"""

USER_EXPERIENCE_PROMPT = """
Te egy szakértő értékelő vagy. A feladatod a felhasználói élmény értékelése egy beszélgetés során.

PERSONA:
{persona_info}

BESZÉLGETÉS:
{conversation}

Értékeld a felhasználói élményt:
1. Elérte-e a felhasználó a célját frusztráció nélkül?
2. Mennyi ideig tartott (turns)?
3. Megfelelő-e a válaszok stílusa a personának?
4. Pozitív vagy negatív élmény volt?

VÁLASZ FORMÁTUM (JSON):
{{
  "frustration_detected": true/false,
  "turn_count": szám,
  "appropriate_for_persona": true/false,
  "overall_experience": "positive" vagy "negative" vagy "neutral",
  "ux_score": 0-100,
  "pain_points": ["fájdalmas pont 1", ...],
  "delightful_moments": ["pozitív pillanat 1", ...],
  "reasoning": "részletes indoklás"
}}

Csak a JSON-t add vissza, semmi mást.
"""

EFFICIENCY_PROMPT = """
Te egy szakértő értékelő vagy. A feladatod a beszélgetés hatékonyságának értékelése.

VÁRHATÓ TURNS: {expected_turns}
TÉNYLEGES TURNS: {actual_turns}

BESZÉLGETÉS:
{conversation}

Értékeld a hatékonyságot:
1. Optimális volt-e a turns száma?
2. Volt-e felesleges ismétlés?
3. Gyorsan jutottak-e el a célhoz?

VÁLASZ FORMÁTUM (JSON):
{{
  "expected_turns": szám,
  "actual_turns": szám,
  "turn_efficiency": 0-100,
  "redundancy_detected": true/false,
  "time_wasted_turns": szám,
  "efficiency_score": 0-100,
  "reasoning": "részletes indoklás"
}}

Csak a JSON-t add vissza, semmi mást.
"""


# ============================================================================
# EVALUATOR CLASS
# ============================================================================

class MultiTurnEvaluator:
    """
    Multi-turn beszélgetések kiértékelője.

    LLM-et használ (GPT-4o mini) mint bírót több dimenzióban.
    """

    def __init__(
        self,
        openai_api_key: str = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0  # Determinisztikus értékeléshez
    ):
        """
        Inicializálás.

        Args:
            openai_api_key: OpenAI API key
            model: Model név (alapértelmezett: gpt-4o-mini)
            temperature: Temperature (alapértelmezett: 0.0)
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

        # Cost tracking
        self.cost_tracker = get_cost_tracker()

    def evaluate_conversation(
        self,
        conversation: List[Dict[str, str]],
        goal: Goal,
        persona: Persona,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Teljes beszélgetés kiértékelése minden dimenzióban.

        Args:
            conversation: Beszélgetés lista
            goal: Beszélgetési cél
            persona: Felhasználói persona
            metadata: Extra metaadatok (pl. időmérés)

        Returns:
            Teljes értékelési eredmény
        """
        print(f"Evaluating conversation (goal: {goal.name}, persona: {persona.name})...")

        # Formázott beszélgetés
        conversation_str = self._format_conversation(conversation)

        # 1. Goal Achievement
        print("  - Evaluating goal achievement...")
        goal_eval = self._evaluate_goal_achievement(conversation_str, goal)

        # 2. Conversation Quality
        print("  - Evaluating conversation quality...")
        quality_eval = self._evaluate_conversation_quality(conversation_str)

        # 3. Response Relevance
        print("  - Evaluating response relevance...")
        relevance_eval = self._evaluate_response_relevance(conversation_str)

        # 4. User Experience
        print("  - Evaluating user experience...")
        ux_eval = self._evaluate_user_experience(conversation_str, persona)

        # 5. Efficiency
        print("  - Evaluating efficiency...")
        efficiency_eval = self._evaluate_efficiency(
            conversation_str,
            goal.expected_turns,
            len([m for m in conversation if m["role"] == "user"])
        )

        # Összesített pontszám
        overall_score = self._calculate_overall_score({
            "goal": goal_eval,
            "quality": quality_eval,
            "relevance": relevance_eval,
            "ux": ux_eval,
            "efficiency": efficiency_eval
        })

        return {
            "goal_achievement": goal_eval,
            "conversation_quality": quality_eval,
            "response_relevance": relevance_eval,
            "user_experience": ux_eval,
            "efficiency": efficiency_eval,
            "overall_score": overall_score,
            "metadata": metadata or {},
            "conversation": conversation
        }

    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Beszélgetés formázása LLM számára."""
        lines = []
        for i, msg in enumerate(conversation, 1):
            role = "USER" if msg["role"] == "user" else "ASSISTANT"
            lines.append(f"[Turn {i//2 + 1}] {role}: {msg['content']}")
        return "\n".join(lines)

    def _call_llm_judge(self, prompt: str, dimension: str = "unknown") -> Dict[str, Any]:
        """LLM hívás és JSON parse."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=3000  # Increased from 1000 to handle longer JSON responses
            )

            content = response.choices[0].message.content.strip()

            # Cost tracking
            if response.usage:
                cost = self.cost_tracker.record_llm_cost(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    attributes={"dimension": dimension}
                )
                # Log cost per dimension
                # print(f"  [Cost] {dimension}: ${cost:.6f} ({response.usage.prompt_tokens} in + {response.usage.completion_tokens} out)")

            # JSON parse
            # Ha nem JSON formátumú, próbáljuk kinyerni
            if not content.startswith("{"):
                # Keressük meg a JSON blokkot
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end > start:
                    content = content[start:end]

            return json.loads(content)

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response was: {content}")
            print(f"\n[DEBUG] Full LLM response (first 500 chars):\n{content[:500]}")
            print(f"[DEBUG] Response length: {len(content)} characters")
            print(f"[DEBUG] Finish reason: {response.choices[0].finish_reason if 'response' in locals() else 'N/A'}")
            return {"error": "JSON parse failed", "raw_response": content, "parse_error": str(e)}
        except Exception as e:
            print(f"LLM call error: {e}")
            return {"error": str(e)}

    def _evaluate_goal_achievement(self, conversation: str, goal: Goal) -> Dict[str, Any]:
        """Goal achievement értékelés."""
        success_criteria_str = "\n".join([f"- {c}" for c in goal.success_criteria])
        milestones_str = "\n".join([
            f"Turn {m['turn']}: {m['description']}"
            for m in goal.milestones
        ])

        prompt = GOAL_ACHIEVEMENT_PROMPT.format(
            goal_name=goal.name,
            goal_description=goal.description,
            success_criteria=success_criteria_str,
            milestones=milestones_str,
            conversation=conversation
        )

        return self._call_llm_judge(prompt, dimension="goal_achievement")

    def _evaluate_conversation_quality(self, conversation: str) -> Dict[str, Any]:
        """Conversation quality értékelés."""
        prompt = CONVERSATION_QUALITY_PROMPT.format(conversation=conversation)
        return self._call_llm_judge(prompt, dimension="conversation_quality")

    def _evaluate_response_relevance(self, conversation: str) -> Dict[str, Any]:
        """Response relevance értékelés."""
        prompt = RESPONSE_RELEVANCE_PROMPT.format(conversation=conversation)
        return self._call_llm_judge(prompt, dimension="response_relevance")

    def _evaluate_user_experience(self, conversation: str, persona: Persona) -> Dict[str, Any]:
        """User experience értékelés."""
        persona_info = f"""
Név: {persona.name}
Türelem: {persona.patience}/10
Szakértelem: {persona.expertise}
Frusztráció küszöb: {persona.behavior_patterns.get('frustration_threshold', 5)} rossz válasz
        """.strip()

        prompt = USER_EXPERIENCE_PROMPT.format(
            persona_info=persona_info,
            conversation=conversation
        )
        return self._call_llm_judge(prompt, dimension="user_experience")

    def _evaluate_efficiency(
        self,
        conversation: str,
        expected_turns: int,
        actual_turns: int
    ) -> Dict[str, Any]:
        """Efficiency értékelés."""
        prompt = EFFICIENCY_PROMPT.format(
            expected_turns=expected_turns,
            actual_turns=actual_turns,
            conversation=conversation
        )
        return self._call_llm_judge(prompt, dimension="efficiency")

    def _calculate_overall_score(self, evaluations: Dict[str, Dict[str, Any]]) -> float:
        """
        Összesített pontszám számítása (súlyozott átlag).

        Súlyok:
        - Goal Achievement: 40%
        - Conversation Quality: 20%
        - Response Relevance: 20%
        - User Experience: 10%
        - Efficiency: 10%
        """
        weights = {
            "goal": 0.4,
            "quality": 0.2,
            "relevance": 0.2,
            "ux": 0.1,
            "efficiency": 0.1
        }

        scores = {}
        scores["goal"] = evaluations["goal"].get("achievement_score", 0)
        scores["quality"] = evaluations["quality"].get("overall_quality_score", 0)
        scores["relevance"] = evaluations["relevance"].get("overall_relevance_score", 0)
        scores["ux"] = evaluations["ux"].get("ux_score", 0)
        scores["efficiency"] = evaluations["efficiency"].get("efficiency_score", 0)

        overall = sum(scores[k] * weights[k] for k in weights.keys())
        return round(overall, 2)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from personas import get_persona
    from goals import get_goal

    print("=== Multi-Turn Evaluator Test ===\n")

    # Test conversation
    conversation = [
        {"role": "user", "content": "Ki az a Maugli?"},
        {"role": "assistant", "content": "Maugli egy emberi gyerek, akit farkasok neveltek fel a dzsungelben. Rudyard Kipling The Jungle Book című művének főszereplője."},
        {"role": "user", "content": "Ki nevelte fel őt?"},
        {"role": "assistant", "content": "Mauglit főként a farkasfalkához tartozó Anya és Apa farkas nevelte fel, de tanítói voltak Balú, a medve és Bagíra, a fekete párduc is."}
    ]

    # Goal és Persona
    goal = get_goal("mowgli_identity")
    persona = get_persona("patient_intermediate")

    # Evaluator
    evaluator = MultiTurnEvaluator()

    # Értékelés
    result = evaluator.evaluate_conversation(conversation, goal, persona)

    # Eredmény kiírása
    print("\n=== EVALUATION RESULTS ===\n")
    print(f"Overall Score: {result['overall_score']}/100\n")
    print(f"Goal Achievement: {result['goal_achievement'].get('achievement_score', 'N/A')}/100")
    print(f"  Goal Reached: {result['goal_achievement'].get('goal_reached', 'N/A')}")
    print(f"\nConversation Quality: {result['conversation_quality'].get('overall_quality_score', 'N/A')}/100")
    print(f"\nResponse Relevance: {result['response_relevance'].get('overall_relevance_score', 'N/A')}/100")
    print(f"\nUser Experience: {result['user_experience'].get('ux_score', 'N/A')}/100")
    print(f"\nEfficiency: {result['efficiency'].get('efficiency_score', 'N/A')}/100")