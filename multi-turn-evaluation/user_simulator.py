"""
User Simulator - Felhasználói viselkedés szimulálása LLM-mel.

Ez a modul felelős a felhasználói viselkedés szimulálásáért.
LLM-et használ olyan kérdések generálására, amelyek:
- Megfelelnek a persona jellemzőinek
- Haladnak a goal felé
- Reagálnak az assistant válaszaira

A szimulátor követi:
- Persona jellemzőit (türelem, szakértelem, stílus)
- Goal progressziót (milestones)
- Beszélgetés kontextusát
- Frusztrációs szintet
"""

import os
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import json

from personas import Persona
from goals import Goal

load_dotenv()


class UserSimulator:
    """
    Felhasználó szimulátor osztály.

    Ez az osztály LLM-et használ olyan felhasználó szimulálására,
    aki egy adott persona szerint viselkedik és egy adott goal felé halad.
    """

    def __init__(
        self,
        persona: Persona,
        goal: Goal,
        openai_api_key: str = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7
    ):
        """
        Inicializálás.

        Args:
            persona: Felhasználói persona
            goal: Beszélgetési cél
            openai_api_key: OpenAI API key (vagy környezeti változóból)
            model: OpenAI model (alapértelmezett: gpt-4o-mini)
            temperature: Temperature (alapértelmezett: 0.7)
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=api_key)
        self.persona = persona
        self.goal = goal
        self.model = model
        self.temperature = temperature

        # Állapot követés
        self.current_turn = 0
        self.frustration_level = 0
        self.goal_progress = []  # Elért mérföldkövek
        self.conversation_history = []

    def generate_initial_query(self) -> str:
        """
        Kezdő kérdés generálása.

        Returns:
            Kezdő kérdés szövege
        """
        # A goal-ban már definiálva van az initial_query,
        # de a persona stílusa szerint módosítjuk
        base_query = self.goal.initial_query

        if self.persona.communication_style["verbosity"] == "terse":
            # Rövidebb, tömörebb
            return base_query

        elif self.persona.communication_style["verbosity"] == "detailed":
            # Részletesebb, kontextussal
            prompt = f"""
A következő kérdést szeretném feltenni egy AI asszisztensnek: "{base_query}"

Fogalmazd át ezt a kérdést úgy, hogy:
- Részletesebb és pontosabb legyen
- Formális hangnemű legyen
- Tartalmazzon kontextust vagy háttér információt

Add vissza csak a kérdést, semmi mást.
            """.strip()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )

            return response.choices[0].message.content.strip()

        else:
            # Balanced - használjuk az eredeti kérdést
            return base_query

    def generate_next_query(
        self,
        conversation_history: List[Dict[str, str]],
        last_assistant_response: str
    ) -> Optional[str]:
        """
        Következő kérdés generálása a beszélgetés alapján.

        Args:
            conversation_history: Eddigi beszélgetés
            last_assistant_response: Utolsó assistant válasz

        Returns:
            Következő kérdés vagy None ha a beszélgetés véget ért
        """
        self.current_turn += 1
        self.conversation_history = conversation_history

        # Ellenőrizzük, hogy feladta-e
        if self.current_turn >= self.persona.behavior_patterns.get("gives_up_after", 10):
            return None

        # Aktuális milestone
        current_milestone = self._get_current_milestone()

        # Frusztráció ellenőrzés
        is_frustrated = self._check_frustration(last_assistant_response)

        # Prompt összeállítása
        prompt = self._build_user_prompt(
            conversation_history,
            last_assistant_response,
            current_milestone,
            is_frustrated
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=200
            )

            next_query = response.choices[0].message.content.strip()

            # Ha a válasz jelzi, hogy vége a beszélgetésnek
            if self._should_end_conversation(next_query):
                return None

            return next_query

        except Exception as e:
            print(f"Error generating next query: {e}")
            return None

    def _get_current_milestone(self) -> Optional[Dict[str, Any]]:
        """Aktuális milestone lekérése a turn alapján."""
        for milestone in self.goal.milestones:
            if milestone["turn"] == self.current_turn:
                return milestone
        return None

    def _check_frustration(self, response: str) -> bool:
        """
        Frusztráció ellenőrzés - rossz válasz esetén nő a frusztráció.

        Args:
            response: Assistant válasza

        Returns:
            True ha frusztrált, False egyébként
        """
        # Egyszerű heurisztika: rövid válaszok vagy "nem tudom" válaszok
        if len(response) < 50:
            self.frustration_level += 1
        elif "nem tudom" in response.lower() or "nincs információ" in response.lower():
            self.frustration_level += 1

        threshold = self.persona.behavior_patterns.get("frustration_threshold", 5)
        return self.frustration_level >= threshold

    def _build_user_prompt(
        self,
        conversation_history: List[Dict[str, str]],
        last_response: str,
        current_milestone: Optional[Dict[str, Any]],
        is_frustrated: bool
    ) -> str:
        """User simulator prompt összeállítása."""

        # Beszélgetés history formázása
        history_str = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history[-4:]  # Utolsó 4 üzenet
        ])

        # Milestone információ
        milestone_str = ""
        if current_milestone:
            milestone_str = f"""
Jelenlegi mérföldkő (turn {current_milestone['turn']}):
- Cél: {current_milestone['description']}
- Szükséges információ: {', '.join(current_milestone['required_info'])}
            """.strip()

        # Persona jellemzők
        persona_traits = f"""
Személyiség:
- Név: {self.persona.name}
- Szakértelem: {self.persona.expertise}
- Türelem: {self.persona.patience}/10
- Kommunikációs stílus: {self.persona.communication_style['verbosity']}, {self.persona.communication_style['formality']}
- Kérdés pontossága: {self.persona.communication_style['question_clarity']}
        """.strip()

        # Frusztráció kezelés
        frustration_note = ""
        if is_frustrated:
            if self.persona.behavior_patterns.get("shows_emotion", False):
                frustration_note = "\nFrusztrált vagy! Mutasd ki az érzelmeidet (pl. 'ez nem segít', 'nem értem')."
            else:
                frustration_note = "\nFrusztrált vagy, de ne mutasd ki. Fogalmazz pontosabban."

        # Teljes prompt
        prompt = f"""
Te egy felhasználó vagy, aki egy AI asszisztenssel beszélget. A következő persona szerint viselkedsz:

{persona_traits}

A célt, amit el akarsz érni:
Cél: {self.goal.name}
Leírás: {self.goal.description}
Kezdő kérdés volt: "{self.goal.initial_query}"

{milestone_str}

Eddigi beszélgetés (utolsó üzenetek):
{history_str}

Utolsó válasz az assisztenttől:
"{last_response}"

{frustration_note}

Feladatod:
1. Elemezd az utolsó választ
2. Döntsd el, hogy közelebb jutottál-e a célodhoz
3. Generálj egy KÖVETŐ KÉRDÉST, ami:
   - Megfelel a persona kommunikációs stílusának
   - Halad a cél felé (figyelj a milestone-ra!)
   - Reagál az utolsó válaszra
   - Ha frusztrált vagy, az tükröződjön a kérdésben (persona szerint)

FONTOS szabályok:
- Ha a célodat elérted, írj: "GOAL_REACHED"
- Ha feladod (túl sok rossz válasz), írj: "GIVING_UP"
- Egyébként csak a KÖVETKEZŐ KÉRDÉST írd, semmi mást
- NE ismételd meg ugyanazt a kérdést
- NE adj hosszú magyarázatot
- Csak a kérdést add vissza

Következő kérdés:
        """.strip()

        return prompt

    def _should_end_conversation(self, response: str) -> bool:
        """Ellenőrzi, hogy véget ért-e a beszélgetés."""
        end_markers = ["GOAL_REACHED", "GIVING_UP", "köszönöm", "kész vagyok"]
        return any(marker in response for marker in end_markers)

    def evaluate_progress(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Goal progress kiértékelése.

        Args:
            conversation_history: Teljes beszélgetés

        Returns:
            Progress dict:
            {
                "milestones_reached": int,
                "total_milestones": int,
                "progress_percentage": float,
                "goal_reached": bool
            }
        """
        # Egyszerű heurisztika: mérföldkövek számolása turn alapján
        milestones_reached = min(self.current_turn, len(self.goal.milestones))
        total_milestones = len(self.goal.milestones)

        return {
            "milestones_reached": milestones_reached,
            "total_milestones": total_milestones,
            "progress_percentage": (milestones_reached / total_milestones * 100) if total_milestones > 0 else 0,
            "goal_reached": milestones_reached >= total_milestones,
            "current_turn": self.current_turn,
            "frustration_level": self.frustration_level
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from personas import get_persona
    from goals import get_goal

    print("=== User Simulator Test ===\n")

    # Persona és Goal betöltése
    persona = get_persona("patient_intermediate")
    goal = get_goal("mowgli_identity")

    print(f"Persona: {persona.name}")
    print(f"Goal: {goal.name}\n")

    # Simulator létrehozása
    simulator = UserSimulator(persona, goal)

    # Kezdő kérdés
    initial_query = simulator.generate_initial_query()
    print(f"Initial Query: {initial_query}\n")

    # Szimulált assistant válasz
    fake_response = "Maugli egy emberi gyerek, akit farkasok neveltek fel a dzsungelben."

    # Következő kérdés generálása
    conversation = [
        {"role": "user", "content": initial_query},
        {"role": "assistant", "content": fake_response}
    ]

    next_query = simulator.generate_next_query(conversation, fake_response)
    print(f"Next Query: {next_query}\n")

    # Progress kiértékelés
    progress = simulator.evaluate_progress(conversation)
    print("Progress:")
    print(json.dumps(progress, indent=2))
