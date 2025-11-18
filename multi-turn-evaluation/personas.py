"""
Felhasználói personák definiálása a multi-turn evaluation számára.

Ez a modul különböző felhasználói típusokat definiál, amelyek különböző
türelmi szinttel, szakértelemmel és kommunikációs stílussal rendelkeznek.

Persona struktúra:
- name: Persona neve
- description: Rövid leírás
- patience: Türelem szint (1-10, ahol 10 a leginkább türelmes)
- expertise: Szakértelem szint (novice, intermediate, expert)
- communication_style: Kommunikációs stíl jellemzői
- behavior_patterns: Viselkedési minták
"""

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class Persona:
    """Felhasználói persona osztály"""
    name: str
    description: str
    patience: int  # 1-10 skála
    expertise: str  # "novice", "intermediate", "expert"
    communication_style: Dict[str, Any]
    behavior_patterns: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Konvertálás dict-té"""
        return {
            "name": self.name,
            "description": self.description,
            "patience": self.patience,
            "expertise": self.expertise,
            "communication_style": self.communication_style,
            "behavior_patterns": self.behavior_patterns
        }


# ============================================================================
# PERSONA DEFINÍCIÓK
# ============================================================================

IMPATIENT_NOVICE = Persona(
    name="Türelmetlen Kezdő",
    description="Új felhasználó, aki gyors válaszokat vár, könnyen frusztrálódik",
    patience=3,
    expertise="novice",
    communication_style={
        "verbosity": "terse",  # Rövid, tömör kérdések
        "formality": "informal",  # Informális hangnem
        "question_clarity": "vague",  # Nem mindig pontos kérdések
        "follow_up_tendency": "high"  # Sok követő kérdés
    },
    behavior_patterns={
        "frustration_threshold": 2,  # 2 nem megfelelő válasz után frusztrált
        "gives_up_after": 4,  # 4 kör után feladja
        "asks_clarification": True,  # Kér pontosítást
        "restates_question": True,  # Újrafogalmazza a kérdést ha nem kap jó választ
        "shows_emotion": True  # Érzelmeket mutat (pl. "ez nem működik!", "nem értem")
    }
)

PATIENT_INTERMEDIATE = Persona(
    name="Türelmes Haladó",
    description="Tapasztalt felhasználó, aki pontosan fogalmaz és türelmes",
    patience=8,
    expertise="intermediate",
    communication_style={
        "verbosity": "balanced",  # Kiegyensúlyozott hosszúságú kérdések
        "formality": "semi-formal",  # Félig formális
        "question_clarity": "clear",  # Pontos, érthető kérdések
        "follow_up_tendency": "medium"  # Közepes mennyiségű követő kérdés
    },
    behavior_patterns={
        "frustration_threshold": 5,  # 5 nem megfelelő válasz után frusztrált
        "gives_up_after": 8,  # 8 kör után feladja
        "asks_clarification": True,  # Kér pontosítást
        "restates_question": False,  # Ritkán fogalmaz újra
        "shows_emotion": False  # Nem mutat érzelmeket
    }
)

EXPERT_RESEARCHER = Persona(
    name="Szakértő Kutató",
    description="Tapasztalt szakember, aki mélyre ásó kérdéseket tesz fel",
    patience=10,
    expertise="expert",
    communication_style={
        "verbosity": "detailed",  # Részletes, összetett kérdések
        "formality": "formal",  # Formális hangnem
        "question_clarity": "very_clear",  # Nagyon pontos kérdések
        "follow_up_tendency": "low"  # Kevés követő kérdés, mert pontosan kérdez
    },
    behavior_patterns={
        "frustration_threshold": 8,  # 8 nem megfelelő válasz után frusztrált
        "gives_up_after": 12,  # 12 kör után feladja
        "asks_clarification": False,  # Ritkán kér pontosítást
        "restates_question": False,  # Nem fogalmaz újra
        "shows_emotion": False  # Nem mutat érzelmeket
    }
)

CASUAL_EXPLORER = Persona(
    name="Kíváncsi Böngésző",
    description="Kíváncsi felhasználó, aki szabadon kérdezget különböző témákról",
    patience=6,
    expertise="novice",
    communication_style={
        "verbosity": "variable",  # Változó hosszúságú kérdések
        "formality": "informal",  # Informális
        "question_clarity": "moderate",  # Közepes pontosság
        "follow_up_tendency": "high"  # Sok követő kérdés
    },
    behavior_patterns={
        "frustration_threshold": 4,  # 4 nem megfelelő válasz után frusztrált
        "gives_up_after": 6,  # 6 kör után feladja
        "asks_clarification": True,  # Kér pontosítást
        "restates_question": True,  # Újrafogalmazza
        "shows_emotion": True,  # Mutat érzelmeket
        "topic_jumping": True  # Témát válthat
    }
)

FOCUSED_LEARNER = Persona(
    name="Célratörő Tanuló",
    description="Tanulni vágyó felhasználó, aki szisztematikusan halad",
    patience=7,
    expertise="intermediate",
    communication_style={
        "verbosity": "balanced",  # Kiegyensúlyozott
        "formality": "semi-formal",  # Félig formális
        "question_clarity": "clear",  # Pontos kérdések
        "follow_up_tendency": "medium"  # Közepes követő kérdések
    },
    behavior_patterns={
        "frustration_threshold": 6,  # 6 nem megfelelő válasz után frusztrált
        "gives_up_after": 10,  # 10 kör után feladja
        "asks_clarification": True,  # Kér pontosítást
        "restates_question": False,  # Nem fogalmaz újra gyakran
        "shows_emotion": False,  # Nem mutat érzelmeket
        "progressive_learning": True  # Fokozatosan halad
    }
)


# ============================================================================
# PERSONA REGISTRY
# ============================================================================

PERSONAS: Dict[str, Persona] = {
    "impatient_novice": IMPATIENT_NOVICE,
    "patient_intermediate": PATIENT_INTERMEDIATE,
    "expert_researcher": EXPERT_RESEARCHER,
    "casual_explorer": CASUAL_EXPLORER,
    "focused_learner": FOCUSED_LEARNER
}


def get_persona(persona_id: str) -> Persona:
    """
    Persona lekérése ID alapján.

    Args:
        persona_id: Persona azonosító (pl. "impatient_novice")

    Returns:
        Persona objektum

    Raises:
        ValueError: Ha a persona_id nem létezik
    """
    if persona_id not in PERSONAS:
        available = ", ".join(PERSONAS.keys())
        raise ValueError(f"Unknown persona: {persona_id}. Available: {available}")

    return PERSONAS[persona_id]


def list_personas() -> List[str]:
    """
    Elérhető personák listája.

    Returns:
        Persona ID-k listája
    """
    return list(PERSONAS.keys())


def get_persona_summary(persona: Persona) -> str:
    """
    Persona összefoglaló szöveg generálása.

    Args:
        persona: Persona objektum

    Returns:
        Összefoglaló string
    """
    return f"""
Persona: {persona.name}
Leírás: {persona.description}
Türelem: {persona.patience}/10
Szakértelem: {persona.expertise}
Kommunikáció: {persona.communication_style['verbosity']}, {persona.communication_style['formality']}
Frusztrációs küszöb: {persona.behavior_patterns['frustration_threshold']} rossz válasz
Feladási pont: {persona.behavior_patterns['gives_up_after']} kör után
    """.strip()


if __name__ == "__main__":
    # Példa használat és tesztelés
    print("=== Elérhető Personák ===\n")

    for persona_id in list_personas():
        persona = get_persona(persona_id)
        print(get_persona_summary(persona))
        print("-" * 60)
