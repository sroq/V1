"""
Beszélgetési célok (goals) definiálása a multi-turn evaluation számára.

Ez a modul különböző beszélgetési célokat határoz meg, amelyeket
a szimulált felhasználók el akarnak érni a RAG asszisztenssel folytatott
beszélgetés során.

Goal struktúra:
- name: Cél neve
- description: Rövid leírás
- difficulty: Nehézség (easy, medium, hard)
- initial_query: Kezdő kérdés
- success_criteria: Sikerkritériumok
- expected_turns: Várható körök száma
- milestones: Mérföldkövek a cél eléréséhez
"""

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class Goal:
    """Beszélgetési cél osztály"""
    name: str
    description: str
    difficulty: str  # "easy", "medium", "hard"
    initial_query: str
    success_criteria: List[str]
    expected_turns: int
    milestones: List[Dict[str, Any]]
    category: str  # "factual", "comprehension", "multi_topic", "exploratory"

    def to_dict(self) -> Dict[str, Any]:
        """Konvertálás dict-té"""
        return {
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty,
            "initial_query": self.initial_query,
            "success_criteria": self.success_criteria,
            "expected_turns": self.expected_turns,
            "milestones": self.milestones,
            "category": self.category
        }


# ============================================================================
# GOAL DEFINÍCIÓK - The Jungle Book alapú
# ============================================================================

# === EASY GOALS ===

GOAL_MOWGLI_IDENTITY = Goal(
    name="Maugli személyazonosságának tisztázása",
    description="Megérteni, hogy ki Maugli és milyen szerepe van a történetben",
    difficulty="easy",
    initial_query="Ki az a Maugli?",
    success_criteria=[
        "Megtudta, hogy Maugli egy emberi gyerek",
        "Megtudta, hogy farkasok nevelték fel",
        "Megtudta, hogy a dzsungelben él"
    ],
    expected_turns=2,
    milestones=[
        {
            "turn": 1,
            "description": "Alapvető információ: Maugli emberi gyerek a dzsungelben",
            "required_info": ["emberi gyerek", "dzsungel"]
        },
        {
            "turn": 2,
            "description": "További részletek: farkasok, nevelés",
            "required_info": ["farkasok", "nevelték"]
        }
    ],
    category="factual"
)

GOAL_BALOO_IDENTITY = Goal(
    name="Balú személyazonosságának tisztázása",
    description="Megérteni, hogy ki Balú és milyen kapcsolata van Mauglihoz",
    difficulty="easy",
    initial_query="Ki az a Balú?",
    success_criteria=[
        "Megtudta, hogy Balú egy medve",
        "Megtudta, hogy Maugli tanítója",
        "Megtudta a kapcsolatukat"
    ],
    expected_turns=2,
    milestones=[
        {
            "turn": 1,
            "description": "Balú azonosítása: medve, tanító",
            "required_info": ["medve", "tanító"]
        },
        {
            "turn": 2,
            "description": "Kapcsolat Mauglihoz",
            "required_info": ["Maugli", "tanít"]
        }
    ],
    category="factual"
)

# === MEDIUM GOALS ===

GOAL_SHERE_KHAN_CONFLICT = Goal(
    name="Ser Kán konfliktusának megértése",
    description="Megérteni Ser Kán és Maugli közötti konfliktust",
    difficulty="medium",
    initial_query="Miért üldözi Ser Kán Mauglit?",
    success_criteria=[
        "Megtudta Ser Kán motivációját",
        "Megértette az emberiség vs dzsungel témát",
        "Megértette a veszély természetét"
    ],
    expected_turns=4,
    milestones=[
        {
            "turn": 1,
            "description": "Ki Ser Kán és mi a problémája",
            "required_info": ["tigris", "utálja az embereket"]
        },
        {
            "turn": 2,
            "description": "Miért veszélyes Mauglira",
            "required_info": ["emberi gyerek", "veszély"]
        },
        {
            "turn": 3,
            "description": "Történelmi háttér",
            "required_info": ["puskás", "sérülés vagy bosszú"]
        },
        {
            "turn": 4,
            "description": "Konfliktus kimenetele",
            "required_info": ["megoldás", "tűz vagy harc"]
        }
    ],
    category="comprehension"
)

GOAL_JUNGLE_LAW = Goal(
    name="A dzsungel törvényének megértése",
    description="Megérteni a dzsungel törvényét és szerepét a történetben",
    difficulty="medium",
    initial_query="Mi az a dzsungel törvénye?",
    success_criteria=[
        "Megtudta a törvény alapelveit",
        "Megértette a szerepét a közösségben",
        "Példákat kapott az alkalmazására"
    ],
    expected_turns=3,
    milestones=[
        {
            "turn": 1,
            "description": "Törvény alapelve",
            "required_info": ["törvény", "szabályok", "közösség"]
        },
        {
            "turn": 2,
            "description": "Ki tanítja és ki követi",
            "required_info": ["Balú", "állatok", "Maugli"]
        },
        {
            "turn": 3,
            "description": "Példák alkalmazásra",
            "required_info": ["példa", "szabály"]
        }
    ],
    category="comprehension"
)

# === HARD GOALS ===

GOAL_MOWGLI_JOURNEY = Goal(
    name="Maugli útjának teljes megértése",
    description="Végigkövetni Maugli fejlődését a könyvben, kezdettől a végéig",
    difficulty="hard",
    initial_query="Mesélj Maugli történetéről!",
    success_criteria=[
        "Megértette Maugli eredetét",
        "Követte a fejlődését és tanulását",
        "Megértette a két világ közötti helyzetét",
        "Megértette a végső döntését"
    ],
    expected_turns=6,
    milestones=[
        {
            "turn": 1,
            "description": "Eredet: hogyan került a dzsungelbe",
            "required_info": ["emberi gyerek", "farkasok találták"]
        },
        {
            "turn": 2,
            "description": "Nevelkedés és oktatás",
            "required_info": ["Balú", "Bagíra", "tanulás"]
        },
        {
            "turn": 3,
            "description": "Konfliktusok (Ser Kán, majmok)",
            "required_info": ["veszély", "kaland"]
        },
        {
            "turn": 4,
            "description": "Kapcsolat emberekkel",
            "required_info": ["falu", "emberek"]
        },
        {
            "turn": 5,
            "description": "Identitás válság",
            "required_info": ["két világ", "választás"]
        },
        {
            "turn": 6,
            "description": "Végső döntés és helykeresés",
            "required_info": ["döntés", "otthon"]
        }
    ],
    category="comprehension"
)

GOAL_CHARACTERS_COMPARISON = Goal(
    name="Karakterek összehasonlítása",
    description="Több karakter összehasonlítása és szerepük megértése",
    difficulty="hard",
    initial_query="Hasonlítsd össze Balút és Bagírát mint Maugli tanítóit!",
    success_criteria=[
        "Megértette Balú jellemét és tanítási módszerét",
        "Megértette Bagíra jellemét és tanítási módszerét",
        "Azonosította a különbségeket",
        "Megértette, hogy kiegészítik egymást"
    ],
    expected_turns=5,
    milestones=[
        {
            "turn": 1,
            "description": "Balú jellemzése",
            "required_info": ["medve", "vidám", "törvény"]
        },
        {
            "turn": 2,
            "description": "Bagíra jellemzése",
            "required_info": ["párduc", "komoly", "bölcs"]
        },
        {
            "turn": 3,
            "description": "Tanítási módszereik",
            "required_info": ["különbség", "megközelítés"]
        },
        {
            "turn": 4,
            "description": "Kapcsolatuk Mauglihoz",
            "required_info": ["védelem", "tanítás"]
        },
        {
            "turn": 5,
            "description": "Kiegészítő szerepek",
            "required_info": ["együttműködés", "egyensúly"]
        }
    ],
    category="multi_topic"
)

# === EXPLORATORY GOALS ===

GOAL_JUNGLE_EXPLORATION = Goal(
    name="Dzsungel felfedezése",
    description="Különböző témák felfedezése a dzsungelről és lakóiról",
    difficulty="medium",
    initial_query="Milyen állatok élnek a dzsungelben?",
    success_criteria=[
        "Több állatról kapott információt",
        "Megértette a közösség dinamikáját",
        "Érdekesnek találta a választ"
    ],
    expected_turns=4,
    milestones=[
        {
            "turn": 1,
            "description": "Főbb állatok listája",
            "required_info": ["farkasok", "medve", "párduc", "tigris"]
        },
        {
            "turn": 2,
            "description": "Egy állat részletes bemutatása",
            "required_info": ["részletek", "karakter"]
        },
        {
            "turn": 3,
            "description": "Kapcsolatok és hierarchia",
            "required_info": ["kapcsolat", "rendszer"]
        },
        {
            "turn": 4,
            "description": "Érdekességek",
            "required_info": ["történet", "esemény"]
        }
    ],
    category="exploratory"
)


# ============================================================================
# GOAL REGISTRY
# ============================================================================

GOALS: Dict[str, Goal] = {
    # Easy
    "mowgli_identity": GOAL_MOWGLI_IDENTITY,
    "baloo_identity": GOAL_BALOO_IDENTITY,

    # Medium
    "shere_khan_conflict": GOAL_SHERE_KHAN_CONFLICT,
    "jungle_law": GOAL_JUNGLE_LAW,
    "jungle_exploration": GOAL_JUNGLE_EXPLORATION,

    # Hard
    "mowgli_journey": GOAL_MOWGLI_JOURNEY,
    "characters_comparison": GOAL_CHARACTERS_COMPARISON,
}


def get_goal(goal_id: str) -> Goal:
    """
    Goal lekérése ID alapján.

    Args:
        goal_id: Goal azonosító

    Returns:
        Goal objektum

    Raises:
        ValueError: Ha a goal_id nem létezik
    """
    if goal_id not in GOALS:
        available = ", ".join(GOALS.keys())
        raise ValueError(f"Unknown goal: {goal_id}. Available: {available}")

    return GOALS[goal_id]


def list_goals() -> List[str]:
    """
    Elérhető célok listája.

    Returns:
        Goal ID-k listája
    """
    return list(GOALS.keys())


def get_goals_by_difficulty(difficulty: str) -> List[Goal]:
    """
    Célok szűrése nehézség szerint.

    Args:
        difficulty: Nehézség (easy, medium, hard)

    Returns:
        Goal objektumok listája
    """
    return [goal for goal in GOALS.values() if goal.difficulty == difficulty]


def get_goals_by_category(category: str) -> List[Goal]:
    """
    Célok szűrése kategória szerint.

    Args:
        category: Kategória (factual, comprehension, multi_topic, exploratory)

    Returns:
        Goal objektumok listája
    """
    return [goal for goal in GOALS.values() if goal.category == category]


def get_goal_summary(goal: Goal) -> str:
    """
    Goal összefoglaló szöveg generálása.

    Args:
        goal: Goal objektum

    Returns:
        Összefoglaló string
    """
    milestones_str = "\n".join([
        f"  {i+1}. {m['description']}"
        for i, m in enumerate(goal.milestones)
    ])

    return f"""
Cél: {goal.name}
Leírás: {goal.description}
Nehézség: {goal.difficulty}
Kategória: {goal.category}
Kezdő kérdés: "{goal.initial_query}"
Várható körök: {goal.expected_turns}

Mérföldkövek:
{milestones_str}

Sikerkritériumok:
{chr(10).join([f"  - {c}" for c in goal.success_criteria])}
    """.strip()


if __name__ == "__main__":
    # Példa használat és tesztelés
    print("=== Elérhető Célok ===\n")

    for difficulty in ["easy", "medium", "hard"]:
        goals = get_goals_by_difficulty(difficulty)
        print(f"\n{'='*60}")
        print(f"Nehézség: {difficulty.upper()} ({len(goals)} cél)")
        print(f"{'='*60}\n")

        for goal in goals:
            print(get_goal_summary(goal))
            print("-" * 60)
