"""
Konfigurációs beállítások a single-turn értékeléshez.

Ez a modul tartalmazza az összes konfigurációs paramétert az LLM as a Judge
értékelési rendszerhez, beleértve az adatbázis kapcsolatokat, API kulcsokat,
modell beállításokat és értékelési paramétereket.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Környezeti változók betöltése a szülő könyvtár .env fájljából
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ENV_PATH = PROJECT_ROOT / '.env'

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print(f"Figyelmeztetés: .env fájl nem található itt: {ENV_PATH}")

# ============================================================================
# ADATBÁZIS KONFIGURÁCIÓ
# ============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'rag_db'),
    'user': os.getenv('DB_USER', 'rag_user'),
    'password': os.getenv('DB_PASSWORD', ''),
}

# ============================================================================
# OPENAI API KONFIGURÁCIÓ
# ============================================================================

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Modell konfiguráció
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding modell
ASSISTANT_MODEL = "gpt-4o-mini"             # RAG válasz generálás (rögzített követelmény)
JUDGE_MODEL = "gpt-4o-mini"                 # LLM as a Judge modell (költséghatékony)

# Hőmérséklet beállítások
TEMPERATURE_ASSISTANT = 0.7   # Asszisztens válasz generálás (kreatív)
TEMPERATURE_JUDGE = 0.0       # Bírói értékelés (determinisztikus, reprodukálható)

# Token limitek
MAX_TOKENS_ASSISTANT = 500    # Maximum tokenek asszisztens válaszokhoz
MAX_TOKENS_JUDGE = 800        # Maximum tokenek bírói értékelésekhez (növelve a részletes érveléshez)

# ============================================================================
# RAG PIPELINE KONFIGURÁCIÓ
# ============================================================================

# Vektoros keresés paraméterek
TOP_K_CHUNKS = 3              # Visszaadott chunk-ok száma a vektoros keresésből
SIMILARITY_THRESHOLD = 0.3    # Minimum koszinusz hasonlósági küszöb

# Kontextus összeállítás
CONTEXT_TEMPLATE = """[Forrás {index}] (Hasonlóság: {similarity:.3f})
{content}
---"""

# ============================================================================
# ÉRTÉKELÉSI KONFIGURÁCIÓ
# ============================================================================

# Golden dataset paraméterek
GOLDEN_DATASET_SIZE_TARGET = 25  # Célzott Q&A párok száma
GOLDEN_DATASET_PATH = BASE_DIR / 'data' / 'golden_dataset.json'

# Kategória eloszlás (cél)
CATEGORY_DISTRIBUTION = {
    'factual': 5,           # Könnyű ténybeli felidézés
    'detail': 8,            # Közepes részlet kinyerés
    'comprehension': 8,     # Közepes-nehéz megértés
    'multi_chunk': 4        # Nehéz több chunk-os érvelés
}

# Forrásadat elérési útvonalak
RAG_EVAL_DIR = PROJECT_ROOT / 'rag-level-evaluation'
GENERATED_QUESTIONS_PATH = RAG_EVAL_DIR / 'data' / 'generated_questions.json'

# ============================================================================
# KIMENET ELÉRÉSI ÚTVONALAK
# ============================================================================

DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
CHARTS_DIR = RESULTS_DIR / 'charts'

# Könyvtárak létrehozása ha nem léteznek
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Kimeneti fájl elérési útvonalak
ASSISTANT_RESPONSES_PATH = DATA_DIR / 'assistant_responses.json'
CORRECTNESS_EVALUATIONS_PATH = DATA_DIR / 'correctness_evaluations.json'
RELEVANCE_EVALUATIONS_PATH = DATA_DIR / 'relevance_evaluations.json'
FINAL_REPORT_PATH = DATA_DIR / 'final_report.json'

# ============================================================================
# NAPLÓZÁSI KONFIGURÁCIÓ
# ============================================================================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# VALIDÁCIÓ
# ============================================================================

def validate_config():
    """
    Konfigurációs beállítások validálása.

    Raises:
        ValueError: Ha szükséges konfiguráció hiányzik vagy érvénytelen
    """
    errors = []

    # OpenAI API kulcs ellenőrzése
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY nincs beállítva a környezeti változókban")

    # Adatbázis jelszó ellenőrzése
    if not DB_CONFIG['password']:
        errors.append("DB_PASSWORD nincs beállítva a környezeti változókban")

    # Forrásadat elérési útvonalak ellenőrzése
    if not GENERATED_QUESTIONS_PATH.exists():
        errors.append(f"Generált kérdések fájl nem található: {GENERATED_QUESTIONS_PATH}")

    if errors:
        raise ValueError(f"Konfiguráció validálás sikertelen:\n" + "\n".join(f"  - {e}" for e in errors))

if __name__ == '__main__':
    # Konfiguráció tesztelése
    try:
        validate_config()
        print("✓ A konfiguráció érvényes")
        print(f"\nAdatbázis: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        print(f"Modellek: Asszisztens={ASSISTANT_MODEL}, Bíró={JUDGE_MODEL}")
        print(f"Top-K chunk-ok: {TOP_K_CHUNKS}")
    except ValueError as e:
        print(f"✗ Konfiguráció validálás sikertelen:\n{e}")
