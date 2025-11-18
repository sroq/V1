"""
Konfigurációs beállítások a RAG értékelő rendszerhez.

Ez a fájl tartalmazza az összes konstanst és konfigurációs paramétert,
amely az értékelési folyamathoz szükséges.
"""
import os
from dotenv import load_dotenv

# Környezeti változók betöltése a szülő könyvtár .env fájljából
# Ez lehetővé teszi, hogy az értékelő rendszer ugyanazokat a beállításokat használja,
# mint a fő RAG alkalmazás
parent_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(parent_env_path)

# ============================================================================
# Adatbázis Konfiguráció
# ============================================================================
# PostgreSQL kapcsolati paraméterek
# Ezeket a .env fájlból olvassuk be, de alapértelmezett értékeket is megadunk
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),  # Szerver címe
    'port': int(os.getenv('POSTGRES_PORT', 5432)),    # Port szám (alapértelmezett: 5432)
    'dbname': os.getenv('POSTGRES_DB', 'rag_assistant'),  # Adatbázis neve
    'user': os.getenv('POSTGRES_USER', 'rag_user'),       # Felhasználónév
    'password': os.getenv('POSTGRES_PASSWORD', 'rag_dev_password_2024'),  # Jelszó
}

# ============================================================================
# OpenAI Konfiguráció
# ============================================================================
# OpenAI API kulcs - KÖTELEZŐ! Nélküle nem fog működni a rendszer
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Embedding modell neve
# text-embedding-3-small: 1536 dimenziós vektorokat generál
OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')

# Embedding vektor dimenziók száma
# FONTOS: Ennek egyeznie kell az adatbázisban tárolt vektorok méretével!
OPENAI_EMBEDDING_DIMENSION = int(os.getenv('OPENAI_EMBEDDING_DIMENSION', 1536))

# Chat modell a kérdésgeneráláshoz
# GPT-4o mini: Gyors, költséghatékony, de jó minőségű kérdéseket generál
OPENAI_CHAT_MODEL = 'gpt-4o-mini'

# ============================================================================
# RAG Konfiguráció
# ============================================================================
# Hány darab chunk-ot kérjünk le egy kereséskor (top-K)
# Alapértelmezetten 5 - ez a "K" érték a Top-K Recall metrikában
DEFAULT_MATCH_COUNT = int(os.getenv('DEFAULT_MATCH_COUNT', 5))

# Minimális hasonlósági küszöb (similarity threshold)
# 0.0 = minden chunk elfogadott
# 1.0 = csak tökéletes egyezés
# 0.3 = kiegyensúlyozott beállítás (jó recall, elfogadható precision)
# FONTOS: Ez a paraméter döntően befolyásolja az értékelési eredményeket!
DEFAULT_MATCH_THRESHOLD = float(os.getenv('DEFAULT_MATCH_THRESHOLD', 0.3))

# ============================================================================
# Értékelési Konfiguráció
# ============================================================================
# Prompt template a kérdésgeneráláshoz
# Ez a prompt utasítja a GPT-4o mini-t, hogy hogyan generáljon kérdést egy chunk-ból
QUESTION_GENERATION_PROMPT = """Based on the following text chunk, generate ONE specific question that can only be answered using this exact content.

The question should be:
- Natural and conversational
- Require the information in this chunk to answer
- Not too broad or too narrow
- Answerable with the provided content

Text chunk:
{chunk_content}

Generate only the question, without any preamble or explanation."""

# ============================================================================
# Fájl Útvonalak
# ============================================================================
# Az értékelési rendszer által használt könyvtárak és fájlok
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')  # Generált adatok könyvtára
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')  # Eredmények könyvtára
QUESTIONS_FILE = os.path.join(DATA_DIR, 'generated_questions.json')  # Generált kérdések fájlja
EVALUATION_RESULTS_FILE = os.path.join(DATA_DIR, 'evaluation_results.json')  # Értékelési eredmények fájlja

# Könyvtárak létrehozása, ha még nem léteznek
# exist_ok=True: nem dob hibát, ha már létezik
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
