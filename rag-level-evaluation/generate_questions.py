"""
Értékelési kérdések generálása dokumentum chunk-okból.

Ez a script az értékelési folyamat ELSŐ LÉPÉSE:
1. Lekéri az összes chunk-ot az adatbázisból
2. Minden chunk-hoz generál EGY kérdést GPT-4o mini használatával
3. A kérdéseket JSON fájlba menti későbbi értékeléshez

A generált kérdések később az evaluate_rag.py-ban kerülnek felhasználásra,
ahol teszteljük, hogy a RAG rendszer visszaadja-e az eredeti chunk-ot.

Futtatás:
    python3 generate_questions.py

Output:
    data/generated_questions.json

Megjegyzés:
    Ez a lépés időigényes lehet (1-2 mp / chunk), mert minden chunk-hoz
    egy OpenAI API hívást végez. 100 chunk esetén kb. 3-5 perc.
"""
import json
from tqdm import tqdm  # Progress bar library
from utils import get_db_connection, fetch_all_chunks, generate_question
from config import QUESTIONS_FILE
import sys


def main():
    """Fő függvény: kérdések generálása az összes chunk-hoz."""

    # ========================================================================
    # 1. KEZDŐ BANNER ÉS ÜDVÖZLÉS
    # ========================================================================
    print("=" * 80)
    print("RAG EVALUATION - QUESTION GENERATION")
    print("=" * 80)
    print()

    # ========================================================================
    # 2. ADATBÁZIS KAPCSOLÓDÁS
    # ========================================================================
    print("Connecting to database...")
    try:
        conn = get_db_connection()
        print("✓ Database connection established")
    except Exception as e:
        # Ha nem sikerül kapcsolódni, kilépünk hibaüzenettel
        print(f"✗ Failed to connect to database: {e}")
        sys.exit(1)  # Exit kód 1 = hiba történt

    # ========================================================================
    # 3. CHUNKÖK LEKÉRÉSE AZ ADATBÁZISBÓL
    # ========================================================================
    print("\nFetching document chunks...")
    try:
        # fetch_all_chunks() visszaadja az összes chunk-ot a document_chunks táblából
        chunks = fetch_all_chunks(conn)
        print(f"✓ Found {len(chunks)} chunks")
    except Exception as e:
        print(f"✗ Failed to fetch chunks: {e}")
        conn.close()  # Kapcsolat lezárása kilépés előtt
        sys.exit(1)

    # Ha nincs egyetlen chunk sem, kilépünk
    if len(chunks) == 0:
        print("\n⚠ No chunks found in database. Please run the chunking pipeline first.")
        conn.close()
        sys.exit(0)  # Exit kód 0 = normális kilépés (nincs hiba, csak nincs adat)

    # ========================================================================
    # 4. KÉRDÉSGENERÁLÁS MINDEN CHUNK-HOZ
    # ========================================================================
    print(f"\nGenerating questions (this may take a while)...")

    # Dictionary a kérdések tárolására: {chunk_id: {question, metadata, ...}}
    questions_data = {}

    # Lista a sikertelen chunk ID-khoz (hibakezelés céljából)
    failed_chunks = []

    # Iterálás minden chunk-on tqdm progress bar-ral
    # tqdm() automatikusan megjeleníti a haladást és becsült hátralévő időt
    for chunk in tqdm(chunks, desc="Generating questions"):
        chunk_id = str(chunk['id'])  # UUID string-gé konvertálása
        content = chunk['content']    # Chunk szöveges tartalma

        try:
            # Kérdés generálása GPT-4o mini-vel
            # Ez meghívja az OpenAI API-t, ezért ~1-2 másodpercig tart
            question = generate_question(content)

            # Kérdés és metaadatok tárolása
            # Minden mezőt eltárolunk, hogy később könnyű legyen elemezni
            questions_data[chunk_id] = {
                'chunk_id': chunk_id,           # Eredeti chunk ID
                'question': question,           # Generált kérdés
                'chunk_content': content,       # Chunk teljes tartalma (ground truth)
                'document_id': str(chunk['document_id']),  # Szülő dokumentum
                'chunk_index': chunk['chunk_index'],       # Pozíció a dokumentumban
                'token_count': chunk['token_count'],       # Token szám
                'metadata': chunk['metadata']              # JSONB metadata (chunking stratégia, stb.)
            }

        except Exception as e:
            # Ha hiba történt (pl. OpenAI API hiba, rate limit, stb.)
            # akkor rögzítjük, de folytatjuk a többivel
            print(f"\n⚠ Failed to generate question for chunk {chunk_id}: {e}")
            failed_chunks.append(chunk_id)

    # ========================================================================
    # 5. ADATBÁZIS KAPCSOLAT LEZÁRÁSA
    # ========================================================================
    # Már nem kell az adatbázis, bezárjuk a kapcsolatot
    conn.close()

    # ========================================================================
    # 6. KÉRDÉSEK MENTÉSE JSON FÁJLBA
    # ========================================================================
    print(f"\nSaving questions to {QUESTIONS_FILE}...")
    try:
        # JSON fájl írása UTF-8 encoding-gal
        with open(QUESTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(
                questions_data,
                f,
                indent=2,              # Szépen formázott JSON (2 szóköz behúzás)
                ensure_ascii=False     # Unicode karakterek megőrzése (nem \uXXXX)
            )
        print(f"✓ Saved {len(questions_data)} questions")
    except Exception as e:
        print(f"✗ Failed to save questions: {e}")
        sys.exit(1)

    # ========================================================================
    # 7. ÖSSZEFOGLALÓ STATISZTIKÁK
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total chunks: {len(chunks)}")
    print(f"Questions generated: {len(questions_data)}")
    print(f"Failed: {len(failed_chunks)}")

    # Ha voltak sikertelen chunk-ok, megjelenítjük az első 10-et
    if failed_chunks:
        print(f"\nFailed chunk IDs: {', '.join(failed_chunks[:10])}")
        if len(failed_chunks) > 10:
            print(f"... and {len(failed_chunks) - 10} more")

    # Fájl elérési út ismételt kiírása (kényelmi funkció)
    print(f"\nQuestions saved to: {QUESTIONS_FILE}")
    print()


# ============================================================================
# SCRIPT BELÉPÉSI PONT
# ============================================================================
# Ha ezt a fájlt közvetlenül futtatjuk (nem importálunk), akkor hívja meg a main()-t
if __name__ == '__main__':
    main()
