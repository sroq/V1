"""
OpenAI API utility függvények embedding generáláshoz és kérdésgeneráláshoz.

Ez a modul biztosítja az OpenAI API-val való kommunikációt:
- Embedding generálás (text-embedding-3-small modell)
- Kérdésgenerálás chunk-okból (GPT-4o mini modell)
"""
from openai import OpenAI
from typing import List
import sys
import os

# Szülő könyvtár hozzáadása a path-hoz config importáláshoz
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_CHAT_MODEL,
    QUESTION_GENERATION_PROMPT
)

# Import cost tracking
from cost_metrics import get_cost_tracker

# OpenAI kliens inicializálása az API kulccsal
# Ez a kliens objektum használható az összes OpenAI API híváshoz
client = OpenAI(api_key=OPENAI_API_KEY)

# Cost tracking inicializálása
cost_tracker = get_cost_tracker()


def generate_embedding(text: str) -> List[float]:
    """
    Embedding (vektor reprezentáció) generálása szöveghez OpenAI API-val.

    Ez a függvény a text-embedding-3-small modellt használja, amely
    1536 dimenziós vektorokat generál. Ezek a vektorok szemantikai hasonlóságot
    reprezentálnak - hasonló jelentésű szövegek hasonló vektorokat kapnak.

    Args:
        text: A szöveg, amihez embedding-et szeretnénk generálni.
              Lehet kérdés, válasz, dokumentum chunk, stb.

    Returns:
        Lista 1536 float számmal, amely az embedding vektor.
        Példa: [0.1234, -0.5678, 0.9012, ..., 0.3456]

    Raises:
        Exception: Ha hiba történik az OpenAI API hívás során
                   (pl. hálózati hiba, érvénytelen API kulcs, quota limit)

    Költség:
        text-embedding-3-small: ~$0.02 / 1M token
        Átlagos szöveg: 100-500 token
        100 embedding: ~$0.001 (elhanyagolható)

    Példa:
        >>> embedding = generate_embedding("Who is Mowgli?")
        >>> print(f"Vektor dimenziók száma: {len(embedding)}")
        1536
        >>> print(f"Első 5 érték: {embedding[:5]}")
        [0.1234, -0.5678, 0.9012, -0.2345, 0.6789]
    """
    try:
        # OpenAI Embeddings API hívás
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,  # "text-embedding-3-small"
            input=text  # A szöveg, amihez embedding-et generálunk
        )
        # Az API válaszból az embedding vektor kinyerése
        # response.data[0] az első (és egyetlen) embedding a válaszban
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Hiba az embedding generálása során: {e}")
        raise


def generate_question(chunk_content: str) -> str:
    """
    Kérdés generálása egy chunk szövegéből GPT-4o mini használatával.

    Ez a függvény a QUESTION_GENERATION_PROMPT template-et használja
    egy instruction promptot létrehozni, amely megmondja a modellnek,
    hogy milyen kérdést generáljon.

    Args:
        chunk_content: A chunk szöveges tartalma, amiből kérdést generálunk.
                       Optimális hossz: 200-1000 karakter

    Returns:
        A generált kérdés string formátumban.
        Példa: "Who is Mowgli in The Jungle Book?"

    Raises:
        Exception: Ha hiba történik az OpenAI API hívás során

    Költség:
        GPT-4o mini: ~$0.15 / 1M input tokens, ~$0.60 / 1M output tokens
        Átlagos chunk: 500 tokens input
        Átlagos kérdés: 20 tokens output
        100 kérdés: ~$0.01 (nagyon olcsó)

    Megjegyzések:
        - temperature=0.7: Közepes kreativitás (0.0=determinisztikus, 1.0=nagyon kreatív)
        - max_tokens=150: Maximum 150 token hosszú kérdés (általában ~20-50 token)
        - A prompt angol nyelvű, de a kérdés nyelve a chunk nyelvétől függ

    Példa:
        >>> chunk = "Mowgli is a human child who was raised by wolves..."
        >>> question = generate_question(chunk)
        >>> print(question)
        "Who is Mowgli and how was he raised?"
    """
    try:
        # A prompt template kitöltése a chunk tartalmával
        # {chunk_content} helyére behelyettesítjük a tényleges chunk szöveget
        prompt = QUESTION_GENERATION_PROMPT.format(chunk_content=chunk_content)

        # GPT-4o mini Chat Completions API hívás
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,  # "gpt-4o-mini"
            messages=[
                {"role": "user", "content": prompt}  # User message: kérjük a kérdést
            ],
            temperature=0.7,  # Kreativitás szintje (0.0-1.0)
            max_tokens=150    # Maximum válaszhossz tokenekben
        )

        # Track cost
        if response.usage:
            cost_tracker.record_llm_cost(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                attributes={"operation": "question_generation"}
            )

        # A generált kérdés kinyerése a válaszból
        # response.choices[0] az első (és egyetlen) választási lehetőség
        # .message.content a tényleges generált szöveg
        # .strip() eltávolítja a whitespace karaktereket elejéről/végéről
        question = response.choices[0].message.content.strip()
        return question
    except Exception as e:
        print(f"Hiba a kérdés generálása során: {e}")
        raise


def batch_generate_embeddings(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    Embedding-ek generálása több szöveghez batch-ekben (kötegekben).

    Ez a függvény hatékonyabb, mint egyenként hívni a generate_embedding-et,
    mert az OpenAI API támogatja a batch embedding generálást, ami:
    - Gyorsabb (egy API hívás batch_size szöveghez)
    - Kevesebb hálózati overhead
    - Ugyanaz a költség, mint egyenként

    Args:
        texts: Lista string-ekkel, amelyekhez embedding-et generálunk.
               Példa: ["Who is Mowgli?", "What is the Law of the Jungle?", ...]
        batch_size: Hány szöveget dolgozzunk fel egy batch-ben.
                    Default: 100 (OpenAI ajánlás: max 100-200)

    Returns:
        Lista embedding vektorokkal, ugyanabban a sorrendben, mint a texts.
        Példa: [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
        len(result) == len(texts)

    Raises:
        Exception: Ha hiba történik bármelyik batch feldolgozása során.
                   A hiba esetén a teljes folyamat leáll.

    Teljesítmény:
        - 100 szöveg batch-enként: ~1 API hívás, ~2-3 másodperc
        - 1000 szöveg: ~10 API hívás, ~20-30 másodperc

    Példa:
        >>> questions = ["Who is Mowgli?", "What is the jungle?", ...]
        >>> embeddings = batch_generate_embeddings(questions, batch_size=50)
        >>> print(f"{len(embeddings)} embedding generálva")
        >>> print(f"Első embedding dimenziók: {len(embeddings[0])}")
        100 embedding generálva
        Első embedding dimenziók: 1536
    """
    embeddings = []  # Üres lista az eredmények gyűjtéséhez

    # Végigiterálunk a szövegeken batch_size méretű lépésekkel
    # range(0, len(texts), batch_size) generálja a kezdő indexeket
    # Példa: len(texts)=250, batch_size=100 → i = 0, 100, 200
    for i in range(0, len(texts), batch_size):
        # Az aktuális batch kinyerése: texts[i:i+batch_size]
        # Példa: i=0, batch_size=100 → texts[0:100] (első 100 elem)
        batch = texts[i:i + batch_size]

        try:
            # OpenAI Embeddings API hívás a teljes batch-re
            # Az API támogatja a lista inputot, és lista outputot ad vissza
            response = client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=batch  # Lista string-ekkel (nem egyetlen string!)
            )

            # Az összes embedding kinyerése a válaszból
            # response.data egy lista EmbeddingObject-ekkel
            # Minden item.embedding egy vektor (lista float-okkal)
            batch_embeddings = [item.embedding for item in response.data]

            # A batch eredményeinek hozzáadása a teljes eredménylistához
            # extend() lista hozzáadása (nem append, ami listát ad listához)
            embeddings.extend(batch_embeddings)

        except Exception as e:
            # Batch szám számítása hibaüzenethez: i // batch_size + 1
            # Példa: i=200, batch_size=100 → 200 // 100 + 1 = 3. batch
            print(f"Hiba az embeddings batch generálása során ({i // batch_size + 1}. batch): {e}")
            raise

    return embeddings
