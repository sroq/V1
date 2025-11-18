# Single-Turn Evaluation - LLM as a Judge

Ez a könyvtár az RAG assistant **válaszminőségének** értékelésére szolgál **LLM as a Judge** módszerrel, golden dataset (ground truth Q&A párok) alapján.

## Áttekintés

A single-turn evaluation rendszer két fő metrikát mér:

1. **Correctness (Helyesség)**: Az AI válasz faktikai helyességét hasonlítja össze a ground truth válasszal
2. **Relevance (Relevancia)**: Az AI válasz relevanciáját értékeli a felhasznál

ói kérdéshez képest

### Evaluation Flow

```
Golden Dataset (20-25 Q&A pár)
    ↓
RAG Assistant Futtatás (embedding → search → response)
    ↓
LLM Judge Evaluation
    ├─ Correctness: generated_response vs ground_truth
    └─ Relevance: generated_response vs question
    ↓
Metrics & Visualization
```

## Követelmények

### Környezet

- Python 3.10+
- PostgreSQL + pgvector (meglévő RAG database)
- OpenAI API key

### Telepítés

```bash
cd single-turn-evaluation
pip install -r requirements.txt
```

### Környezeti Változók

A projekt gyökérkönyvtárában lévő `.env` fájlt használja (shared with `rag-level-evaluation`):

```env
# OpenAI API
OPENAI_API_KEY=sk-...

# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db
DB_USER=rag_user
DB_PASSWORD=...
```

## Használat

### 1. Golden Dataset Generálás

Először hozz létre egy 20-25 kérdés-válasz párt tartalmazó golden datasetet:

```bash
python3 scripts/1_generate_golden_dataset.py
```

Ez a script:
- Beolvassa a meglévő `generated_questions.json`-t
- Segít 20-25 reprezentatív kérdés kiválasztásában
- LLM-mel generál ground truth válaszokat
- Manuális review-t kér

**Output**: `data/golden_dataset.json`

### 2. Assistant Futtatás

Futtasd a RAG assistant-ot minden kérdésre:

```bash
python3 scripts/2_run_assistant.py
```

**Output**: `data/assistant_responses.json`

### 3. Correctness Evaluation

Értékeld a válaszok helyességét:

```bash
python3 scripts/3_evaluate_correctness.py
```

**Output**: `data/correctness_evaluations.json`

### 4. Relevance Evaluation

Értékeld a válaszok relevanciáját:

```bash
python3 scripts/4_evaluate_relevance.py
```

**Output**: `data/relevance_evaluations.json`

### 5. Eredmények Elemzése

Aggregáld a metrikákat és készíts vizualizációkat:

```bash
python3 scripts/5_analyze_results.py
```

**Output**:
- `data/final_report.json` (aggregált metrikák)
- `results/charts/*.png` (grafikonok)
- `results/summary_report.txt` (összefoglaló)

## Golden Dataset Struktúra

```json
{
  "metadata": {
    "dataset_id": "jungle_book_golden_v1",
    "created_at": "2025-11-11T19:00:00",
    "total_pairs": 25
  },
  "entries": [
    {
      "id": "q001",
      "question": "Who is the author of The Jungle Book?",
      "ground_truth_answer": "Rudyard Kipling. Published in 1894.",
      "source_chunk_id": "752db7d8-...",
      "category": "factual",
      "difficulty": "easy"
    }
  ]
}
```

### Kategóriák

- **factual** (5 kérdés): Egyszerű faktikus emlékezés
  - Példa: "Who is the author?"
- **detail** (8 kérdés): Részletes információ kinyerése
  - Példa: "What did Tabaqui find to eat?"
- **comprehension** (8 kérdés): Megértés, következtetés
  - Példa: "Why did Bagheera pay a bull?"
- **multi_chunk** (4 kérdés): Több chunk összekapcsolása
  - Példa: "How does training prepare Mowgli for confrontation?"

## Metrikák

### Correctness (Helyesség)

**Judge Prompt**: Ground truth vs AI válasz összehasonlítás

**Értékelés**:
- `CORRECT`: Faktikusan helyes, tartalmazza a kulcs információkat
- `INCORRECT`: Hibás, ellentmondásos vagy hiányos

**Output**: `{is_correct: bool, reasoning: str, decision: "CORRECT"|"INCORRECT"}`

### Relevance (Relevancia)

**Judge Prompt**: Kérdés vs AI válasz relevancia

**Értékelés**:
- `RELEVANT`: Válasz közvetlenül megválaszolja a kérdést
- `IRRELEVANT`: Válasz nem kapcsolódik a kérdéshez

**Output**: `{is_relevant: bool, reasoning: str, decision: "RELEVANT"|"IRRELEVANT"}`

### Aggregált Metrikák

- **Correctness Rate**: Helyes válaszok aránya (%)
- **Relevance Rate**: Releváns válaszok aránya (%)
- **By Category**: Metrikák kategóriánként (factual, detail, comprehension, multi_chunk)
- **By Difficulty**: Metrikák nehézségi szint szerint (easy, medium, hard)

## Példakód Adaptáció

Ez a rendszer a `/04-41-peldakok/02-code` példakód alapján készült, de adaptálva az RAG assistant architektúrára:

**Újrahasznált komponensek**:
- ✅ LLM as a Judge prompt pattern (CORRECTNESS + RELEVANCE)
- ✅ Golden dataset struktúra
- ✅ Evaluation pipeline flow

**Módosított komponensek**:
- ❌ VectorDB: Qdrant → PostgreSQL + pgvector
- ❌ Model: gpt-3.5-turbo → gpt-4o-mini
- ❌ Domain: Generic docs → The Jungle Book

## Költség

**Becslés (25 Q&A pár)**:
- Golden dataset generation: ~$0.0004
- Assistant responses: ~$0.0008
- Correctness evaluation: ~$0.0005
- Relevance evaluation: ~$0.0004
- **TOTAL**: ~$0.002 (< 1 cent!)

## Fejlesztői Megjegyzések

### Database Connection

A `llm_judge/database.py` újrahasználja a PostgreSQL connection logic-ot a `rag-level-evaluation` projektből.

### Prompts

A `llm_judge/prompts.py` tartalmazza a judge prompt template-eket. Ezek finomhangolhatók a domain-specifikus igényekhez.

### Assistant Runner

Az `llm_judge/assistant_runner.py` implementálja a teljes RAG pipeline-t:
1. Query embedding (OpenAI)
2. Vector search (PostgreSQL + pgvector)
3. Response generation (GPT-4o-mini)

## Troubleshooting

### "Configuration validation failed: OPENAI_API_KEY is not set"

Ellenőrizd, hogy a projekt gyökérkönyvtárában létezik-e a `.env` fájl és tartalmazza az `OPENAI_API_KEY` változót.

### "Generated questions file not found"

Futtasd előbb a `rag-level-evaluation` pipeline-t, hogy létrejöjjön a `generated_questions.json` fájl.

### "Database connection error"

Ellenőrizd, hogy a PostgreSQL fut-e és a credentials helyesek-e a `.env` fájlban.

## Kapcsolódó Projektek

- **rag-level-evaluation**: RAG retrieval metrikák (Hit Rate, MRR, Precision/Recall)
- **assistant**: Next.js frontend + RAG backend

## Licenc

Belső használatra, oktatási célokra.
