




# RAG-Based AI Assistant System - Complete Guide

Egy teljes körű RAG (Retrieval-Augmented Generation) alapú AI asszisztens rendszer dokumentum feldolgozás, vektoros keresés és összehangoló LLM-vel. Ez az útmutató végigvezet a telepítésen, konfiguráción és minden fő komponens használatán.

## 📋 Tartalomjegyzék

1. [Projekt Áttekintés](#projekt-áttekintés)
2. [Technológiai Stack](#technológiai-stack)
3. [Előfeltételek](#előfeltételek)
4. [Gyors Indítás](#gyors-indítás)
5. [Részletes Telepítési Útmutató](#részletes-telepítési-útmutató)
6. [1. Komponens: Dokumentumok Feltöltése](#1-komponens-dokumentumok-feltöltése)
7. [2. Komponens: AI Asszisztens](#2-komponens-ai-asszisztens)
8. [3. Komponens: Evaluáció](#3-komponens-evaluáció)
9. [Monitorozás és Költségkövetés](#monitorozás-és-költségkövetés)
10. [Hibaelhárítás](#hibaelhárítás)
11. [További Információk](#további-információk)

---

## Projekt Áttekintés

Ez a rendszer egy komplett RAG asszisztens infrastruktúra, amely képes:

- **Dokumentumok feldolgozása**: PDF, DOCX, TXT, Markdown, HTML formátumok támogatásával
- **Intelligens chunking**: 4 különböző stratégiával optimalizált szövegtöredékek
- **Vektoros keresés**: PostgreSQL + pgvector alapú hasonlósági keresés
- **LLM-alapú reranking**: Pontosság javítása felfelé mérésűséggel
- **RAG chat**: GPT-4o mini alapú valós idejű streaming válaszok
- **Teljes observability**: OpenTelemetry + Jaeger + Prometheus + Grafana nyomkövetés
- **Komprehenzív evaluáció**: Retrieval, response quality, és multi-turn conversation értékelés

**Dokumentum**: The Jungle Book (Rudyard Kipling)
**Felhasználási eset**: Kérdés-válasz asszisztens a könyv tartalmáról

---

## Technológiai Stack

### Backend
- **Database**: PostgreSQL 15+ + pgvector extension
- **Vector Storage**: pgvector (1536 dimenziók)
- **Document Processing**: Python + unstructured library
- **Embedding Model**: OpenAI text-embedding-3-small
- **LLM**: OpenAI GPT-4o mini
- **Deployment**: Docker + Docker Compose

### Frontend
- **Framework**: Next.js 15 (App Router)
- **UI**: React 18 + Tailwind CSS
- **AI Integration**: Vercel AI SDK
- **Language**: TypeScript

### Observability
- **Tracing**: OpenTelemetry + Jaeger
- **Metrics**: Prometheus
- **Visualization**: Grafana (8-panel cost tracking dashboard)
- **Cost Tracking**: OpenTelemetry metrics (embeddings, reranking, completions)

---

## Előfeltételek

### System Requirements
- **macOS/Linux/Windows** (WSL ajánlott Windows-on)
- **Docker** és **Docker Compose** (1.29.0+)
- **Node.js** 18.x vagy újabb (assistant komponenshez)
- **Python** 3.10+ (chunking pipeline-hez)
- **PostgreSQL client** (psql) - teszteléshez (opcionális)

### API Keys & Services
- **OpenAI API Key** - Embedding és chat completions
- **OpenAI Credits** - Szűks a feldolgozáshoz (~$0.01-0.05 per teljes cycle)

### Lemezterület
- **~5GB** Docker images és adatbázis számára
- **~1GB** Python függőségek

---

## Gyors Indítás

Teljes system indítása **5 perc alatt**:

### 1. Projekt másolása
```bash
cd /path/to/project
```

### 2. Environment konfigurálása
```bash
# Másolja az .env.example-t
cp .env.example .env

# Szerkessze a .env fájlt és adja meg az OpenAI API kulcsot
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

### 3. Docker konténerek indítása
```bash
docker-compose up -d
```

**Várjon 30-60 másodpercet, amíg az adatbázis inicializálódik.**

Ellenőrzés:
```bash
docker ps  # Lásd a futó konténereket

# PostgreSQL ellenőrzése
psql postgresql://rag_user:rag_dev_password_2024@localhost:5432/rag_assistant -c "SELECT COUNT(*) FROM document_chunks;"
```

### 4. AI Asszisztens indítása
```bash
cd assistant
npm install  # Első alkalommal
npm run dev
```

Az alkalmazás elérhető lesz: **http://localhost:3000**

### 5. Tesztelés
Írjon be egy kérdést:
- "Ki az a Mowgli?"
- "Mi az a Dzsungel Törvénye?"
- "Mesélj nekem Shere Khan-ról"

✅ **Kész!** A system működik.

---

## Részletes Telepítési Útmutató

### Lépés 1: Docker Konténerek Beállítása

#### A. PostgreSQL + pgvector

```bash
# Konténer indítása
docker-compose up -d postgres

# Ellenőrizze a konténer naplóit (erre 20-30s szükséges)
docker logs hf4-v1-postgres

# Kapcsolódjon az adatbázishoz
psql postgresql://rag_user:rag_dev_password_2024@localhost:5432/rag_assistant

# SQL parancsok az adatbázisban
CREATE EXTENSION vector;  -- pgvector extension

-- Documents tábla
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    created_at TIMESTAMP,
    modified_at TIMESTAMP,
    metadata JSONB,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document chunks tábla
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    metadata JSONB,
    token_count INTEGER,
    content_hash TEXT,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexek
CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops);
```

#### B. Observability Stack (Opcionális, de ajánlott)

Az observability komponensek már be vannak állítva a `docker-compose.yml`-ben:

```bash
# Jaeger (Tracing) - http://localhost:16686
docker-compose up -d jaeger

# Prometheus (Metrics) - http://localhost:9090
docker-compose up -d prometheus

# Grafana (Visualization) - http://localhost:3001
docker-compose up -d grafana

# OpenTelemetry Collector
docker-compose up -d otel-collector
```

#### C. Teljes Stack Indítása

```bash
# Minden konténer indítása
docker-compose up -d

# Naplók megtekintése
docker-compose logs -f postgres    # PostgreSQL naplók
docker-compose logs -f jaeger      # Jaeger naplók
docker-compose logs -f grafana     # Grafana naplók

# Leállítás (adatok megmaradnak)
docker-compose stop

# Teljes törlés (adatok elvesznek!)
docker-compose down -v
```

---

### Lépés 2: Python Chunking Pipeline Telepítése

#### A. Függőségek

```bash
cd chunking
pip install -r requirements.txt
```

#### B. Environment Konfigurálása

A `.env` fájl már tartalmaznia kell a szükséges konfigurációt a projekt gyökerében:

```bash
# Ellenőrizze/szerkessze a gyökér .env fájlt
cat ../.env | grep -E "OPENAI|DB_|DEFAULT"
```

Szükséges env változók:
```bash
OPENAI_API_KEY=sk-proj-...              # OpenAI API kulcs
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_assistant
DB_USER=rag_user
DB_PASSWORD=rag_dev_password_2024
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

#### C. Adatbázis Ellenőrzése

```bash
# Csatlakozzon az adatbázishoz
python -c "
import psycopg2
conn = psycopg2.connect(
    host='localhost',
    port=5432,
    database='rag_assistant',
    user='rag_user',
    password='rag_dev_password_2024'
)
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM document_chunks;')
print(f'Chunks: {cursor.fetchone()[0]}')
cursor.close()
"
```

---

### Lépés 3: Next.js Assistant Telepítése

#### A. Függőségek

```bash
cd assistant
npm install
```

#### B. Environment Konfigurálása

A `.env.local` már létezik az assistant könyvtárban:

```bash
# Ellenőrizze az .env.local fájlt
cat .env.local
```

Szükséges env változók:
```bash
OPENAI_API_KEY=sk-proj-...                                           # OpenAI API
DATABASE_URL=postgresql://rag_user:rag_dev_password_2024@localhost:5432/rag_assistant

OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSION=1536

DEFAULT_MATCH_COUNT=5
DEFAULT_MATCH_THRESHOLD=0.3
```

#### C. Build & Run

```bash
# Development módban
npm run dev
# Elérhető: http://localhost:3000

# Production build
npm run build
npm start
```

---

## 1. Komponens: Dokumentumok Feltöltése

### Áttekintés

A dokumentum feldolgozás pipeline a következő lépéseket hajtja végre:

```
Dokumentumok (PDF, DOCX, TXT, MD, HTML)
    ↓
Dokumentum Betöltés (unstructured library)
    ↓
Chunking (4 stratégia)
    ↓
Embedding Generálás (OpenAI text-embedding-3-small)
    ↓
PostgreSQL/pgvector Feltöltés
    ↓
RAG Kész Adatbázis
```

### Chunking Stratégiák

#### 1. Szemantikus Chunking (Ajánlott)
```bash
cd chunking
python chunker.py \
    --input ../Documents/ \
    --strategy semantic \
    --upload
```

**Előnyei**:
- Megőrzi a dokumentum struktúráját
- Tiszteletben tartja az értelmes határokat (bekezdések, fejlécek)
- Legjobb RAG teljesítmény

#### 2. Fix Méretű Chunking
```bash
python chunker.py \
    --input ../Documents/ \
    --strategy fixed \
    --chunk-size 512 \
    --chunk-overlap 50 \
    --upload
```

**Előnyei**:
- Konzisztens chunk méretek
- Kiszámítható token felhasználás
- Egyenletes feldolgozás

#### 3. Rekurzív Chunking
```bash
python chunker.py \
    --input ../Documents/ \
    --strategy recursive \
    --chunk-size 512 \
    --upload
```

**Előnyei**:
- Hierarchikus struktúra megőrzése
- Markdown/strukturált dokumentumokhoz jó
- Intelligens felosztás

#### 4. Dokumentum Típus Specifikus
```bash
python chunker.py \
    --input ../Documents/ \
    --strategy document_specific \
    --upload
```

**Előnyei**:
- Vegyes dokumentum típusok optimális kezelése
- Típus-tudatos feldolgozás
- Automatikus formátum felismerés

### Parancssor Opciók

| Opció | Leírás | Példa |
|-------|--------|--------|
| `--input` | **Kötelező**. Fájl vagy könyvtár | `--input documents/` |
| `--strategy` | Chunking stratégia | `--strategy semantic` |
| `--chunk-size` | Chunk méret (token) | `--chunk-size 512` |
| `--chunk-overlap` | Átvérés mértéke | `--chunk-overlap 50` |
| `--upload` | Feltöltés adatbázisba | `--upload` |
| `--batch-size` | Batch méret embedding-ekhez | `--batch-size 50` |
| `--clear-progress` | Progress törlése | `--clear-progress` |
| `--log-level` | Log szint | `--log-level DEBUG` |

### Haladó Opciók

```bash
# Meglévő progress folytatása
python chunker.py --input documents/ --upload

# Progress törlése és újraindítás
python chunker.py --input documents/ --clear-progress --upload

# Konfiguráció validálása
python chunker.py --input documents/ --validate

# Debug módban
python chunker.py --input documents/ --log-level DEBUG

# Egyéni batch méret
python chunker.py --input documents/ --batch-size 100 --upload
```

### Felhasználás Más Dokumentumokkal

```bash
# 1. Dokumentumokat másolja a Documents/ mappába
cp /path/to/your/document.pdf Documents/

# 2. Feldolgozás
cd chunking
python chunker.py \
    --input ../Documents/ \
    --strategy semantic \
    --upload

# 3. Előző dokumentumok eltávolítása (opcionális)
# Adatbázis frissítése - az UPDATE strategy beállítás a config.yaml-ben
# update_strategy: replace  (helyettesítés)
# update_strategy: version  (verzionálás)
# update_strategy: upsert   (módosítás)
```

### Figyelemmel Kísérés

```bash
# Progress file megtekintése
cat .chunking_progress.json

# Pipeline logok
tail -f chunking_pipeline.log

# Adatbázisban tárolt chunks
psql postgresql://rag_user:rag_dev_password_2024@localhost:5432/rag_assistant
> SELECT COUNT(*) FROM document_chunks;
```

### Költségbecslés

Az OpenAI embedding API költsége:

```python
from chunking.embeddings import EmbeddingGenerator

embedder = EmbeddingGenerator(api_key="your-key")
cost = embedder.estimate_cost(num_chunks=1000)
print(f"Becsült költség: ${cost['estimated_cost_usd']}")

# text-embedding-3-small: $0.02 / 1M token
# Átlagos chunk: ~400 token
# 1000 chunk ≈ $0.008
```

### Hibaelhárítás - Dokumentum Feltöltés

| Hiba | Megoldás |
|------|----------|
| `OPENAI_API_KEY not set` | Adja meg az API kulcsot a `.env` fájlban |
| `Database connection failed` | Ellenőrizze, hogy PostgreSQL fut-e (`docker-compose up -d postgres`) |
| `pgvector extension not installed` | SQL-ben: `CREATE EXTENSION vector;` |
| `Rate limit exceeded` | Csökkentse a batch_size-t a config.yaml-ben |
| `File too large` | Növelje a `max_file_size_mb`-t a konfigban |

---

## 2. Komponens: AI Asszisztens

### Áttekintés

A RAG chat asszisztens a feldolgozott dokumentumokból vektoros keresés segítségével gyűjt kontextust, és GPT-4o mini-vel generál válaszokat.

### RAG Pipeline

```
User Query
    ↓
Embedding Generation (OpenAI text-embedding-3-small)
    ↓
Vector Search (pgvector cosine similarity)
    ↓
Top 15 Chunk Retrieval
    ↓
LLM Reranking (GPT-4o mini pointwise scoring)
    ↓
Blended Scoring (70% LLM + 30% embedding)
    ↓
Top 5 Selection
    ↓
Context Assembly
    ↓
System Prompt Preparation
    ↓
GPT-4o mini Streaming Response
    ↓
Browser Display (Real-time)
```

### Telepítés & Futtatás

#### A. Függőségek Telepítése

```bash
cd assistant
npm install
```

#### B. Development Szerverindítás

```bash
npm run dev
# Elérhető: http://localhost:3000
```

#### C. Production Build

```bash
npm run build
npm start
```

### Használat

1. **Nyissa meg az alkalmazást**: http://localhost:3000
2. **Írjon be egy kérdést** az input mezőbe
3. **Vájon a streaming válaszra**

Példa kérdések:
- "Ki az a Mowgli és milyen a háttere?"
- "Mi az a Dzsungel Törvénye?"
- "Írj le a kapcsolatot Mowgli és Baloo között"
- "Ki az a Shere Khan és miért fontos?"
- "Milyen állatok élnek a dzsungelben ebben a történetben?"

### Testreszabás

#### Vektor Keresés Paraméterei

`.env.local`:
```env
DEFAULT_MATCH_COUNT=5          # Visszaadott chunks (1-20)
DEFAULT_MATCH_THRESHOLD=0.3    # Minimum similarity (0.0-1.0)
```

#### LLM Paraméterek

`assistant/app/api/chat/route.ts`:
```typescript
const result = streamText({
  model: openai('gpt-4o-mini'),
  messages: modelMessages,
  temperature: 0.7,      // Kreativitás (0.0-1.0)
  maxTokens: 1000,       // Maximum hossz
});
```

#### System Prompt

`assistant/lib/rag.ts` → `buildSystemPrompt()` függvény szerkesztése:
```typescript
export function buildSystemPrompt(context: string): string {
  return `Te egy segítőkész asszisztens vagy The Jungle Book könyvről.
Válaszolj magyar nyelven a következő kontextus alapján:

KONTEXTUS:
${context}

Kérlek, válaszolj pontosan és tömören.`;
}
```

### Backend Session Management

A rendszer automatikusan tárol minden beszélgetést az adatbázisban:

```bash
# Session históriájának megtekintése
psql postgresql://rag_user:rag_dev_password_2024@localhost:5432/rag_assistant

> SELECT session_id, role, content, created_at
  FROM chat_messages
  ORDER BY created_at DESC
  LIMIT 10;

# Session összefoglalók
> SELECT * FROM v_session_summary
  ORDER BY last_activity_at DESC
  LIMIT 5;
```

### Figyelemmel Kísérés

```bash
# API logok
npm run dev  # Kiírja az API hívásokat

# Chunk metaadat nézete
psql postgresql://rag_user:rag_dev_password_2024@localhost:5432/rag_assistant
> SELECT * FROM chat_rag_context
  WHERE chat_message_id = 'MESSAGE_ID'
  ORDER BY rank_position;
```

### Hibaelhárítás - AI Asszisztens

| Hiba | Megoldás |
|------|----------|
| "No relevant context found" | Csökkentse a `DEFAULT_MATCH_THRESHOLD`-ot |
| "Database connection failed" | Ellenőrizze a `DATABASE_URL` env változót |
| "OpenAI API error" | Validálja az `OPENAI_API_KEY`-t |
| "Streaming nem működik" | Ellenőrizze a böngésző konzolt (DevTools) |
| "Lassú válaszok" | Csökkentse a `DEFAULT_MATCH_COUNT`-ot |

---

## 3. Komponens: Evaluáció

A rendszer 3 szintű evaluációs rendszert biztosít a RAG minőség mérésére.

### 3.1 RAG-Level Evaluation (Retrieval Quality)

**Mit mér**: A vektoros keresés teljesítménye - hányszor találja meg az eredeti chunk-ot a kérdés alapján.

#### Telepítés

```bash
cd rag-level-evaluation
pip install -r requirements.txt
```

#### Futtatás

```bash
# Teljes pipeline
python3 run_evaluation.py

# Meglévő kérdések használata
python3 run_evaluation.py --skip-generation

# Kérdések újragenerálása
python3 run_evaluation.py --regenerate

# Csak elemzés
python3 run_evaluation.py --skip-generation --skip-evaluation
```

#### Lépésenkénti Futtatás

```bash
# 1. Kérdésgenerálás
python3 generate_questions.py

# 2. RAG értékelés
python3 evaluate_rag.py

# 3. Elemzés
python3 analyze_results.py
```

#### Kimeneti Metrikák

**Binary Single-Relevance Metrics**:
- **Hit Rate@K**: Az eredeti chunk megtalálható-e a top-K-ban?
- **First Position Accuracy**: Az eredeti chunk az első helyen van-e?
- **Average Rank**: Átlagos pozíció, ha megtalálható
- **Average Similarity**: Átlagos cosine similarity score

**Classical IR Metrics**:
- **MRR (Mean Reciprocal Rank)**: Az első releváns chunk átlagos reciprok rangja
- **Precision@K**: A top-K közül hány releváns (K=1,3,5,10)
- **Recall@K**: Az összes releváns közül hány van a top-K-ban

**Embedding Quality Metrics**:
- **Separation Margin**: Releváns vs irreleváns similarity különbsége
- **ROC-AUC**: Embedding model classification minősége (0-1)
- **Distribution Analysis**: Similarity score eloszlásai

**Chunk Quality Metrics**:
- **CSCI**: Chunk Size Consistency Index (konzisztencia)
- **RQCSB**: Retrieval Quality per Chunk Size Bucket (bucket teljesítmény)
- **PSS**: Position Stability Score (pozíció stabilitása)

#### Kimenetek

```
results/YYYYMMDD_HHMMSS/
├── summary.json              # Aggregált metrikák
├── detailed_results.csv      # Minden query részletei
├── metrics_by_strategy.csv   # Stratégiánkénti bontás
└── plots/
    ├── overall_metrics.png          # Binary metrics
    ├── rank_distribution.png        # Rank eloszlás
    ├── similarity_distribution.png  # Similarity eloszlás
    ├── metrics_by_strategy.png      # Stratégia összehasonlítás
    ├── precision_recall_curves.png  # P/R görbék
    ├── mrr_comparison.png           # MRR összehasonlítás
    ├── embedding_quality.png        # Embedding elválasztás
    ├── chunk_size_distribution.png  # Chunk méret eloszlás
    └── rqcsb_heatmap.png            # Bucket teljesítmény heatmap
```

#### Tipikus Eredmények

Egy jó RAG rendszernél várható:
- **Hit Rate@5**: 80-95%
- **First Position Accuracy**: 60-80%
- **MRR**: 0.65-0.85
- **Separation Margin**: > 0.2 (jó) vagy > 0.3 (kiváló)
- **ROC-AUC**: 0.8+ (jó) vagy 0.9+ (kiváló)

---

### 3.2 Single-Turn Evaluation (Response Quality)

**Mit mér**: A generált válaszok minősége - helyes-e és releváns-e az asszisztens válasza.

#### Telepítés

```bash
cd single-turn-evaluation
pip install -r requirements.txt
```

#### Futtatás

```bash
# Teljes pipeline (ajánlott)
python3 scripts/1_generate_golden_dataset.py   # Golden Q&A dataset
python3 scripts/2_run_assistant.py             # RAG futtatás
python3 scripts/3_evaluate_correctness.py      # Helyes-e?
python3 scripts/4_evaluate_relevance.py        # Releváns-e?
python3 scripts/5_analyze_results.py           # Elemzés & chartok
```

#### Kimeneti Metrikák

**Correctness**: Megegyezik-e az asszisztens válasza a ground truth-tal?
- **CORRECT**: Teljesen helyes válasz
- **INCORRECT**: Helytelen vagy hiányzó információ

**Relevance**: Releváns-e a válasz a kérdéshez?
- **RELEVANT**: Közvetlenül választ az asszisztens
- **IRRELEVANT**: Nem kapcsolódó tartalom

#### Kimenetek

```
single-turn-evaluation/
├── data/
│   ├── golden_dataset.json          # Generált Q&A párok (20-25 kérdés)
│   ├── assistant_responses.json     # Asszisztens válaszok
│   ├── correctness_evaluation.json  # Helyes/helytelen besorolás
│   └── relevance_evaluation.json    # Releváns/irreleváns besorolás
├── results/
│   ├── summary_report.md            # Text összefoglaló
│   ├── overall_metrics.png          # Metrika chart
│   ├── by_category.png              # Kategória szerinti bontás
│   └── by_difficulty.png            # Nehézség szerinti bontás
```

#### Tipikus Eredmények

Egy jó asszisztens várható teljesítménye:
- **Correctness Rate**: 75-95%
- **Relevance Rate**: 85-100%
- **Both Correct & Relevant**: 70-90%

---

### 3.3 Multi-Turn Evaluation (Conversation Quality)

**Mit mér**: A multi-turn beszélgetések minősége - mennyire jó az asszisztens többkörös interakcióban.

#### Telepítés

```bash
cd multi-turn-evaluation
pip install -r requirements.txt
```

#### Futtatás

```bash
# Teljes batch (30 persona-goal kombinácó)
python3 run_multi_turn_evaluation.py --batch

# Specifikus persona-goal kombináció
python3 run_multi_turn_evaluation.py --goal mowgli_identity --persona patient_intermediate

# Egyéni max körszám
python3 run_multi_turn_evaluation.py --goal jungle_ecosystem --persona curious_expert --max-turns 15
```

#### Elérhető Personas

| Persona | Türelmesség | Szakértelem | Jellemzés |
|---------|-------------|-------------|----------|
| `patient_intermediate` | Magas (3 rossz) | Közepes | Türelmes, közepes szintű tudás |
| `impatient_beginner` | Alacsony (1 rossz) | Kezdő | Türelmetlen, kevés előismeret |
| `curious_expert` | Nagyon magas (5+) | Szakértő | Kíváncsi, mély megértés |

#### Elérhető Goals

**Mintaként** (teljesebb lista a `goals.py` fájlban):
- `mowgli_identity`: Ki az a Mowgli?
- `jungle_ecosystem`: A dzsungel szerkezete
- `book_author`: A szerző információja
- `character_relationships`: Karakterek közötti kapcsolatok
- ...több mint 10 goal

#### Evaluáció Dimenziói

| Dimenzió | Mit mér | Súly |
|----------|---------|------|
| **Goal Achievement** | Elérte-e az asszisztens a célokat? | 40% |
| **Conversation Quality** | Koherencia, természetesség, info minőség | 20% |
| **Response Relevance** | Relevancia és pontosság | 20% |
| **User Experience** | Frustrációszint, persona megfelelőség | 10% |
| **Efficiency** | Körszám optimalizálása, redundancia | 10% |

#### Kimenetek

```
multi-turn-evaluation/results/
├── summary_table.csv                    # Összesített eredmények
├── persona_goal_results_TIMESTAMP.json  # Részletes JSON results
├── dimension_breakdown.png              # Dimenzió teljesítmény
└── goal_achievement_heatmap.png         # Goal elérési heatmap
```

#### Tipikus Eredmények

Egy jó multi-turn asszisztens várható:
- **Goal Achievement**: 70-90%
- **Conversation Quality**: 75-90%
- **Response Relevance**: 80-95%
- **Overall Score**: 75-85%

---

### Evaluáció Összehasonlítása

| Evaluáció Típus | Mit Mér | Költ | Futási Idő | Javasolt Gyakoriság |
|-----------------|---------|------|-----------|-------------------|
| **RAG-Level** | Retrieval teljesítmény | ~$0.00005 | 3-10 perc | Chunking módosítás után |
| **Single-Turn** | Response quality | ~$0.002 | 2-5 perc | Napi |
| **Multi-Turn** | Conversation quality | ~$0.001 | 5-15 perc | Heti |

---

## Monitorozás és Költségkövetés

### OpenTelemetry + Grafana

A rendszer teljes cost tracking és performance monitoring-ot biztosít.

#### Dashboards

**Jaeger (Distributed Tracing)**:
- URL: http://localhost:16686
- Mit mutat: RAG pipeline span-ok, latency breakdown
- Hasznos: Bottleneck azonosítás

**Prometheus (Metrics)**:
- URL: http://localhost:9090
- Mit mutat: Cost és token metrics
- Hasznos: Trend analízis

**Grafana (Visualization)**:
- URL: http://localhost:3001
- Bejelentkezés: `admin` / `admin`
- Dashboard: "RAG Assistant - Cost Tracking"
- Mutat: 8-panel cost breakdown

#### Elérhető Metrikák

```
# RAG Pipeline költségei
rag_assistant_rag_cost_embedding_USD_total
rag_assistant_rag_cost_reranking_USD_total
rag_assistant_rag_cost_chat_completion_USD_total

# Evaluáció költségei
rag_assistant_rag_cost_evaluation_llm_usd_USD_total

# Token felhasználás
rag_assistant_rag_tokens_embedding_total
rag_assistant_rag_tokens_llm_input_total
rag_assistant_rag_tokens_llm_output_total
```

#### Típikus Költségek

| Művelet | Költség | Leírás |
|--------|---------|--------|
| **RAG kérés** | $0.001-0.002 | Embedding + reranking + completion |
| **RAG-level eval** | $0.00002/kérdés | Question generation |
| **Single-turn eval** | $0.002/25 Q&A | 3 LLM judge hívás |
| **Multi-turn eval** | $0.001/konversáció | 5 LLM judge hívás |

### Grafana Dashboard Importálása

1. **Grafana megnyitása**: http://localhost:3001
2. **Dashboard Import**: Menü → Dashboards → Import
3. **JSON betöltése**: `grafana-dashboard-costs.json`
4. **Save**: Mentés az alapértelmezett datasource-val

---

## Hibaelhárítás

### Docker Problémák

#### Konténerek nem indulnak

```bash
# Ellenőrizze a naplókat
docker-compose logs postgres
docker-compose logs jaeger

# Köv logok követése
docker-compose logs -f

# Hardvér reset
docker-compose down -v
docker-compose up -d
```

#### Port már használatban

```bash
# Keresse meg a folyamatot
lsof -i :5432   # PostgreSQL
lsof -i :3001   # Grafana
lsof -i :16686  # Jaeger

# Állítsa le
kill -9 <PID>

# Vagy módosítsa a docker-compose.yml portokat
```

### Database Problémák

#### Nincs csatlakozás az adatbázishoz

```bash
# Tesztkezelés
psql -h localhost -p 5432 -U rag_user -d rag_assistant

# Jelszó beíráskor: rag_dev_password_2024
```

#### pgvector nincs telepítve

```bash
psql postgresql://rag_user:rag_dev_password_2024@localhost:5432/rag_assistant

> CREATE EXTENSION vector;
```

#### Nincs chunk az adatbázisban

```bash
# Chunking futtatása
cd chunking
python chunker.py --input ../Documents/ --strategy semantic --upload

# Ellenőrizés
psql postgresql://rag_user:rag_dev_password_2024@localhost:5432/rag_assistant
> SELECT COUNT(*) FROM document_chunks;
```

### Python/Chunking Problémák

#### ImportError: unstructured

```bash
pip install unstructured[all-docs]
# Ha az továbbra sem működik:
pip install pdf2image pdfplumber pillow
```

#### OpenAI API Hiba

```bash
# Ellenőrizze az API kulcsot
echo $OPENAI_API_KEY

# Validáció
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

#### Rate Limit

```bash
# Csökkentse a batch size-t
python chunker.py --input documents/ --batch-size 20 --upload
```

### Next.js Problémák

#### npm install hiba

```bash
# Clean install
rm -rf node_modules package-lock.json
npm install
```

#### Build hiba

```bash
# Clear Next cache
rm -rf .next
npm run build
```

#### Streaming nem működik

```bash
# Ellenőrizze az API route-ot
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "test"}]}'
```

### Evaluation Problémák

#### "No chunks found"

```bash
# Győződjön meg, hogy van chunk az adatbázisban
python -c "
import psycopg2
conn = psycopg2.connect('postgresql://rag_user:rag_dev_password_2024@localhost:5432/rag_assistant')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM document_chunks')
print(f'Total chunks: {cur.fetchone()[0]}')
"
```

#### Lassú evaluáció

- **RAG-Level**: Normális 1-2s/chunk (OpenAI API)
- **Single-Turn**: Normális 3-5 perc / 25 kérdés
- **Multi-Turn**: Normális 2-5 perc / konversáció

#### OpenAI API Limit

```bash
# Rate limit kezelés - automatikus retry logika
# Ha továbbra is problémás, csökkentsen a API hívások számán
```

---

## További Információk

### Dokumentáció

- **CLAUDE.md**: Claude Code projekt instrukcciók
- **REQUIREMENTS.md**: Részletes technikai követelmények
- **SESSION_MANAGEMENT.md**: Conversation history implementáció
- **chunking/README.md**: Chunking pipeline dokumentáció
- **assistant/README.md**: AI Asszisztens dokumentáció
- **rag-level-evaluation/README.md**: Retrieval evaluáció
- **single-turn-evaluation/README.md**: Response evaluáció
- **multi-turn-evaluation/README.md**: Conversation evaluáció

### Projekt Szerkezet

```
/
├── chunking/                    # Dokumentum feldolgozás
├── assistant/                   # Next.js chat UI
├── rag-level-evaluation/        # Retrieval értékelés
├── single-turn-evaluation/      # Response értékelés
├── multi-turn-evaluation/       # Conversation értékelés
├── database/                    # DB schema & migrations
├── Documents/                   # Input dokumentumok (The Jungle Book)
├── docker-compose.yml           # Docker konfigurálás
├── .env                         # Environment változók
├── prometheus.yml               # Prometheus config
├── otel-collector-config.yaml   # OpenTelemetry config
├── grafana-dashboard-costs.json # Grafana dashboard
├── REQUIREMENTS.md              # Követelmények
├── CLAUDE.md                    # Claude instrukciók
└── README.md                    # Ez a fájl
```

### Legjobb Gyakorlatok

#### Dokumentum Feldolgozás
1. Válassza a **semantic** chunking-ot a legtöbb esetben
2. **Tesztelje** pequeño dokumentumon előbb
3. **Figyelmmel kísérje** az embedding költségeket
4. **Validálja** a chunk minőséget előbb

#### AI Asszisztens
1. **Hangoljon** a `DEFAULT_MATCH_THRESHOLD` értéken
2. **Teszteljen** különböző system prompt-okkal
3. **Monitorozzon** a Grafana dashboardon
4. **Optimalizáljon** reranking paramétereken

#### Evaluáció
1. **Futtassa először** a RAG-level evaluációt
2. **Majd** a Single-turn evaluációt
3. **Végül** a Multi-turn evaluációt
4. **Iteráljon** a meglépések alapján

### Szokásos Workflow

```
1. SETUP (Egyszeri)
   └─ Docker containers
   └─ Python & Node.js environment

2. INGESTION (Dokumentum feltöltés)
   └─ Documents másolása Documents/
   └─ Chunking pipeline futtatása
   └─ Adatbázis ellenőrzése

3. TESTING (AI asszisztens)
   └─ npm run dev
   └─ Kérdések tesztelése
   └─ Response minőség ellenőrzése

4. EVALUATION (Minőség mérés)
   └─ RAG-level evaluation
   └─ Single-turn evaluation
   └─ Multi-turn evaluation

5. OPTIMIZATION (Finomhangolás)
   └─ Metrikai alapján módosítás
   └─ Iterálás 3-4 között
   └─ Költségkövetés (Grafana)

6. MONITORING (Termelés)
   └─ Jaeger: trace Analysis
   └─ Prometheus: metric trends
   └─ Grafana: cost dashboard
```

### Support & Troubleshooting

**Probléma?**
1. Nézze meg a fenti Hibaelhárítás szekciót
2. Ellenőrizze a komponens-specifikus README-ket
3. Nézze meg a logokat: `docker logs`, `npm run dev`, `tail -f chunking_pipeline.log`
4. Tesztelje az API-t curl-lel vagy Postman-nel

**API Tesztelés**:
```bash
# RAG Chat API
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Who is Mowgli?"
      }
    ]
  }'
```

---

## Version History

- **1.0.0** (2025-11-18): Teljes dokumentáció
  - Gyors indítás útmutató
  - 3 komponens részletes útmutató
  - Monitorozás és költségkövetés
  - Hibaelhárítás
  - Best practices

---

## Licensz

Ez a projekt az AI asszisztens fejlesztési projekt része.

---

## Kontakt & Támogatás

Kérdések vagy problémák esetén:
1. Ellenőrizze a fenti dokumentációt
2. Keresse meg az [Hibaelhárítás](#hibaelhárítás) szekciót
3. Nézze meg a komponens-specifikus README fájlokat
4. Ellenőrizze a logokat Debug módban

**Konfigurálható komponensek**:
- Chunking stratégia
- Vector search paraméterek
- LLM prompt és beállítások
- Evaluáció konfigurálása

**Monitoring&Observability**:
- Jaeger: http://localhost:16686
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

---

**Jó munkát! 🚀**
