# RAG-Level Evaluation System

Értékelő rendszer RAG (Retrieval-Augmented Generation) rendszerek minőségének mérésére RAG-szintű (retrieval teljesítmény) single-turn kérdés-válasz forgatókönyv alapján.

##概要 (Áttekintés)

Ez az értékelő rendszer a következő folyamatot hajtja végre:

1. **Kérdésgenerálás**: Minden dokumentum chunk-ból egy kérdést generálunk GPT-4o mini használatával
2. **RAG Értékelés**: A generált kérdéseket feltesszük a RAG rendszernek és megvizsgáljuk, hogy visszakapjuk-e az eredeti chunk-ot
3. **Metrikák**: Kétféle metrika csoportot mérünk:
   - **Binary Single-Relevance Metrics**:
     - **Hit Rate@K**: Az eredeti chunk benne van-e a top-K lekért chunk-ban?
     - **First Position Accuracy**: Az eredeti chunk az első helyen van-e?
   - **Classical IR Metrics**:
     - **MRR (Mean Reciprocal Rank)**: Átlagos reciprok rank
     - **Precision@K**: Precision különböző K értékekre (K=1,3,5,10)
     - **Recall@K**: Recall különböző K értékekre (K=1,3,5,10)
4. **Elemzés**: Részletes statisztikai elemzés és vizualizációk készítése

## Projekt Struktúra

```
rag-level-evaluation/
├── README.md                     # Ez a fájl
├── requirements.txt              # Python függőségek
├── config.py                     # Konfigurációs beállítások
│
├── generate_questions.py         # 1. lépés: Kérdésgenerálás
├── evaluate_rag.py              # 2. lépés: RAG értékelés
├── analyze_results.py           # 3. lépés: Eredmények elemzése
├── run_evaluation.py            # Teljes pipeline futtatása
│
├── utils/                       # Utility modulok
│   ├── __init__.py
│   ├── db_utils.py             # Adatbázis műveletek
│   ├── openai_utils.py         # OpenAI API hívások
│   └── metrics.py              # Metrika kalkulációk
│
├── data/                        # Generált adatok
│   ├── generated_questions.json
│   └── evaluation_results.json
│
└── results/                     # Timestamped eredmények
    └── YYYYMMDD_HHMMSS/
        ├── summary.json
        ├── detailed_results.csv
        ├── metrics_by_strategy.csv
        └── plots/
            ├── overall_metrics.png          # Binary metrics (Hit Rate@K, First Position Accuracy)
            ├── rank_distribution.png
            ├── similarity_distribution.png
            ├── metrics_by_strategy.png
            ├── precision_recall_curves.png  # Precision@K és Recall@K görbék (ÚJ)
            └── mrr_comparison.png           # MRR összehasonlítás stratégiánként (ÚJ)
```

## Telepítés

### 1. Python környezet létrehozása (opcionális, de ajánlott)

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# vagy
venv\Scripts\activate     # Windows
```

### 2. Függőségek telepítése

```bash
cd rag-level-evaluation
pip install -r requirements.txt
```

### 3. Környezeti változók

Az értékelő rendszer a szülő könyvtár `.env` fájlját használja. Győződj meg róla, hogy a következő változók be vannak állítva:

```bash
# OpenAI API
OPENAI_API_KEY=sk-proj-...

# PostgreSQL + pgvector
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_assistant
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_dev_password_2024

# Embedding konfiguráció
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSION=1536

# RAG konfiguráció
DEFAULT_MATCH_COUNT=5
DEFAULT_MATCH_THRESHOLD=0.3
```

## Használat

### Teljes pipeline futtatása (ajánlott)

```bash
python3 run_evaluation.py
```

Ez a parancs lefuttatja az összes lépést:
1. Kérdésgenerálás a chunk-okból
2. RAG retrieval értékelés
3. Eredmények elemzése és vizualizáció

### Opciók

```bash
# Meglévő kérdések használata (újragenerálás nélkül)
python3 run_evaluation.py --skip-generation

# Kérdések újragenerálása
python3 run_evaluation.py --regenerate

# Csak elemzés futtatása (meglévő eredményekből)
python3 run_evaluation.py --skip-generation --skip-evaluation
```

### Lépésenkénti futtatás

Ha külön-külön szeretnéd futtatni a lépéseket:

```bash
# 1. Kérdésgenerálás
python3 generate_questions.py

# 2. RAG értékelés
python3 evaluate_rag.py

# 3. Eredmények elemzése
python3 analyze_results.py
```

## Kimenet

### 1. Generált kérdések (`data/generated_questions.json`)

```json
{
  "chunk-uuid-1": {
    "chunk_id": "chunk-uuid-1",
    "question": "Who is Mowgli in The Jungle Book?",
    "chunk_content": "Mowgli is a human child...",
    "document_id": "doc-uuid",
    "chunk_index": 0,
    "token_count": 256,
    "metadata": {
      "chunking_strategy": "semantic"
    }
  }
}
```

### 2. Értékelési eredmények (`data/evaluation_results.json`)

```json
[
  {
    "chunk_id": "chunk-uuid-1",
    "question": "Who is Mowgli?",
    "retrieved_chunk_ids": ["chunk-uuid-1", "chunk-uuid-2", ...],
    "retrieved_similarities": [0.92, 0.85, ...],
    "metrics": {
      "hit_rate_at_k": 1,
      "first_position_accuracy": 1,
      "rank": 1,
      "similarity_score": 0.92,
      "found": true,
      "precision_at_1": 1.0,
      "precision_at_3": 0.33,
      "precision_at_5": 0.20,
      "precision_at_10": 0.10,
      "recall_at_1": 1.0,
      "recall_at_3": 1.0,
      "recall_at_5": 1.0,
      "recall_at_10": 1.0,
      "top_k_recall": 1,
      "top_1_precision": 1
    },
    "chunking_strategy": "semantic",
    "token_count": 256
  }
]
```

### 3. Összesített eredmények (`results/YYYYMMDD_HHMMSS/`)

- **summary.json**: Aggregált metrikák
  - **Binary Single-Relevance Metrics**:
    - Hit Rate@K % (korábban: Top-K Recall)
    - First Position Accuracy % (korábban: Top-1 Precision)
    - Átlagos rank (ha megtalálható)
    - Átlagos similarity score
  - **Classical IR Metrics**:
    - MRR (Mean Reciprocal Rank)
    - Avg Precision@1, @3, @5, @10
    - Avg Recall@1, @3, @5, @10
  - **Backward Compatibility**: Régi metrika nevek (top_k_recall_percentage, top_1_precision_percentage) megtartva aliasként

- **detailed_results.csv**: Minden kérdés részletei táblázatos formában (mind a binary, mind a classical IR metrikákkal)

- **metrics_by_strategy.csv**: Metrikák chunking stratégia szerint bontva (mind a binary, mind a classical IR metrikákkal)

- **plots/**: Vizualizációk
  - `overall_metrics.png`: Binary metrics (Hit Rate@K, First Position Accuracy)
  - `rank_distribution.png`: Rank eloszlás
  - `similarity_distribution.png`: Similarity score eloszlás
  - `metrics_by_strategy.png`: Binary metrics teljesítmény stratégiánként
  - `precision_recall_curves.png`: **ÚJ** - Precision@K és Recall@K görbék különböző K értékekre
  - `mrr_comparison.png`: **ÚJ** - MRR összehasonlítás stratégiánként

## Metrikák Magyarázata

Az értékelési rendszer kétféle metrika csoportot számol: **Binary Single-Relevance Metrics** és **Classical IR Metrics**.

### Binary Single-Relevance Metrics

Ezek a metrikák egyetlen releváns chunk létezését feltételezik (az eredeti chunk, amiből a kérdést generáltuk), és bináris (0/1) eredményt adnak.

#### Hit Rate@K (korábban: Top-K Recall)
Az eredeti chunk benne van-e a top-K (alapértelmezetten 5) lekért chunk között?

- **1 (100%)**: Az eredeti chunk megtalálható a top-K-ban
- **0 (0%)**: Az eredeti chunk nincs a top-K-ban

**Aggregált érték**: Az összes kérdés átlaga százalékban (0-100%)

#### First Position Accuracy (korábban: Top-1 Precision)
Az eredeti chunk az első helyen van-e a lekért chunkok között?

- **1 (100%)**: Az eredeti chunk az első helyen van (legjobb match)
- **0 (0%)**: Az eredeti chunk nincs első helyen

**Aggregált érték**: Az összes kérdés átlaga százalékban (0-100%)

#### Rank
Ha az eredeti chunk megtalálható, hányadik helyen van (1-indexelt)?

- **1**: Első helyen (legjobb match)
- **2-5**: A top-5-ben, de nem első
- **None**: Nincs a top-K-ban

**Aggregált érték**: Átlagos rank (csak ahol megtalálható volt)

#### Similarity Score
Ha az eredeti chunk megtalálható, mekkora a cosine similarity score?

- **0.9-1.0**: Nagyon hasonló
- **0.7-0.9**: Közepesen hasonló
- **0.3-0.7**: Gyengén hasonló
- **None**: Nincs a threshold felett

**Aggregált érték**: Átlagos similarity (csak ahol megtalálható volt)

---

### Classical IR Metrics

Ezek a klasszikus Information Retrieval metrikák, amelyek a lekért dokumentumok halmaza és a releváns dokumentumok halmaza közötti átfedést mérik.

#### MRR (Mean Reciprocal Rank)
Az első releváns találat átlagos reciprok rangja. Azt méri, hogy átlagosan mennyire "előre" van az első releváns chunk.

**Formula**: `MRR = 1/N × Σ(1/rank_i)`

ahol `rank_i` az első releváns chunk pozíciója az i-edik query esetén.

**Értelmezés**:
- **1.0**: Minden esetben az első helyen volt a releváns chunk (tökéletes)
- **0.5**: Átlagosan a 2. helyen volt
- **0.33**: Átlagosan a 3. helyen volt
- **0.0**: Egyik esetben sem volt releváns chunk a top-K-ban

#### Precision@K
A top-K lekért chunk közül hány releváns (hányad rész)?

**Formula**: `Precision@K = (releváns ÉS lekért top-K) / K`

**Értelmezés single-relevance esetén** (1 releváns chunk):
- **Precision@1 = 1.0**: Az első chunk releváns
- **Precision@3 = 0.33**: A releváns chunk a top-3-ban van (1 releváns / 3 lekért)
- **Precision@5 = 0.20**: A releváns chunk a top-5-ben van (1 releváns / 5 lekért)
- **Precision@10 = 0.10**: A releváns chunk a top-10-ben van (1 releváns / 10 lekért)

**Aggregált érték**: Átlagos Precision@K az összes kérdésre (0-1 tartomány)

#### Recall@K
Az összes releváns chunk közül hány van a top-K-ban (hányad rész)?

**Formula**: `Recall@K = (releváns ÉS lekért top-K) / (összes releváns)`

**Értelmezés single-relevance esetén** (1 releváns chunk):
- **Recall@1 = 1.0**: Az első chunk a releváns chunk (megtaláltuk)
- **Recall@3 = 1.0**: A releváns chunk a top-3-ban van (megtaláltuk)
- **Recall@5 = 1.0**: A releváns chunk a top-5-ben van (megtaláltuk)
- **Recall@10 = 0.0**: A releváns chunk nincs a top-10-ben (nem találtuk meg)

**Single-relevance esetén**: Recall@K megegyezik a Hit Rate@K-val (bináris 0 vagy 1 érték)

**Aggregált érték**: Átlagos Recall@K az összes kérdésre (0-1 tartomány)

---

### Backward Compatibility

A régi metrika nevek (`top_k_recall`, `top_1_precision`) továbbra is elérhetők az API-ban aliasként, így a meglévő kód változtatás nélkül működik.

**Régi nevek → Új nevek mapping:**
- `top_k_recall` → `hit_rate_at_k`
- `top_1_precision` → `first_position_accuracy`

---

### Embedding Quality Metrics (ÚJ)

Az **Embedding Quality Metrics** azt méri, hogy mennyire jól különíti el az embedding model a releváns chunk-okat az irreleváns chunk-októl a **similarity score** alapján.

Ez **eltér** a retrieval metrics-től (Hit Rate, MRR):
- **Retrieval metrics**: A teljes RAG pipeline teljesítményét mérik (embedding + chunking + reranking)
- **Embedding metrics**: Kizárólag az embedding model minőségét mérik

#### Separation Margin

**Formula**: `Separation Margin = Relevant Mean - Irrelevant Mean`

**Értelmezés**:
- **> 0.3**: Kiváló elválasztás - Az embedding model nagyon jól különíti el a releváns és irreleváns chunk-okat
- **> 0.2**: Jó elválasztás - Jól működő embedding
- **> 0.1**: Közepes elválasztás - Elfogadható, de fejleszthető
- **< 0.1**: Gyenge elválasztás - Az embedding model nem különíti el jól a chunk-okat

**Példa**:
- Relevant Similarity (mean): 0.75
- Irrelevant Similarity (mean): 0.45
- **Separation Margin**: 0.30 → **Kiváló!**

#### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

Az embedding model képessége a releváns és irreleváns chunk-ok binary classification-jére similarity score alapján.

**Értelmezés**:
- **1.0**: Tökéletes elválasztás (minden releváns > minden irreleváns)
- **0.9-1.0**: Kiváló embedding model
- **0.8-0.9**: Jó embedding model
- **0.7-0.8**: Közepes embedding model
- **0.5-0.7**: Gyenge embedding model (fejlesztendő!)
- **0.5**: Random guess (az embedding nem működik)
- **< 0.5**: Rosszabb mint random (valami hiba van)

**Példa**:
- ROC-AUC: 0.85 → **Jó embedding model**
- ROC-AUC: 0.63 → **Gyenge, fejlesztendő**

#### Standard Deviation (Konzisztencia)

Az embedding model konzisztenciáját méri - mennyire változó a similarity score ugyanazon típusú chunk-okra.

**Ideális eset**:
- Alacsony std dev mindkét kategóriában (konzisztens scoring)
- Releváns chunk-ok: std dev < 0.10
- Irreleváns chunk-ok: std dev < 0.10

**Probléma jelek**:
- Magas std dev (> 0.15) → Inkonzisztens embedding model
- Átfedés a két distribution között → Gyenge elválasztás

#### Sample Sizes

- **Relevant count**: Hány releváns chunk similarity score-t mértünk
- **Irrelevant count**: Hány irreleváns chunk similarity score-t mértünk

**Single-turn evaluation esetén**:
- Relevant count = Found count (hány query-nél találtuk meg az eredeti chunk-ot)
- Irrelevant count ≈ (K - 1) × Found count (ahol K = top-K méret, pl. 5)

#### Vizualizáció

Az **Embedding Quality Distribution** plot (histogram):
- **Kék oszlopok**: Releváns chunk-ok similarity distribution
- **Lila oszlopok**: Irreleváns chunk-ok similarity distribution
- **Függőleges vonalak**: Átlagok (mean similarity)
- **Sárga annotation**: Separation margin érték

**Ideális kép**: Két jól elkülönült distribution, minimális átfedéssel.

**Probléma jelek**: Nagy átfedés, hasonló mean értékek, kis separation margin.

---

## Tipikus Eredmények

Egy jó RAG rendszer várható teljesítménye:

### Binary Single-Relevance Metrics

- **Hit Rate@K (K=5)**: 80-95% (a legtöbb esetben az eredeti chunk a top-5-ben van)
- **First Position Accuracy**: 60-80% (sok esetben az eredeti chunk a legjobb match)
- **Average Rank**: 1.5-2.5 (ha megtalálható, általában az első vagy második helyen)
- **Average Similarity**: 0.75-0.90 (magas hasonlóság)

### Classical IR Metrics

- **MRR**: 0.65-0.85 (az első releváns chunk átlagosan az 1.2-1.5. pozícióban van)
- **Avg Precision@1**: 0.60-0.80 (megegyezik a First Position Accuracy-vel)
- **Avg Precision@3**: 0.27-0.32 (1 releváns / 3 lekért ≈ 0.33, ha Hit Rate@3 ≈ 85-95%)
- **Avg Precision@5**: 0.16-0.19 (1 releváns / 5 lekért = 0.20, ha Hit Rate@5 ≈ 80-95%)
- **Avg Precision@10**: 0.08-0.095 (1 releváns / 10 lekért = 0.10, ha Hit Rate@10 ≈ 80-95%)
- **Avg Recall@K**: Megegyezik a Hit Rate@K-val single-relevance esetén (0.80-0.95 K=5-nél)

**Megjegyzés**: Single-relevance forgatókönyv esetén (1 releváns chunk per query) a Precision@K és Recall@K értékek függ attól, hogy a releváns chunk benne van-e a top-K-ban. Ha igen, Recall@K = 1.0, ha nem, Recall@K = 0.0.

### Embedding Quality Metrics (ÚJ)

Egy jó embedding model várható teljesítménye:

- **Separation Margin**: 0.20-0.35 (jó-kiváló elválasztás)
- **ROC-AUC**: 0.80-0.95 (jó-kiváló embedding model)
- **Relevant Similarity (mean)**: 0.70-0.90 (magas hasonlóság a releváns chunk-okhoz)
- **Irrelevant Similarity (mean)**: 0.30-0.50 (alacsony hasonlóság az irreleváns chunk-okhoz)
- **Relevant Similarity (std)**: < 0.10 (konzisztens scoring)
- **Irrelevant Similarity (std)**: < 0.12 (konzisztens scoring)

**Valós példa (jelenlegi rendszer):**
```
Separation Margin: 0.0499  ← GYENGE! (< 0.1)
ROC-AUC: 0.6630            ← KÖZEPES (0.5-0.7 tartomány)
Relevant Mean: 0.5716
Irrelevant Mean: 0.5217    ← Túl közel a relevant-hez!
```

**Következtetés**: Az embedding model **nem különíti el jól** a releváns és irreleváns chunk-okat. Érdemes kipróbálni:
- Másik embedding modelt (pl. `text-embedding-3-large`)
- Chunk size optimalizálást
- Reranking algorithm finomhangolását

---

## Chunk Quality Metrics (ÚJ)

A chunk quality metrikák a **chunking stratégia minőségét** és **chunk-ok konzisztenciáját** mérik. Három fő metrika csoport:

### 1. Chunk Size Consistency Index (CSCI)

**Mit mér**: Mennyire konzisztensek a chunk méretek egymáshoz képest.

**Formula**:
```
CSCI = 1 - (std_dev_chunk_size / mean_chunk_size)
```

**Értelmezés**:
- **CSCI ≥ 0.8**: Nagyon konzisztens chunk méretek (kiváló)
- **0.6 ≤ CSCI < 0.8**: Jó konzisztencia
- **0.4 ≤ CSCI < 0.6**: Közepes konzisztencia
- **CSCI < 0.4**: Nagy variancia (gyenge konzisztencia)

**Példa kimenet**:
```
Chunk Size Consistency Index (CSCI): 0.7421
  Mean chunk size: 867.9 tokens
  Std deviation: 223.8 tokens
  Range: 14-1024 tokens
```

**Jelentése**: CSCI = 0.7421 **jó konzisztencia**, de van némi variancia (14-1024 token range). Fixed-size chunking esetén ez az érték általában > 0.9 lenne.

---

### 2. Retrieval Quality per Chunk Size Bucket (RQCSB)

**Mit mér**: Hogyan teljesítenek a különböző méretű chunk-ok a retrieval során.

**Bucket definíciók**:
- **Tiny**: 0-100 tokens
- **Small**: 100-300 tokens
- **Medium**: 300-600 tokens
- **Large**: 600-1000 tokens
- **XLarge**: 1000+ tokens

**Metrikák bucket-enként**:
- **Hit Rate**: Hány %-ban találta meg a chunk-ot
- **Avg Rank**: Átlagos pozíció a ranking-ben
- **Avg Similarity**: Átlagos cosine similarity
- **First Position Accuracy**: Hány %-ban volt első helyen

**Példa kimenet**:
```
Retrieval Performance by Chunk Size:
  Tiny     ( 1 chunks): Hit Rate=100.00%, Avg Rank=1.00
  Small    ( 3 chunks): Hit Rate=100.00%, Avg Rank=1.00
  Medium   ( 7 chunks): Hit Rate=100.00%, Avg Rank=1.43
  Large    (46 chunks): Hit Rate=91.30%, Avg Rank=1.38
  XLarge   (21 chunks): Hit Rate=100.00%, Avg Rank=1.86
```

**Insights**:
- **Large bucket (600-1000 tokens)**: 46 chunk, de **csak 91.30% hit rate** → Ez a bucket gyengébb
- **XLarge bucket (1000+ tokens)**: 100% hit rate, de **magasabb avg rank (1.86)** → Megtalálja, de nem első helyen
- **Small/Medium buckets**: Legjobb retrieval (100% hit rate, alacsony rank)

**Következtetés**: A **túl nagy chunk-ok (XLarge)** lassabban kerülnek előre, de megbízhatóbbak mint a large bucket. Optimális chunk size: **300-600 tokens (Medium)** vagy **600-1000 (Large)**.

**Vizualizáció**: `rqcsb_heatmap.png` - Heatmap a bucket-enkénti teljesítményről.

---

### 3. Position Stability Score (PSS)

**Mit mér**: Mennyire "stabil" egy chunk pozíciója különböző query-k esetén.

**Koncepció**:
- Ha egy chunk mindig ugyanazon a ranken jelenik meg → **stabil** (magas PSS)
- Ha a chunk pozíciója nagyon változó → **instabil** (alacsony PSS)

**Formula**:
```
PSS[chunk_id] = 1 - (std_dev_rank / mean_rank)
```

**Értelmezés**:
- **PSS ≥ 0.8**: Nagyon stabil chunk (megbízható retrieval)
- **0.6 ≤ PSS < 0.8**: Stabil chunk
- **0.4 ≤ PSS < 0.6**: Közepes stabilitás
- **PSS < 0.4**: Instabil chunk (változó ranking)

**Példa kimenet**:
```
Position Stability (PSS): 0.5838 (±0.1346)
  Chunks analyzed: 66
  Top 3 Stable Chunks:
    1. Chunk b73aea1e-443d-42b1-b526-95df7f1fb0e6: PSS=0.8981 (4 appearances)
    2. Chunk 4d453b48-0230-46d8-992c-4027535069b1: PSS=0.8668 (4 appearances)
    3. Chunk 2b8a3316-e217-4c66-9e23-eafb0c3f8e8d: PSS=0.8571 (2 appearances)
```

**Jelentése**:
- **Overall PSS = 0.5838**: Közepes stabilitás a chunk-ok között
- **Top stable chunks**: 0.89-0.86 PSS (nagyon stabilak)
- **66 chunk elemezve**: Csak azok a chunk-ok, amelyek legalább 2x megjelentek a retrieval során

**Miért fontos?**
- Alacsony PSS → Chunk tartalma nem konzisztens VAGY embedding quality gyenge
- Magas PSS → Chunk jól definiált, konzisztens retrieval

---

### Chunk Quality Metrikák - Vizualizációk

**1. Chunk Size Distribution (`chunk_size_distribution.png`)**
- Histogram: Token count eloszlás
- Bucket határok kijelölve (100, 300, 600, 1000)
- Mean és median vonalak
- Stats doboz: Std, Min, Max

**2. RQCSB Heatmap (`rqcsb_heatmap.png`)**
- Sorok: Chunk size buckets (Tiny, Small, Medium, Large, XLarge)
- Oszlopok: Metrikák (Hit Rate, Avg Rank, Avg Similarity, First Pos Acc)
- Színkódolás: Sötétebb = jobb teljesítmény
- Cellákban: Eredeti értékek

---

### Tipikus Eredmények és Értelmezés

**Jó chunking stratégia várható teljesítménye:**

| Metrika | Várt Érték | Jelenlegi | Státusz |
|---------|------------|-----------|---------|
| **CSCI** | > 0.8 (fixed), > 0.6 (semantic) | 0.7421 | ✅ JÓ |
| **Hit Rate (overall)** | > 90% | 94.87% | ✅ KIVÁLÓ |
| **Hit Rate (Large bucket)** | > 95% | 91.30% | ⚠️ JAVÍTANDÓ |
| **Avg Rank (XLarge)** | < 1.5 | 1.86 | ⚠️ MAGAS |
| **Position Stability** | > 0.7 | 0.5838 | ⚠️ KÖZEPES |

**Actionable Insights**:

1. **Large bucket (600-1000 tokens) gyenge**:
   - **Probléma**: 91.30% hit rate (alacsonyabb mint többi bucket)
   - **Megoldás**: Chunk size threshold csökkentése 800-ra vagy semantic boundary finomhangolás

2. **XLarge chunk-ok lassúak**:
   - **Probléma**: Avg rank = 1.86 (nem első helyen)
   - **Megoldás**: Chunk max size korlátozása 1000 token-re

3. **Közepes Position Stability**:
   - **Probléma**: PSS = 0.5838 (chunk-ok pozíciója változó)
   - **Megoldás**: Embedding quality javítása (lásd: Embedding Quality Metrics) vagy chunk tartalmi konzisztencia növelése

---

### Multi-Strategy Comparison (Jövőbeli Funkció)

⚠️ **Jelenleg csak semantic stratégia van az adatokban** → Stratégia-összehasonlítás nem lehetséges.

**Következő lépések multi-strategy értékeléshez**:

1. **Chunking pipeline futtatása mind a 4 stratégiával**:
   ```bash
   python chunking/main.py --strategy fixed_size
   python chunking/main.py --strategy semantic
   python chunking/main.py --strategy recursive
   python chunking/main.py --strategy document_specific
   ```

2. **Evaluation futtatása mindegyikre**:
   ```bash
   python generate_questions.py --strategy fixed_size
   python evaluate_rag.py --strategy fixed_size
   # ... (repeat for other strategies)
   ```

3. **analyze_results.py automatikusan generál**:
   - `metrics_by_strategy.csv` - Stratégiánként aggregált metrikák
   - Strategy comparison plot-ok

**Várható eredmények**:
- **Fixed-size**: Magas CSCI (> 0.9), de alacsonyabb semantic quality
- **Semantic**: Jobb retrieval, de változó chunk size
- **Recursive**: Köztes megoldás
- **Document-specific**: Dokumentum típustól függő teljesítmény

---

## Hibakeresés

### "No chunks found in database"
- Ellenőrizd, hogy a chunking pipeline lefutott-e
- Ellenőrizd az adatbázis kapcsolatot

### "Failed to connect to database"
- Ellenőrizd, hogy a PostgreSQL fut-e (`docker-compose up -d`)
- Ellenőrizd a `.env` fájlban a database credentials-t

### "OpenAI API error"
- Ellenőrizd az `OPENAI_API_KEY` környezeti változót
- Ellenőrizd az API quota-t és rate limit-et

### Lassú kérdésgenerálás
- Normális: 1-2 másodperc / chunk (OpenAI API hívás miatt)
- 100 chunk esetén ~3-5 perc
- 1000 chunk esetén ~30-50 perc

## Testreszabás

### Kérdésgenerálási prompt módosítása

Szerkeszd a `config.py` fájlban a `QUESTION_GENERATION_PROMPT` változót:

```python
QUESTION_GENERATION_PROMPT = """
Your custom prompt here...
{chunk_content}
"""
```

### Top-K érték módosítása

Szerkeszd a `.env` fájlt:

```bash
DEFAULT_MATCH_COUNT=10  # 5 helyett 10
```

### Similarity threshold módosítása

```bash
DEFAULT_MATCH_THRESHOLD=0.5  # 0.3 helyett 0.5
```

## Kiegészítések és Továbbfejlesztési Lehetőségek

- **Multi-turn evaluation**: Több körös beszélgetések értékelése
- **Answer quality metrics**: A generált válaszok minőségének értékelése (BLEU, ROUGE, semantic similarity)
- **Human evaluation**: Emberi értékelők által validált ground truth
- **Chunk size experiments**: Különböző chunk méretek összehasonlítása
- **Embedding model comparison**: Különböző embedding modellek tesztelése
- **Query augmentation**: Kérdés parafrázisok generálása és tesztelése

## Licensz

Ez a projekt a fő RAG Assistant projekt része.

## Kapcsolat

Kérdések esetén nézd meg a fő projekt dokumentációját vagy nyiss issue-t a GitHub-on.
