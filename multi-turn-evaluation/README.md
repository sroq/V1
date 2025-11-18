# Multi-Turn Conversation Evaluation

Átfogó evaluation rendszer RAG-alapú AI asszisztensek **multi-turn beszélgetéseinek** értékelésére.

## Áttekintés

Ez a rendszer **end-to-end szimulációt** végez, ahol különböző felhasználói personák különböző célokat próbálnak elérni a RAG asszisztenssel folytatott beszélgetés során. Az értékelés **LLM as a Judge** módszertant használ (GPT-4o mini) több dimenzióban.

### Fő komponensek

1. **Personas** (`personas.py`): Felhasználói típusok definíciói
2. **Goals** (`goals.py`): Beszélgetési célok definíciói
3. **User Simulator** (`user_simulator.py`): Felhasználói viselkedés szimulálása LLM-mel
4. **Assistant Client** (`assistant_client.py`): Backend API kommunikáció
5. **Evaluator** (`evaluator.py`): Multi-dimenziós LLM-alapú értékelés
6. **Orchestrator** (`run_multi_turn_evaluation.py`): Fő pipeline koordinátor

## Értékelési dimenziók

Az értékelés **5 dimenzióban** történik (LLM as a Judge):

| Dimenzió | Súly | Leírás |
|----------|------|---------|
| **Goal Achievement** | 40% | Elérte-e a felhasználó a célját? |
| **Conversation Quality** | 20% | Koherencia, természetesség, kontextus megértés |
| **Response Relevance** | 20% | Válaszok relevanciája, pontossága, teljessége |
| **User Experience** | 10% | Frusztráció, megfelelő stílus, élmény minősége |
| **Efficiency** | 10% | Optimális turns szám, felesleges ismétlések |

**Overall Score**: Súlyozott átlag (0-100 skála)

## Felhasználói personák

5 különböző persona különböző jellemzőkkel:

| Persona | Türelem | Szakértelem | Frusztráció küszöb | Feladás |
|---------|---------|-------------|-------------------|---------|
| **Türelmetlen Kezdő** | 3/10 | novice | 2 rossz válasz | 4 kör |
| **Türelmes Haladó** | 8/10 | intermediate | 5 rossz válasz | 8 kör |
| **Szakértő Kutató** | 10/10 | expert | 8 rossz válasz | 12 kör |
| **Kíváncsi Böngésző** | 6/10 | novice | 4 rossz válasz | 6 kör |
| **Célratörő Tanuló** | 7/10 | intermediate | 6 rossz válasz | 10 kör |

### Persona jellemzők

- **Türelem**: Mennyire türelmes (1-10)
- **Szakértelem**: novice, intermediate, expert
- **Kommunikációs stílus**: verbosity, formality, question clarity
- **Viselkedési minták**: frusztráció kezelés, feladási pont, érzelmek

## Beszélgetési célok

7 különböző cél 3 nehézségi szinten (The Jungle Book alapú):

### Easy (2 cél)

- **Maugli személyazonosságának tisztázása** (2 kör)
- **Balú személyazonosságának tisztázása** (2 kör)

### Medium (3 cél)

- **Ser Kán konfliktusának megértése** (4 kör)
- **A dzsungel törvényének megértése** (3 kör)
- **Dzsungel felfedezése** (4 kör, exploratory)

### Hard (2 cél)

- **Maugli útjának teljes megértése** (6 kör)
- **Karakterek összehasonlítása** (5 kör, Balú vs Bagíra)

### Goal struktúra

Minden goal tartalmazza:
- **Kezdő kérdés**: Első user query
- **Sikerkritériumok**: Lista a teljesítendő feltételekről
- **Mérföldkövek**: Várható haladás turn-ről turn-re
- **Várható turns**: Optimális beszélgetés hossz

## Telepítés

### Előfeltételek

```bash
# Python 3.9+
# A backend asszisztens futnia kell: http://localhost:3000
```

### Függőségek

```bash
cd multi-turn-evaluation
pip install -r requirements.txt
```

A `requirements.txt` tartalma:

```
openai>=1.0.0
requests>=2.31.0
python-dotenv>=1.0.0
```

### Környezeti változók

Hozz létre egy `.env` fájlt a root könyvtárban (ha még nincs):

```bash
OPENAI_API_KEY=your_openai_api_key_here
ASSISTANT_BASE_URL=http://localhost:3000  # Opcionális, default: localhost:3000
```

## Használat

### 1. Backend indítása

**FONTOS**: Az evaluation előtt indítsd el a RAG asszisztens backendet:

```bash
cd assistant
npm run dev
```

Várj, amíg fut: `http://localhost:3000`

### 2. Single Conversation (egy beszélgetés)

Egy persona-goal pár tesztelése:

```bash
cd multi-turn-evaluation
python3 run_multi_turn_evaluation.py --persona patient_intermediate --goal mowgli_identity
```

**Kimenet**:
- Részletes beszélgetés log
- Turn-by-turn user-assistant interakció
- LLM-based evaluation minden dimenzióban
- JSON mentés: `results/patient_intermediate_mowgli_identity_TIMESTAMP.json`

### 3. Batch Evaluation (minden kombináció)

Összes persona × goal kombináció futtatása:

```bash
python3 run_multi_turn_evaluation.py --batch
```

**Kimenet**:
- 35 beszélgetés (5 persona × 7 goal)
- Egyedi JSON fájl minden beszélgetéshez
- Batch összefoglaló: `results/run_TIMESTAMP/batch_summary.json`
- Vizualizációk: `results/run_TIMESTAMP/plots/`

### 4. Specifikus batch (szűrt)

Csak bizonyos personák vagy goalok:

```bash
# Csak egy persona, összes goal
python3 run_multi_turn_evaluation.py --batch --persona patient_intermediate

# Csak egy goal, összes persona
python3 run_multi_turn_evaluation.py --batch --goal mowgli_journey

# Egy persona, egy goal (ugyanaz, mint single mode)
python3 run_multi_turn_evaluation.py --persona expert_researcher --goal characters_comparison
```

### 5. Quiet mode (minimális log)

```bash
python3 run_multi_turn_evaluation.py --batch --quiet
```

## Eredmények

### Könyvtárstruktúra

Minden futtatás (single vagy batch) egy saját timestamped könyvtárba menti az eredményeket:

```
results/
├── run_20250115_143022/
│   ├── patient_intermediate_mowgli_identity_20250115_143022.json
│   ├── impatient_novice_baloo_role_20250115_143025.json
│   ├── ...
│   ├── batch_summary.json
│   └── plots/
│       ├── 01_overall_scores.png
│       ├── 02_goal_achievement_by_persona.png
│       ├── ...
│       └── 14_performance_heatmap.png
├── run_20250115_150833/
│   ├── expert_researcher_comparison_20250115_150833.json
│   └── batch_summary.json
└── run_20250115_162204/
    └── ...
```

**Előnyök**:
- ✅ Tiszta szervezés - minden futtatás elkülönítve
- ✅ Könnyű összehasonlítás - timestamp alapján rendezve
- ✅ Nincs fájl-ömlesztés a results könyvtárban
- ✅ Vizualizációk közvetlenül a futtatás mellett

### Egyedi beszélgetés JSON

Minden beszélgetésről mentődik egy JSON fájl a `results/run_TIMESTAMP/` könyvtárban:

```json
{
  "success": true,
  "persona": "patient_intermediate",
  "goal": "mowgli_identity",
  "conversation": [
    { "role": "user", "content": "Ki az a Maugli?" },
    { "role": "assistant", "content": "Maugli..." },
    ...
  ],
  "evaluation": {
    "goal_achievement": {
      "goal_reached": true,
      "achievement_score": 85,
      "milestones_reached": 2,
      "reasoning": "..."
    },
    "conversation_quality": {
      "overall_quality_score": 90,
      "coherence_score": 9,
      "naturalness_score": 8,
      "strengths": [...],
      "weaknesses": [...]
    },
    "response_relevance": {
      "overall_relevance_score": 88,
      "average_relevance": 8.5,
      "responses": [...]
    },
    "user_experience": {
      "ux_score": 92,
      "frustration_detected": false,
      "overall_experience": "positive"
    },
    "efficiency": {
      "efficiency_score": 95,
      "turn_efficiency": 100,
      "redundancy_detected": false
    },
    "overall_score": 88.3
  },
  "metadata": {
    "turns": 2,
    "duration_s": 8.5,
    "user_progress": {
      "milestones_reached": 2,
      "progress_percentage": 100,
      "goal_reached": true,
      "frustration_level": 0
    }
  }
}
```

### Batch összefoglaló JSON

```json
{
  "timestamp": "2025-11-15T...",
  "total_conversations": 35,
  "successful": 34,
  "failed": 1,
  "metrics": {
    "average_overall_score": 82.5,
    "average_turns": 4.2,
    "average_duration_s": 18.7,
    "goal_achievement_rate": 88.2
  },
  "results": [...]
}
```

### Vizualizációk

Batch evaluation után **automatikusan generálódnak** grafikonok a `results/run_TIMESTAMP/plots/` könyvtárban.

**14 grafikon** készül:

1. **Overall Scores** - 5 dimenzió átlaga bar chart
2. **Goal Achievement by Persona** - Persona-nkénti sikeres cél elérés
3. **Goal Achievement by Difficulty** - Nehézség szerint (easy/medium/hard)
4. **Turn Count Distribution** - Körök számának eloszlása
5. **Duration Distribution** - Beszélgetések időtartama
6. **Score Distribution by Dimension** - Box plot minden dimenzióra
7. **Success Rate Heatmap** - Persona × Goal mátrix
8. **User Experience Breakdown** - UX kategóriák (positive/neutral/negative)
9. **Efficiency vs Quality** - Scatter plot (hatékonyság vs minőség)
10. **Dimension Correlation** - Dimenziók közötti korreláció
11. **API Latency Distribution** - API válaszidő eloszlása (performance)
12. **TTFT Distribution** - Time To First Token eloszlása (performance)
13. **Tokens/sec Distribution** - Token generálási sebesség eloszlása (performance)
14. **Performance Metrics Heatmap** - Persona × Goal teljesítmény metrikák (performance)

#### Manuális vizualizáció generálás

Batch summary alapján egyedileg is generálhatod:

```bash
python3 visualize_results.py results/run_TIMESTAMP/batch_summary.json
```

Vagy egyedi output könyvtárba:

```bash
python3 visualize_results.py results/run_TIMESTAMP/batch_summary.json --output-dir custom_plots/
```

#### Példa kimenet

```
results/
└── run_20251115_143022/
    ├── patient_intermediate_mowgli_identity_20251115_143025.json
    ├── impatient_novice_baloo_role_20251115_143027.json
    ├── expert_researcher_comparison_20251115_143030.json
    ├── ...
    ├── batch_summary.json
    └── plots/
        ├── 01_overall_scores.png
        ├── 02_goal_achievement_by_persona.png
        ├── 03_goal_achievement_by_difficulty.png
        ├── 04_turn_distribution.png
        ├── 05_duration_distribution.png
        ├── 06_score_distributions.png
        ├── 07_success_heatmap.png
        ├── 08_ux_breakdown.png
        ├── 09_efficiency_vs_quality.png
        ├── 10_dimension_correlation.png
        ├── 11_api_latency_distribution.png
        ├── 12_ttft_distribution.png
        ├── 13_tokens_per_second_distribution.png
        └── 14_performance_heatmap.png
```

#### Grafikonok részletes magyarázata

##### 1. Overall Scores (`01_overall_scores.png`)

**Mit mutat**: 5 értékelési dimenzió átlagos pontszámai oszlopdiagramon (0-100 skála).

**Hogyan olvasd**:
- Minden oszlop egy dimenziót reprezentál
- Az oszlop magassága a pontszám (0-100)
- Az oszlop tetején látható a pontos érték
- Piros szaggatott vonal: 50%-os küszöb (ez alatt gyenge)

**Színek**: Seaborn "husl" paletta (5 különböző szín a dimenziókhoz)

**Mit keress**:
- **> 70**: Kiváló teljesítmény (zöld tartomány)
- **50-70**: Jó teljesítmény (sárga tartomány)
- **< 50**: Gyenge teljesítmény (piros tartomány)
- Melyik dimenzió a leggyengébb? → Arra kell fejleszteni

**Példa értelmezés**: Ha a "Response Relevance" 45, akkor a válaszok relevanciája gyenge → RAG pipeline javítása szükséges.

---

##### 2. Goal Achievement by Persona (`02_goal_achievement_by_persona.png`)

**Mit mutat**: Melyik persona éri el leggyakrabban a céljait? (%-os arány)

**Hogyan olvasd**:
- X tengely: Persona nevek (5 persona)
- Y tengely: Goal achievement rate (0-100%)
- Oszlopok magassága: hány százalékban érte el a célt
- Az oszlop tetején: pontos százalék érték

**Színek**: "Set2" paletta (5 különböző szín personánként)

**Mit keress**:
- **> 80%**: Nagyon sikeres persona
- **< 50%**: Nehezen boldogul
- Melyik persona a legsikeresebb? → Az asszisztens neki optimalizált
- Melyik persona a legkevésbé sikeres? → Fejlesztési terület

**Példa értelmezés**: Ha "Türelmetlen Kezdő" 30%, "Szakértő Kutató" 90% → Az asszisztens szakértőknek jó, kezdőknek nem.

---

##### 3. Goal Achievement by Difficulty (`03_goal_achievement_by_difficulty.png`)

**Mit mutat**: 2 grafikon egymás mellett:
1. **Achievement rate** nehézség szerint
2. **Átlagos turns** nehézség szerint

**Hogyan olvasd**:
- **Bal oldali**: Easy/Medium/Hard célok elérési aránya
- **Jobb oldali**: Átlagos körök száma nehézség szerint
- Színek: Zöld (easy) → Narancs (medium) → Piros (hard)

**Mit keress**:
- **Bal grafikon**: Csökken-e az arány a nehézséggel? (elvárható)
- **Jobb grafikon**: Nő-e a turns szám a nehézséggel? (elvárható)
- Ha hard goal achievement > 80% → Kiváló!
- Ha easy goal achievement < 60% → Probléma!

**Példa értelmezés**: Easy: 95%, Medium: 70%, Hard: 40% → Normális eloszlás, az asszisztens jól skálázódik.

---

##### 4. Turn Count Distribution (`04_turn_distribution.png`)

**Mit mutat**: Hány körben érnek célba a beszélgetések? (histogram)

**Hogyan olvasd**:
- X tengely: Turns száma
- Y tengely: Előfordulások száma (frequency)
- Piros szaggatott vonal: Átlag
- Zöld szaggatott vonal: Medián
- Oszlopok: Adott turns számú beszélgetések mennyisége

**Színek**: Kék oszlopok, piros/zöld vonalak

**Mit keress**:
- **Csúcs 2-3 körben**: Hatékony asszisztens
- **Csúcs 7+ körben**: Lassú, túl sok kérdés-válasz kör
- **Nagy szórás**: Inkonzisztens teljesítmény

**Példa értelmezés**: Ha a csúcs 2 körben van, de vannak 10+ körös beszélgetések is → Van optimalizálási lehetőség.

---

##### 5. Duration Distribution (`05_duration_distribution.png`)

**Mit mutat**: Mennyi időbe telik egy beszélgetés? (histogram, másodpercben)

**Hogyan olvasd**:
- X tengely: Duration (másodperc)
- Y tengely: Előfordulások száma
- Piros vonal: Átlag
- Zöld vonal: Medián
- Oszlopok: Időtartam eloszlás

**Színek**: Korall (narancs-piros) oszlopok

**Mit keress**:
- **< 10s**: Nagyon gyors (akár túl gyors?)
- **10-30s**: Optimális
- **> 60s**: Lassú, problémás
- Átlag vs medián különbség: Ha nagy, akkor van néhány nagyon lassú beszélgetés

**Példa értelmezés**: Átlag 25s, medián 20s → Néhány outlier lassú beszélgetés húzza fel az átlagot.

---

##### 6. Score Distribution by Dimension (`06_score_distributions.png`)

**Mit mutat**: Box plot minden dimenzióra → Eloszlások összehasonlítása

**Hogyan olvasd a box plot-ot**:
- **Box alsó széle**: 25. percentilis (Q1)
- **Box közepén piros vonal**: Medián (50. percentilis)
- **Box felső széle**: 75. percentilis (Q3)
- **Bajusz vonalak**: Min/max (vagy 1.5×IQR)
- **Pontok**: Outlier értékek

**Színek**: Világoskék box-ok, piros medián vonal

**Mit keress**:
- **Magas medián + keskeny box**: Konzisztens jó teljesítmény ✅
- **Alacsony medián + széles box**: Inkonzisztens, gyenge teljesítmény ❌
- **Sok outlier**: Néhány beszélgetés nagyon eltér az átlagtól
- **Melyik box a legalacsonyabb**: Az a dimenzió a leggyengébb

**Példa értelmezés**: Ha "Goal Achievement" box mediánja 85, de "User Experience" box mediánja 50 → Célokat eléri, de rossz élmény.

---

##### 7. Success Rate Heatmap (`07_success_heatmap.png`)

**Mit mutat**: Persona × Goal mátrix → Melyik persona melyik goal-nál teljesít jól?

**Hogyan olvasd**:
- **Sorok**: Personák (5 sor)
- **Oszlopok**: Goal-ok (7 oszlop)
- **Cellák**: Overall score (0-100)
- **Cellaérték**: Pontos score szám (fehér szöveg)

**Színkódolás**:
- **Zöld**: Magas score (> 80)
- **Sárga**: Közepes score (50-80)
- **Piros**: Alacsony score (< 50)

**Mit keress**:
- **Legsötétebb zöld cellák**: Legjobb kombinációk
- **Piros cellák**: Problémás kombinációk
- **Függőleges minták**: Van olyan goal, ami mindenkinek nehéz?
- **Vízszintes minták**: Van olyan persona, aki mindenhol gyengén teljesít?

**Példa értelmezés**: Ha "Türelmetlen Kezdő" × "Karakterek összehasonlítása" piros (25) → Ez a kombináció nem működik.

---

##### 8. User Experience Breakdown (`08_ux_breakdown.png`)

**Mit mutat**: Kördiagram (pie chart) → UX kategóriák megoszlása

**Hogyan olvasd**:
- **Szeletek**: Positive / Neutral / Negative
- **Szelet mérete**: Előfordulási arány
- **Százalék**: Minden szelet mellett

**Színkódolás**:
- **Zöld**: Positive (jó élmény)
- **Szürke**: Neutral (semleges)
- **Piros**: Negative (rossz élmény)

**Mit keress**:
- **> 60% positive**: Kiváló UX ✅
- **> 30% negative**: Problémás UX ❌
- **Sok neutral**: Nem eléggé engaging

**Példa értelmezés**: 70% positive, 20% neutral, 10% negative → Jó UX, de van hova fejlődni.

---

##### 9. Efficiency vs Quality (`09_efficiency_vs_quality.png`)

**Mit mutat**: Scatter plot → Van-e trade-off a hatékonyság és minőség között?

**Hogyan olvasd**:
- **X tengely**: Efficiency score (0-100)
- **Y tengely**: Overall score (0-100)
- **Pontok**: Beszélgetések
- **Pont színe**: Turn count (színskála)
- **Piros szaggatott vonal**: Trendline (korreláció)

**Színskála** (viridis):
- **Sötétkék**: Kevés turn (2-3)
- **Zöld**: Közepes turn (4-6)
- **Sárga**: Sok turn (7+)

**Mit keress**:
- **Pozitív korreláció** (trendline felfelé): Hatékonyság → Jobb minőség ✅
- **Negatív korreláció** (trendline lefelé): Hatékonyság ↔ Minőség trade-off ❌
- **Jobb felső sarok**: Ideális pontok (hatékony ÉS jó minőség)
- **Bal alsó sarok**: Rossz pontok (lassú ÉS rossz minőség)

**Példa értelmezés**: Ha pontok a jobb felső sarokban → Hatékonyság nem csökkenti a minőséget.

---

##### 10. Dimension Correlation (`10_dimension_correlation.png`)

**Mit mutat**: Korrelációs mátrix → Melyik dimenziók mozognak együtt?

**Hogyan olvasd**:
- **6×6 mátrix**: 6 dimenzió (5 fő + overall)
- **Cellák**: Korrelációs együttható (-1 és +1 között)
- **Cellaérték**: Pontos korreláció (2 tizedes)

**Színkódolás**:
- **Piros**: Pozitív korreláció (+1 = tökéletes)
- **Kék**: Negatív korreláció (-1 = tökéletes fordított)
- **Fehér**: Nincs korreláció (0)

**Mit keress**:
- **Átló**: Mindig 1.0 (önmagával tökéletes korreláció)
- **Erős pozitív**: > 0.7 (együtt mozognak)
- **Gyenge**: < 0.3 (független dimenziók)
- **Negatív**: < 0 (fordított kapcsolat - ritka!)

**Példa értelmezés**:
- Ha "Goal Achievement" ↔ "Overall Score" = 0.9 → Erős kapcsolat (Goal Achievement dominál)
- Ha "Efficiency" ↔ "User Experience" = 0.2 → Független dimenziók

---

##### 11. API Latency Distribution (`11_api_latency_distribution.png`)

**Mit mutat**: API válaszidő eloszlása minden turn-ből (histogram)

**Hogyan olvasd**:
- **X tengely**: API latency milliszekundumban (ms)
- **Y tengely**: Gyakoriság (hány turn essen az adott latency tartományba)
- **Piros szaggatott vonal**: Mean (átlag)
- **Zöld szaggatott vonal**: Median (középérték)
- **Narancssárga szaggatott vonal**: P95 (95. percentilis)

**Színek**: Steelblue (acélkék) histogram oszlopok

**Mit keress**:
- **< 1000ms mean**: Gyors API válasz ✅
- **1000-2000ms mean**: Normál válasz (elfogadható)
- **> 2000ms mean**: Lassú API ❌
- **P95 > 3000ms**: Sok lassú request → Backend optimalizálás kell
- **Széles eloszlás**: Inkonzisztens teljesítmény → Terhelés ingadozás

**Példa értelmezés**: Mean 1200ms, Median 1100ms, P95 1800ms → Konzisztensen gyors, 95%-ban 1.8s alatt válaszol.

---

##### 12. TTFT Distribution (`12_ttft_distribution.png`)

**Mit mutat**: Time To First Token (TTFT) eloszlása minden turn-ből → Mennyi idő az első válasz token megjelenéséig?

**Hogyan olvasd**:
- **X tengely**: TTFT milliszekundumban (ms)
- **Y tengely**: Gyakoriság (hány turn)
- **Piros szaggatott vonal**: Mean (átlag)
- **Zöld szaggatott vonal**: Median (középérték)
- **Narancssárga szaggatott vonal**: P95 (95. percentilis)

**Színek**: Mediumseagreen (tengerzöld) histogram oszlopok

**Mit keress**:
- **< 500ms mean**: Kiváló user experience (azonnal reagál) ✅
- **500-1000ms mean**: Jó (gyors válasz)
- **> 1000ms mean**: Lassú (user érzi a várakozást) ❌
- **P95 > 1500ms**: 5%-ban lassú → Backend optimalizálás
- **Keskeny eloszlás**: Konzisztens streaming start

**Példa értelmezés**: Mean 650ms, Median 600ms, P95 900ms → Gyors streaming indulás, konzisztens.

---

##### 13. Tokens/sec Distribution (`13_tokens_per_second_distribution.png`)

**Mit mutat**: Token generálási sebesség eloszlása → Milyen gyorsan streamel a modell?

**Hogyan olvasd**:
- **X tengely**: Tokens per second (tok/s)
- **Y tengely**: Gyakoriság (hány turn)
- **Piros szaggatott vonal**: Mean (átlag)
- **Zöld szaggatott vonal**: Median (középérték)
- **Narancssárga szaggatott vonal**: P95 (95. percentilis)

**Színek**: Coral (korall) histogram oszlopok

**Mit keress**:
- **> 30 tok/s mean**: Gyors streaming ✅
- **20-30 tok/s mean**: Normál sebesség
- **< 20 tok/s mean**: Lassú streaming ❌
- **P95 < 15 tok/s**: 5%-ban nagyon lassú → Modell/backend probléma
- **Széles eloszlás**: Inkonzisztens throughput → Terhelés hatás

**Példa értelmezés**: Mean 28 tok/s, Median 27 tok/s, P95 32 tok/s → Stabil, gyors streaming.

---

##### 14. Performance Metrics Heatmap (`14_performance_heatmap.png`)

**Mit mutat**: 3 heatmap egymás mellett → Persona × Goal teljesítmény metrikák

**Hogyan olvasd**:
- **3 heatmap**: API Latency / TTFT / Tokens/sec
- **Sorok**: Personák (5 sor)
- **Oszlopok**: Goal-ok (7 oszlop)
- **Cellák**: Átlag érték az adott kombinációra
- **Cellaérték**: Pontos szám (ms vagy tok/s)

**Színkódolás**:
- **API Latency & TTFT**:
  - **Zöld**: Alacsony latency (gyors) → Jó ✅
  - **Sárga**: Közepes latency
  - **Piros**: Magas latency (lassú) → Rossz ❌
- **Tokens/sec**:
  - **Zöld**: Magas throughput (gyors) → Jó ✅
  - **Sárga**: Közepes throughput
  - **Piros**: Alacsony throughput (lassú) → Rossz ❌

**Mit keress**:
- **Függőleges piros oszlop**: Van olyan goal, ami mindig lassú? → Goal complexity probléma
- **Vízszintes piros sor**: Van olyan persona, akinek mindig lassú? → Persona query pattern probléma
- **Konzisztens zöld**: Stabil teljesítmény minden kombinációra ✅
- **Patchwork minta**: Inkonzisztens teljesítmény → Terhelés vagy cache hatás

**Példa értelmezés**: Ha "Karakterek összehasonlítása" goal-nál minden persona piros API latency → Ez a goal túl komplex, több RAG chunk-ot kér.

---

#### Összegzés: Hogyan használd a grafikonokat?

**Quick check** (5 perc):
1. **01_overall_scores.png** → Van-e 50 alatti dimenzió? → Fejlesztési terület
2. **07_success_heatmap.png** → Van-e piros cella? → Problémás kombináció
3. **08_ux_breakdown.png** → Negative > 20%? → UX probléma
4. **11_api_latency_distribution.png** → Mean > 2000ms? → Backend lassú ❌

**Mélyebb elemzés** (15 perc):
5. **02** + **03** → Melyik persona/difficulty okoz problémát?
6. **06_score_distributions.png** → Melyik dimenzió a leginkonzisztensebb?
7. **09_efficiency_vs_quality.png** → Van trade-off?
8. **10_dimension_correlation.png** → Melyik dimenzió a legfontosabb?
9. **12_ttft_distribution.png** + **13_tokens_per_second_distribution.png** → Streaming teljesítmény elemzés
10. **14_performance_heatmap.png** → Melyik goal/persona kombináció a leglassabb?

**Fejlesztési akciók**:
- Ha Response Relevance alacsony → RAG pipeline javítása
- Ha Goal Achievement alacsony → Goal definíciók egyszerűsítése
- Ha Efficiency alacsony → Túl sok turn → Context javítása
- Ha UX negative magas → Persona-goal párosítás felülvizsgálata
- Ha API Latency magas → Backend optimalizálás (cache, indexek)
- Ha TTFT magas → Embedding/reranking optimalizálás
- Ha Tokens/sec alacsony → Model throughput vagy network probléma

## Tesztelés

Minden modul önállóan tesztelhető:

```bash
# Personák listázása
python3 personas.py

# Goalok listázása
python3 goals.py

# User Simulator teszt
python3 user_simulator.py

# Assistant Client teszt (backend kell!)
python3 assistant_client.py

# Evaluator teszt
python3 evaluator.py
```

## Költségek

**OpenAI API használat per beszélgetés:**

- **User Simulator**: ~3-5 LLM call (query generálás)
  - Model: GPT-4o mini
  - Tokens: ~500-1000 / beszélgetés
  - Költség: ~$0.0002-0.0005

- **Evaluator**: 5 LLM call (5 dimenzió)
  - Model: GPT-4o mini
  - Tokens: ~3000-5000 / beszélgetés
  - Költség: ~$0.001-0.002

**Teljes költség:**
- **1 beszélgetés**: ~$0.002-0.003 (~0.6-0.9 Ft)
- **35 beszélgetés (batch)**: ~$0.07-0.10 (~21-30 Ft)

## Fejlesztési lehetőségek

### 1. Több persona és goal

Egyszerűen bővíthető:

```python
# personas.py-ban
NEW_PERSONA = Persona(
    name="Frustrated Expert",
    patience=2,
    expertise="expert",
    ...
)

PERSONAS["frustrated_expert"] = NEW_PERSONA
```

### 2. Egyedi evaluation dimenziók

Új prompt hozzáadása `evaluator.py`-ban:

```python
CUSTOM_DIMENSION_PROMPT = """
...
"""

def _evaluate_custom_dimension(self, conversation: str) -> Dict[str, Any]:
    prompt = CUSTOM_DIMENSION_PROMPT.format(conversation=conversation)
    return self._call_llm_judge(prompt)
```

### 3. Vizualizációk ✅ KÉSZ

**14 grafikon automatikusan generálódik** batch evaluation után!

Lásd: `results/run_TIMESTAMP/plots/` könyvtár vagy `python3 visualize_results.py`

További ötletek:
- Time series elemzés (több batch futtatás összehasonlítása)
- Interaktív dashboard (Streamlit/Dash)
- Exportálás PDF report-ba

### 4. A/B testing

Különböző RAG konfig összehasonlítása:
- Reranking on/off
- Különböző embedding modellek
- Match threshold változtatás

## Troubleshooting

### Backend connection error

```
ERROR: Backend is not running!
```

**Megoldás**: Indítsd el a backend-et:

```bash
cd assistant
npm run dev
```

### OpenAI API error

```
Error: OpenAI API key required
```

**Megoldás**: Állítsd be az `.env` fájlban:

```bash
OPENAI_API_KEY=sk-...
```

### JSON parse error az evaluatorban

```
JSON parse error: Expecting value: line 1...
```

**Megoldás**: LLM nem adott vissza valid JSON-t. A kód automatikusan megpróbálja kinyerni a JSON blokkot, de néha elszáll. Próbáld újra vagy növeld a temperature-t 0.0-ról.

## Hivatkozások

- [RAG Evaluation Guide](https://www.promptingguide.ai/research/rag)
- [LLM as a Judge](https://arxiv.org/abs/2306.05685)
- [Multi-Turn Conversation Evaluation](https://arxiv.org/abs/2311.16383)

## Licensz

Ez a projekt az AI-asszisztens-fejlesztés házi feladat része (Cubix).
