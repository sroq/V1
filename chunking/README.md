# RAG Asszisztens Chunking Pipeline

Átfogó, production-ready dokumentum chunking rendszer RAG-alapú AI asszisztensekhez. Ez a pipeline feldolgozza a dokumentumokat, több stratégia használatával chunk-okat generál, OpenAI-n keresztül embedding-eket készít, és mindent feltölt PostgreSQL-be pgvector-ral.

## Funkciók

- **4 Chunking Stratégia**: Fix méretű, Szemantikus, Rekurzív, és Dokumentum típus specifikus
- **Több Dokumentum Formátum**: PDF, DOCX, TXT, Markdown, HTML
- **OpenAI Embedding-ek**: Batch feldolgozás retry logikával és rate limiting-gel
- **PostgreSQL/pgvector Integráció**: Hatékony vektor tárolás és lekérdezés
- **Progress Követés**: Megszakított feldolgozás folytatása
- **Gazdag Metaadat Kinyerés**: Dokumentum struktúra, címsorok, oldalszámok
- **Átfogó Naplózás**: Részletes feldolgozási logok és hibakezelés
- **Költségbecslés**: Embedding generálási költségek számítása
- **Konfigurálható**: YAML konfiguráció CLI felülírásokkal

## Projekt Struktúra

```
chunking/
├── __init__.py           # Package inicializálás
├── requirements.txt      # Python függőségek
├── config.yaml          # Konfiguráció template
├── chunker.py          # Fő CLI belépési pont
├── loader.py           # Dokumentum betöltő unstructured használatával
├── strategies.py       # 4 chunking stratégia implementáció
├── embeddings.py       # OpenAI embedding generálás
├── database.py         # PostgreSQL/pgvector feltöltő
├── utils.py           # Naplózás, progress követés, segédfunkciók
└── README.md          # Ez a fájl
```

## Telepítés

### Előfeltételek

- Python 3.10+
- PostgreSQL pgvector extension-nel
- OpenAI API kulcs

### Beállítás

1. **Függőségek telepítése**:

```bash
pip install -r chunking/requirements.txt
```

2. **Környezeti változók konfigurálása**:

Hozz létre egy `.env` fájlt a projekt gyökérkönyvtárában:

```bash
# OpenAI API Kulcs
OPENAI_API_KEY=your-openai-api-key-here

# Adatbázis hitelesítési adatok (ha eltér a config.yaml-tól)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_assistant
DB_USER=rag_user
DB_PASSWORD=rag_dev_password_2024
```

3. **Adatbázis beállítás ellenőrzése**:

Győződj meg róla, hogy a PostgreSQL pgvector-ral fut és a szükséges táblák léteznek:

```sql
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

-- Indexek létrehozása
CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops);
```

## Használat

### Alapvető Használat

**Egyetlen fájl feldolgozása**:

```bash
python chunking/chunker.py --input /path/to/document.pdf
```

**Könyvtár feldolgozása**:

```bash
python chunking/chunker.py --input /path/to/documents/
```

**Feltöltés adatbázisba**:

```bash
python chunking/chunker.py --input /path/to/documents/ --upload
```

### Chunking Stratégiák

#### 1. Fix Méretű Chunking

Fix méretű chunk-okra bontja a dokumentumokat átfedéssel.

```bash
python chunking/chunker.py \
    --input documents/ \
    --strategy fixed \
    --chunk-size 512 \
    --chunk-overlap 50 \
    --upload
```

**Használati esetek**:
- Konzisztens chunk méretek szükségesek
- Token limitek az embedding-ekhez
- Egyenletes feldolgozás

**Konfiguráció**:
```yaml
chunking:
  fixed_size:
    chunk_size: 512        # token-ek
    chunk_overlap: 50      # token-ek
```

#### 2. Szemantikus Chunking

Tiszteletben tartja a dokumentum struktúráját és szemantikai határait (bekezdések, szakaszok).

```bash
python chunking/chunker.py \
    --input documents/ \
    --strategy semantic \
    --upload
```

**Használati esetek**:
- Dokumentum struktúra megőrzése
- Kapcsolódó tartalom együtt tartása
- Jobb kontextus a RAG-hez

**Konfiguráció**:
```yaml
chunking:
  semantic:
    max_chunk_size: 1024
    respect_boundaries: true
    combine_short: true
```

#### 3. Rekurzív Chunking

Hierarchikus elválasztókat használ (címsorok, bekezdések, mondatok, szavak).

```bash
python chunking/chunker.py \
    --input documents/ \
    --strategy recursive \
    --chunk-size 512 \
    --chunk-overlap 50 \
    --upload
```

**Használati esetek**:
- Markdown/strukturált dokumentumok
- Hierarchiák megőrzése
- Intelligens felosztás

**Konfiguráció**:
```yaml
chunking:
  recursive:
    separators:
      - "\n## "      # H2
      - "\n### "     # H3
      - "\n\n"       # Bekezdés
      - "\n"         # Sor
      - ". "         # Mondat
      - " "          # Szó
    chunk_size: 512
    chunk_overlap: 50
```

#### 4. Dokumentum Típus Specifikus

Különböző stratégiák különböző dokumentum típusokhoz (Markdown, PDF, Code).

```bash
python chunking/chunker.py \
    --input documents/ \
    --strategy document_specific \
    --upload
```

**Használati esetek**:
- Vegyes dokumentum típusok
- Optimális kezelés formátumonként
- Specializált feldolgozás

**Konfiguráció**:
```yaml
chunking:
  document_specific:
    markdown:
      preserve_headings: true
      chunk_by_section: true
    pdf:
      preserve_pages: true
      extract_tables: true
    code:
      preserve_functions: true
      language_aware: true
```

### Haladó Használat

**Egyéni konfigurációs fájl**:

```bash
python chunking/chunker.py \
    --input documents/ \
    --config my_config.yaml \
    --upload
```

**Beállítás validálása feldolgozás előtt**:

```bash
python chunking/chunker.py \
    --input documents/ \
    --validate
```

**Mentett progress törlése és újraindítás**:

```bash
python chunking/chunker.py \
    --input documents/ \
    --clear-progress \
    --upload
```

**Egyéni batch méret embedding-ekhez**:

```bash
python chunking/chunker.py \
    --input documents/ \
    --batch-size 50 \
    --upload
```

**Debug mód**:

```bash
python chunking/chunker.py \
    --input documents/ \
    --log-level DEBUG
```

## Konfiguráció

A pipeline egy YAML konfigurációs fájlt (`config.yaml`) használ a következő szekciókkal:

### Dokumentum Betöltés

```yaml
document_loading:
  supported_extensions:
    - .pdf
    - .docx
    - .txt
    - .md
    - .html
  recursive: true
  max_file_size_mb: 50
```

### Chunking Stratégiák

```yaml
chunking:
  default_strategy: semantic  # fixed, semantic, recursive, document_specific

  # Stratégia-specifikus konfigurációk...
```

### Embedding-ek

```yaml
embeddings:
  provider: openai
  model: text-embedding-3-small
  dimensions: 1536
  batch_size: 100
  max_retries: 3
  retry_delay: 1.0
  timeout: 30
```

### Adatbázis

```yaml
database:
  host: localhost
  port: 5432
  database: rag_assistant
  user: rag_user
  password: rag_dev_password_2024
  update_strategy: replace  # replace, version, upsert
  batch_size: 500
```

### Feldolgozás

```yaml
processing:
  parallel: false
  max_workers: 4
  progress_bar: true
  save_progress: true
  progress_file: .chunking_progress.json
```

### Naplózás

```yaml
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: chunking_pipeline.log
  console: true
```

## Parancssori Interfész

### Argumentumok

| Argumentum | Típus | Leírás |
|----------|------|-------------|
| `--input` | Útvonal | **Kötelező**. Bemeneti fájl vagy könyvtár útvonala |
| `--config` | Útvonal | Konfigurációs fájl útvonala (alapértelmezett: config.yaml) |
| `--strategy` | Választás | Chunking stratégia (fixed, semantic, recursive, document_specific) |
| `--chunk-size` | Egész szám | Chunk méret token-ekben (fixed/recursive esetén) |
| `--chunk-overlap` | Egész szám | Chunk átfedés token-ekben (fixed/recursive esetén) |
| `--upload` | Flag | Chunk-ok feltöltése adatbázisba |
| `--batch-size` | Egész szám | Batch méret embedding generáláshoz |
| `--validate` | Flag | Beállítás validálása és kilépés |
| `--clear-progress` | Flag | Mentett progress törlése indítás előtt |
| `--log-level` | Választás | Naplózási szint (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `--env-file` | Útvonal | .env fájl útvonala (alapértelmezett: .env) |

## Architektúra

### Pipeline Folyamat

```
Bemeneti Dokumentumok
    ↓
Dokumentum Betöltő (unstructured)
    ↓
Chunking Stratégia
    ↓
Chunk-ok + Metaadatok
    ↓
Embedding Generátor (OpenAI)
    ↓
Embedding-ek
    ↓
Adatbázis Feltöltő (PostgreSQL/pgvector)
    ↓
Tárolt Vektorok
```

### Komponensek

#### 1. DocumentLoader (`loader.py`)

- Dokumentumokat tölt be az unstructured library használatával
- Több formátumot támogat (PDF, DOCX, TXT, MD, HTML)
- Gazdag metaadatokat nyer ki (struktúra, címsorok, oldalszámok)
- Fájlméreteket és formátumokat validál

#### 2. ChunkingStrategy (`strategies.py`)

Négy stratégia implementáció:

- **FixedSizeChunker**: Fix méretű chunk-ok átfedéssel
- **SemanticChunker**: Szemantikai határokat tiszteletben tartó
- **RecursiveChunker**: Hierarchikus felosztás
- **DocumentTypeSpecificChunker**: Típus-tudatos feldolgozás

#### 3. EmbeddingGenerator (`embeddings.py`)

- OpenAI API integráció
- Batch feldolgozás költség optimalizáláshoz
- Retry logika exponenciális backoff-fal
- Rate limiting kezelés
- Költségbecslés

#### 4. DatabaseUploader (`database.py`)

- PostgreSQL/pgvector integráció
- Batch feltöltés
- Tranzakció kezelés
- Frissítési stratégiák (replace, version, upsert)

#### 5. Segédfunkciók (`utils.py`)

- Progress követés folytatási lehetőséggel
- Strukturált naplózás
- Feldolgozási statisztikák
- Konfiguráció kezelés
- Token becslés

## Hibakezelés

A pipeline átfogó hibakezelést tartalmaz:

### Retry Logika

- OpenAI API hívások újrapróbálása rate limit-ek és átmeneti hibák esetén
- Exponenciális backoff konfigurálható késleltetésekkel
- Maximum újrapróbálkozási kísérletek konfigurálhatók

### Biztonságos Degradáció

- Nem támogatott fájltípusok kihagyása figyelmeztetésekkel
- Feldolgozás folytatása egyedi fájl hibák esetén
- Progress mentése folytatáshoz

### Naplózás

- Részletes logok debuggoláshoz
- Hibaüzenetek kontextussal
- Feldolgozási statisztikák

## Progress Követés

A pipeline elmenti a progresst `.chunking_progress.json` fájlba:

```json
{
  "processed_files": [
    "/path/to/doc1.pdf",
    "/path/to/doc2.md"
  ],
  "stats": {
    "total_files": 10,
    "processed_files": 5,
    "failed_files": 1,
    "total_chunks": 1250,
    "total_embeddings": 1250,
    "total_uploaded": 1250
  },
  "last_updated": "2025-11-01T10:30:00"
}
```

Megszakított feldolgozás folytatásához:

```bash
python chunking/chunker.py --input documents/ --upload
```

Újrakezdéshez:

```bash
python chunking/chunker.py --input documents/ --clear-progress --upload
```

## Teljesítmény Optimalizálás

### Batch Feldolgozás

- Embedding-ek batch-ekben generálódnak (alapértelmezett: 100)
- Adatbázis beszúrások tranzakciókban (alapértelmezett: 500)
- Csökkenti az API hívásokat és adatbázis köröket

### Token Becslés

- Gyors becslés (1 token ≈ 4 karakter)
- Elkerüli a tokenizer overhead-et
- Jó chunk méretezéshez

### Gyorsítótárazás

- Tartalom hash-alapú deduplikáció
- Már feldolgozott fájlok kihagyása
- Embedding cache (memóriában)

## Költségbecslés

Becsüld meg az embedding költségeket feldolgozás előtt:

```python
from chunking.embeddings import EmbeddingGenerator

embedder = EmbeddingGenerator(api_key="your-key")
cost_info = embedder.estimate_cost(num_chunks=1000)

print(f"Becsült költség: ${cost_info['estimated_cost_usd']}")
```

A text-embedding-3-small esetén:
- $0.02 / 1M token
- Átlagos chunk: ~400 token
- 1000 chunk ≈ $0.008

## Adatbázis Séma

### documents Tábla

| Oszlop | Típus | Leírás |
|--------|------|-------------|
| id | SERIAL | Elsődleges kulcs |
| file_path | TEXT | Egyedi fájl útvonal |
| file_name | TEXT | Fájlnév |
| file_type | TEXT | Dokumentum típus |
| file_size | INTEGER | Fájlméret byte-okban |
| created_at | TIMESTAMP | Fájl létrehozási idő |
| modified_at | TIMESTAMP | Fájl módosítási idő |
| metadata | JSONB | Egyéni metaadatok |
| processed_at | TIMESTAMP | Feldolgozási időbélyeg |

### document_chunks Tábla

| Oszlop | Típus | Leírás |
|--------|------|-------------|
| id | SERIAL | Elsődleges kulcs |
| document_id | INTEGER | Idegen kulcs documents-hoz |
| chunk_id | TEXT | Egyedi chunk azonosító |
| content | TEXT | Chunk tartalom |
| chunk_index | INTEGER | Chunk pozíció a dokumentumban |
| metadata | JSONB | Chunk metaadatok |
| token_count | INTEGER | Becsült token szám |
| content_hash | TEXT | SHA-256 tartalom hash |
| embedding | vector(1536) | Embedding vektor |
| created_at | TIMESTAMP | Létrehozási időbélyeg |

## Programozott Használat

A pipeline komponenseket programozottan is használhatod:

```python
from pathlib import Path
from chunking.loader import DocumentLoader
from chunking.strategies import SemanticChunker
from chunking.embeddings import EmbeddingGenerator
from chunking.database import DatabaseUploader

# Dokumentum betöltése
loader = DocumentLoader()
doc = loader.load_document(Path("document.pdf"))

# Dokumentum chunkolása
chunker = SemanticChunker(max_chunk_size=1024)
chunks = chunker.chunk_document(doc)

# Embedding-ek generálása
embedder = EmbeddingGenerator(api_key="your-key")
embeddings = embedder.generate_embeddings(chunks)

# Feltöltés adatbázisba
with DatabaseUploader(
    host="localhost",
    port=5432,
    database="rag_assistant",
    user="rag_user",
    password="password"
) as uploader:
    doc_id, num_chunks = uploader.upload_document(
        metadata=doc.metadata,
        chunks=chunks,
        embeddings=embeddings
    )
    print(f"Feltöltve {doc_id} dokumentum {num_chunks} chunk-kal")
```

## Hibaelhárítás

### Gyakori Problémák

**1. OpenAI API Kulcs Hiba**

```
Error: OPENAI_API_KEY environment variable not set
```

Megoldás: Állítsd be az API kulcsot `.env` fájlban vagy környezetben:
```bash
export OPENAI_API_KEY=your-key-here
```

**2. Adatbázis Csatlakozási Hiba**

```
Error: Failed to connect to database
```

Megoldás: Ellenőrizd hogy a PostgreSQL fut és a hitelesítési adatok helyesek:
```bash
psql -h localhost -p 5432 -U rag_user -d rag_assistant
```

**3. pgvector Extension Hiányzik**

```
Error: pgvector extension is not installed
```

Megoldás: Telepítsd és engedélyezd a pgvector-t:
```sql
CREATE EXTENSION vector;
```

**4. Rate Limit Hibák**

```
Error: Rate limit exceeded
```

Megoldás: A pipeline automatikusan újrapróbálkozik. Csökkentsd a `batch_size`-t a konfigban:
```yaml
embeddings:
  batch_size: 50  # Csökkentsd 100-ról
```

**5. Nagy Fájl Kihagyva**

```
Warning: File too large (75.5MB > 50MB)
```

Megoldás: Növeld a `max_file_size_mb`-t a konfigban:
```yaml
document_loading:
  max_file_size_mb: 100
```

## Tesztelés

Teszteld a pipeline-t egy minta dokumentummal:

```bash
# Teszt dokumentum létrehozása
echo "# Teszt Dokumentum

Ez egy teszt dokumentum a chunking pipeline számára.

## 1. Szakasz
Tartalom az 1. szakaszhoz.

## 2. Szakasz
Tartalom a 2. szakaszhoz." > test.md

# Feldolgozás szemantikus chunking-gal
python chunking/chunker.py \
    --input test.md \
    --strategy semantic \
    --log-level DEBUG

# Log kimenet ellenőrzése
cat chunking_pipeline.log
```

## Legjobb Gyakorlatok

1. **Válaszd ki a megfelelő stratégiát**:
   - Használd a `semantic`-ot a legtöbb esetben (megőrzi a struktúrát)
   - Használd a `fixed`-et konzisztens chunk méretekhez
   - Használd a `recursive`-t Markdown-hoz címsorokkal
   - Használd a `document_specific`-ot vegyes típusokhoz

2. **Hangold a chunk méreteket**:
   - Kezdd 512 token-nel a kiegyensúlyozott teljesítményhez
   - Növeld 1024-re több kontextusért
   - Csökkentsd 256-ra pontos visszakereséshez

3. **Monitorozd a költségeket**:
   - Használd az `estimate_cost()`-ot feldolgozás előtt
   - Batch feldolgozás csökkenti az API hívásokat
   - Cache-eld az embedding-eket újraszámítás elkerülésére

4. **Használj progress követést**:
   - Engedélyezd a `save_progress`-t nagy batch-ekhez
   - Folytasd a megszakított feldolgozást
   - Kövesd a statisztikákat monitorozáshoz

5. **Kezeld a hibákat kecsesen**:
   - Nézd át a logokat sikertelen fájlokért
   - Próbálkozz újra más stratégiákkal
   - Ugord át a problémás fájlokat ha szükséges

## A Pipeline Bővítése

### Új Chunking Stratégia Hozzáadása

```python
from chunking.strategies import ChunkingStrategy, Chunk

class CustomChunker(ChunkingStrategy):
    def chunk_document(self, document):
        # Egyéni chunking logika
        chunks = []
        # ...
        return chunks
```

### Egyéni Metaadat Kinyerés

Módosítsd a `loader.py`-t:

```python
def _extract_metadata(self, file_path, elements):
    metadata = super()._extract_metadata(file_path, elements)
    # Adj hozzá egyéni metaadatokat
    metadata.custom_metadata['my_field'] = 'my_value'
    return metadata
```

## Licenc

Ez a projekt a RAG Asszisztens rendszer része.

## Támogatás

Problémák vagy kérdések esetén:
1. Nézd meg a hibaelhárítási szekciót
2. Tekintsd át a logokat a `chunking_pipeline.log`-ban
3. Futtasd `--log-level DEBUG`-gal részletes kimenetért

## Verzió Történet

- **1.0.0** (2025-11-01): Kezdeti kiadás
  - 4 chunking stratégia
  - OpenAI embedding integráció
  - PostgreSQL/pgvector támogatás
  - Progress követés és naplózás
