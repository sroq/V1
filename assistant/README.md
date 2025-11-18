# RAG Chat Assistant - The Jungle Book

Next.js alapú AI chat asszisztens RAG (Retrieval-Augmented Generation) technológiával. Az alkalmazás lehetővé teszi, hogy kérdéseket tegyél fel "The Jungle Book" című Rudyard Kipling műről, és a rendszer a dokumentum tartalmából kikeresett releváns információk alapján válaszol.

## Funkciók

- **RAG-alapú válaszgenerálás**: A rendszer vektoros keresést használ a releváns dokumentumrészletek megtalálásához
- **Streaming válaszok**: Valós idejű streaming válaszok GPT-4o mini használatával
- **Chat interface**: Tiszta, modern felhasználói felület üzenetelőzményekkel
- **Vektor keresés**: pgvector támogatással PostgreSQL-ben
- **Responsive design**: Mobil és desktop eszközökön is használható

## Technológiai stack

- **Framework**: Next.js 15 (App Router)
- **AI SDK**: Vercel AI SDK (useChat hook, streamText)
- **LLM**: OpenAI GPT-4o mini
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimenzió)
- **Database**: PostgreSQL 16 + pgvector extension
- **UI**: React 18 + Tailwind CSS
- **Language**: TypeScript

## Előfeltételek

1. **Node.js** 18.x vagy újabb
2. **PostgreSQL** futó instance pgvector extension-nel
   - Lásd: a projekt gyökérben található `docker-compose.yml`
3. **OpenAI API key**
4. **Feldolgozott dokumentumok** a `document_chunks` táblában

## Telepítés

### 1. Függőségek telepítése

```bash
cd assistant
npm install
```

### 2. Környezeti változók beállítása

A `.env.local` fájl már tartalmazza a szükséges konfigurációt:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key

# Database Configuration
DATABASE_URL=postgresql://rag_user:rag_dev_password_2024@localhost:5432/rag_assistant

# Embedding Model Configuration
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSION=1536

# Vector Search Configuration
DEFAULT_MATCH_COUNT=5
DEFAULT_MATCH_THRESHOLD=0.3
```

### 3. Adatbázis ellenőrzése

Győződj meg róla, hogy:
- A PostgreSQL konténer fut (`hf4-v1-postgres`)
- A `documents` és `document_chunks` táblák léteznek
- Van legalább egy feldolgozott dokumentum (The Jungle Book)

```bash
# Ellenőrzés:
psql postgresql://rag_user:rag_dev_password_2024@localhost:5432/rag_assistant

# SQL:
SELECT COUNT(*) FROM document_chunks;
```

## Futtatás

### Development mode

```bash
npm run dev
```

Az alkalmazás elérhető lesz: http://localhost:3000

### Production build

```bash
npm run build
npm start
```

## Projekt struktúra

```
assistant/
├── app/
│   ├── api/
│   │   └── chat/
│   │       └── route.ts          # RAG API endpoint (POST /api/chat)
│   ├── globals.css                # Tailwind CSS és global styles
│   ├── layout.tsx                 # Root layout
│   └── page.tsx                   # Chat UI főoldal
├── lib/
│   ├── db.ts                      # PostgreSQL connection pool
│   ├── embeddings.ts              # OpenAI embedding generation
│   └── rag.ts                     # RAG logic (retrieval + context)
├── .env.local                     # Környezeti változók
├── package.json
├── tsconfig.json
└── README.md
```

## Architektúra

### RAG Pipeline

1. **User Query** → Frontend input
2. **Embedding Generation** → OpenAI text-embedding-3-small
3. **Vector Search** → pgvector similarity search (cosine distance)
4. **Context Assembly** → Top 5 releváns chunk összefűzése
5. **System Prompt** → Context beillesztése a prompt-ba
6. **LLM Generation** → GPT-4o mini streaming válasz
7. **Streaming Response** → Valós idejű megjelenítés

### API Route Flow

**POST /api/chat**

```typescript
Request:
{
  messages: [
    { role: "user", content: "Who is Mowgli?" }
  ]
}

Response: Stream (text/event-stream)
- Streaming text chunks from GPT-4o mini
```

**Belső folyamat:**

1. Utolsó user message kinyerése
2. Embedding generálása a kérdéshez
3. Vector similarity search:
   ```sql
   SELECT content, 1 - (embedding <=> $1::vector) AS similarity
   FROM document_chunks
   WHERE 1 - (embedding <=> $1::vector) >= 0.7
   ORDER BY embedding <=> $1::vector
   LIMIT 5
   ```
4. Context összeállítása
5. System prompt készítése a context-tel
6. GPT-4o mini hívása streaming mode-ban
7. Response továbbítása a kliensnek

### Frontend Components

**Chat Interface (app/page.tsx)**

- `useChat` hook a Vercel AI SDK-ból
- Automatikus streaming kezelés
- Message history megjelenítése
- Input form és send gomb
- Auto-scroll új üzenetekhez
- Loading és error states

## Használat

1. Nyisd meg az alkalmazást: http://localhost:3000
2. Írj be egy kérdést a The Jungle Book-ról
3. Várd meg a streaming választ
4. Folytasd a beszélgetést

**Példa kérdések:**

- "Who is Mowgli?"
- "What is the Law of the Jungle?"
- "Tell me about Shere Khan"
- "What happens in the story?"
- "Describe the relationship between Mowgli and Baloo"

## Testreszabás

### Vektor keresés paraméterei

`.env.local`:
```env
DEFAULT_MATCH_COUNT=5          # Visszaadott chunks száma (1-20)
DEFAULT_MATCH_THRESHOLD=0.3    # Minimum similarity (0.0-1.0)
```

### GPT-4o mini beállítások

`app/api/chat/route.ts`:
```typescript
const result = streamText({
  model: openai('gpt-4o-mini'),
  messages: modelMessages,
  temperature: 0.7,      // Kreativitás szintje (0.0-1.0)
  maxTokens: 1000,       // Maximum válaszhossz
});
```

### System Prompt testreszabása

`lib/rag.ts` → `buildSystemPrompt()` függvény

## Hibaelhárítás

### "Database connection failed"

- Ellenőrizd, hogy a PostgreSQL konténer fut
- Teszteld a kapcsolatot: `psql $DATABASE_URL`
- Ellenőrizd a `DATABASE_URL` környezeti változót

### "No relevant context found"

- Győződj meg róla, hogy vannak feldolgozott chunks az adatbázisban
- Csökkentsd a `DEFAULT_MATCH_THRESHOLD` értékét
- Növeld a `DEFAULT_MATCH_COUNT` értékét

### "OpenAI API error"

- Ellenőrizd az `OPENAI_API_KEY` környezeti változót
- Győződj meg róla, hogy van elegendő kredit az OpenAI accountodon
- Ellenőrizd a hálózati kapcsolatot

### Streaming nem működik

- Ellenőrizd a böngésző konzolt
- Győződj meg róla, hogy a `/api/chat` route elérhető
- Teszteld a Chrome DevTools Network tabban

## További fejlesztési lehetőségek

- [ ] Conversation history perzisztálása adatbázisban
- [ ] Multi-document támogatás (több könyv)
- [ ] Citation markers (forrás jelölése a válaszokban)
- [ ] Advanced filtering (dátum, karakter, stb.)
- [ ] Chat export funkció
- [ ] Rate limiting
- [ ] User authentication
- [ ] Analytics és logging
- [ ] A/B testing különböző RAG stratégiákkal

## Licenc

Ez egy oktatási projekt. Részletek: lásd a projekt gyökérben.

## Kapcsolat

További információkért lásd a `REQUIREMENTS.md` és `CHUNKING_PIPELINE_SUMMARY.md` fájlokat a projekt gyökérben.
