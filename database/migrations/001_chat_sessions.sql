-- ============================================================================
-- Chat Sessions & Conversation History Migration
-- ============================================================================
--
-- Ez a migration script 3 új táblát hoz létre a backend session/conversation
-- management támogatásához:
--
-- 1. chat_sessions - Session tracking és metadata
-- 2. chat_messages - Teljes üzenet történet (user + assistant)
-- 3. chat_rag_context - RAG metaadatok (chunk IDs, similarity scores, reranking)
--
-- Követelmények:
-- - Session ID és timestamp tárolása
-- - Teljes üzenet történet (user + assistant messages)
-- - RAG context metaadatok (chunks, scores, reranking results)
-- - Határozatlan retention (NINCS auto-cleanup)
-- - Egy session = egy beszélgetés (linear chat)
--
-- ============================================================================

-- ============================================================================
-- 1. CHAT_SESSIONS TÁBLA - SESSION TRACKING
-- ============================================================================

-- Chat session-ök tárolása - Egy session egy folytonos beszélgetést reprezentál.
--
-- Session lifecycle:
-- 1. Frontend POST /api/chat hívás (x-session-id header nélkül)
-- 2. Backend generál UUID v4 session ID-t
-- 3. Session létrehozása: INSERT INTO chat_sessions
-- 4. Minden üzenet hozzáadása ehhez a session-höz
-- 5. Frontend (opcionálisan) eltárolja a session ID-t localStorage-ban
-- 6. Következő híváskor elküldi x-session-id header-ben
-- 7. Backend folytatja a session-t (getOrCreateSession)
--
-- Session modell: Egy session = egy beszélgetés
-- - NINCS multi-conversation support (conversation_id)
-- - Lineáris chat flow: message_1, message_2, ...
-- - Page refresh után folytatható (ha frontend küldi a session ID-t)
--
CREATE TABLE chat_sessions (
    -- Egyedi session azonosító (UUID v4)
    -- Példa: "550e8400-e29b-41d4-a716-446655440000"
    id VARCHAR(36) PRIMARY KEY,

    -- User azonosítás (opcionális, authentication esetén)
    -- NULL = anonymous session
    -- Ha lesz később user authentication, ezt kitöltjük
    user_id VARCHAR(255),

    -- Session neve/címe (opcionális)
    -- Lehet auto-generált az első user message alapján
    -- Példa: "Mowgli kérdések", "Shere Khan története"
    -- NULL = nincs explicit cím
    title VARCHAR(500),

    -- Session kezdési időpont
    -- Amikor az első üzenet létrejött
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Utolsó aktivitás időpontja
    -- Minden új üzenetnél frissül (UPDATE)
    -- Segít a session rendezésében ("recent conversations")
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Session létrehozási időpont (nem változik)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Rugalmas metadata tárolás (JSONB)
    -- Példa értékek:
    -- {
    --   "tags": ["jungle-book", "mowgli"],
    --   "source": "web-ui",
    --   "user_agent": "Mozilla/5.0...",
    --   "total_messages": 12,
    --   "avg_response_time_ms": 1500
    -- }
    metadata JSONB DEFAULT '{}',

    -- Indexek később: session_id (PRIMARY), user_id, created_at, last_activity_at

    -- Constraints
    CONSTRAINT check_session_id_format CHECK (
        id ~ '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    )
);

-- Indexek a chat_sessions táblához
--
-- idx_sessions_user_id: User-enkénti session-ök gyors lekérdezéséhez
-- WHERE záradék: Csak akkor indexel, ha user_id NOT NULL (partial index)
CREATE INDEX idx_sessions_user_id ON chat_sessions(user_id)
WHERE user_id IS NOT NULL;

-- idx_sessions_last_activity: "Recent conversations" lista rendezéséhez
-- DESC sorrend: Legfrissebb session-ök elől
CREATE INDEX idx_sessions_last_activity ON chat_sessions(last_activity_at DESC);

-- idx_sessions_created_at: Session létrehozási idő szerinti rendezés
CREATE INDEX idx_sessions_created_at ON chat_sessions(created_at DESC);

-- idx_sessions_metadata: JSONB mező GIN index (metadata alapú kereséshez)
-- Lehetővé teszi gyors keresést metadata kulcsok/értékek alapján
-- Példa: WHERE metadata @> '{"tags": ["mowgli"]}'
CREATE INDEX idx_sessions_metadata ON chat_sessions USING GIN(metadata);

-- Tábla komment
COMMENT ON TABLE chat_sessions IS
'Chat session tracking - Egy session egy folytonos beszélgetést reprezentál. Retention: határozatlan (nincs auto-cleanup).';

COMMENT ON COLUMN chat_sessions.id IS
'UUID v4 session azonosító. Backend generálja ha nincs x-session-id header.';

COMMENT ON COLUMN chat_sessions.last_activity_at IS
'Utolsó üzenet időpontja. Minden új message-nél UPDATE-elődik.';

-- ============================================================================
-- 2. CHAT_MESSAGES TÁBLA - TELJES ÜZENET TÖRTÉNET
-- ============================================================================

-- Chat üzenetek tárolása (user + assistant + system messages).
--
-- Message lifecycle:
-- 1. User elküldi a kérdést → POST /api/chat
-- 2. Backend INSERT user message → chat_messages
-- 3. RAG context retrieval
-- 4. LLM streaming válasz generálása
-- 5. Streaming befejezése után INSERT assistant message → chat_messages
-- 6. Frontend useChat hook frissíti a UI-t (de nem menti le)
--
-- Role típusok:
-- - 'user': User által küldött üzenet ("Who is Mowgli?")
-- - 'assistant': AI által generált válasz ("Mowgli is a human child...")
-- - 'system': System prompt (ritka, általában nem mentjük le)
--
-- Fontos: Az assistant message CSAK a streaming befejezése UTÁN kerül mentésre!
-- Így biztosítjuk, hogy a teljes generált válasz el van tárolva.
--
CREATE TABLE chat_messages (
    -- Egyedi üzenet azonosító (UUID v4)
    id VARCHAR(36) PRIMARY KEY,

    -- Session ID hivatkozás (FOREIGN KEY)
    -- CASCADE DELETE: Ha törlünk egy session-t, az összes üzenete is törlődik
    session_id VARCHAR(36) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,

    -- Üzenet szerepe (user | assistant | system)
    -- 'user': User üzenet
    -- 'assistant': AI válasz
    -- 'system': System prompt (ritkán használt)
    role VARCHAR(50) NOT NULL,

    -- Üzenet tartalma (teljes szöveg)
    -- User message példa: "Who is Mowgli and what is his story?"
    -- Assistant message példa: "Mowgli egy emberi gyermek, akit farkasok neveltek fel..."
    content TEXT NOT NULL,

    -- Üzenet létrehozási időpont
    -- User message: Amikor a kérés érkezik
    -- Assistant message: Amikor a streaming befejeződik
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Rugalmas metadata tárolás (JSONB)
    -- Példa értékek:
    -- {
    --   "token_count": 156,
    --   "generation_time_ms": 1234,
    --   "model": "gpt-4o-mini",
    --   "temperature": 0.7,
    --   "finish_reason": "stop",
    --   "has_rag_context": true,
    --   "rag_chunk_count": 5
    -- }
    metadata JSONB DEFAULT '{}',

    -- Constraints
    CONSTRAINT check_role CHECK (role IN ('user', 'assistant', 'system')),
    CONSTRAINT check_content_not_empty CHECK (LENGTH(content) > 0)
);

-- Indexek a chat_messages táblához
--
-- idx_messages_session_id: Session-enkénti üzenetek gyors lekérdezéséhez
-- Ez a legfontosabb index! Minden conversation history lekéréshez használjuk.
CREATE INDEX idx_messages_session_id ON chat_messages(session_id);

-- idx_messages_created_at: Időrendi rendezéshez (message ordering)
-- Compound index: (session_id, created_at)
-- Lehetővé teszi gyors rendezett lekérdezést:
-- SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at
CREATE INDEX idx_messages_session_created ON chat_messages(session_id, created_at);

-- idx_messages_role: Role típus szerinti szűréshez
-- Példa: "Csak user üzenetek" vagy "Csak assistant válaszok"
CREATE INDEX idx_messages_role ON chat_messages(role);

-- idx_messages_metadata: JSONB GIN index (metadata kereséshez)
CREATE INDEX idx_messages_metadata ON chat_messages USING GIN(metadata);

-- Tábla komment
COMMENT ON TABLE chat_messages IS
'Chat üzenetek teljes története (user + assistant + system). Minden üzenet hozzá van rendelve egy session-höz.';

COMMENT ON COLUMN chat_messages.role IS
'Üzenet szerepe: user (user üzenet), assistant (AI válasz), system (system prompt)';

COMMENT ON COLUMN chat_messages.content IS
'Teljes üzenet szöveg. Assistant esetén a teljes generált válasz (streaming után).';

-- ============================================================================
-- 3. CHAT_RAG_CONTEXT TÁBLA - RAG METAADATOK
-- ============================================================================

-- RAG context metadata tárolása - Mely chunk-ok lettek felhasználva az egyes
-- assistant válaszokhoz és milyen relevanciával.
--
-- RAG pipeline lépései (app/api/chat/route.ts):
-- 1. User üzenet → Embedding generálás
-- 2. Vector search → 15 chunk lekérése (initial retrieval)
-- 3. LLM reranking → GPT-4o mini pontozza a chunk-okat
-- 4. Blended scoring → 70% LLM + 30% embedding similarity
-- 5. Top-5 kiválasztása
-- 6. Context assembly → Formázott szöveg az LLM-nek
-- 7. LLM válasz generálás
-- 8. RAG metaadatok mentése → chat_rag_context tábla
--
-- Minden assistant message-hez (ahol használtunk RAG-ot) 5 sor kerül ide:
-- - Top-1 chunk (rank_position = 1, highest blended_score)
-- - Top-2 chunk (rank_position = 2)
-- - ...
-- - Top-5 chunk (rank_position = 5)
--
-- Miért fontos ez?
-- - Analytics: Mely chunk-ok a leghasznosabbak?
-- - Debugging: Miért adta vissza ezt a választ?
-- - Evaluation: Mennyire jó a reranking?
-- - Citation: Forrás megjelenítése a frontend-en (később)
--
CREATE TABLE chat_rag_context (
    -- Egyedi RAG context record azonosító
    id VARCHAR(36) PRIMARY KEY,

    -- Chat message ID hivatkozás (FOREIGN KEY)
    -- Melyik assistant message-hez tartozik ez a chunk?
    -- CASCADE DELETE: Ha törlünk egy message-t, a RAG context is törlődik
    chat_message_id VARCHAR(36) NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,

    -- Document chunk ID hivatkozás (FOREIGN KEY)
    -- Melyik chunk lett felhasználva?
    -- CASCADE DELETE: Ha törlünk egy chunk-ot, a RAG context is törlődik
    chunk_id UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,

    -- Embedding similarity score (0-1)
    -- Ez a pure vector search eredménye (cosine similarity)
    -- Példa: 0.8234 = 82.34% hasonlóság
    embedding_similarity NUMERIC(5,4),

    -- LLM relevance score (0-1)
    -- GPT-4o mini által adott relevanciapont (reranking)
    -- Példa: 0.9100 = 91% releváns
    llm_relevance_score NUMERIC(5,4),

    -- Blended score (0-1)
    -- Final score: 70% LLM + 30% embedding
    -- Példa: 0.70 * 0.91 + 0.30 * 0.82 = 0.883
    -- Ez alapján történik a top-5 kiválasztása!
    blended_score NUMERIC(5,4),

    -- Rank pozíció (1-5)
    -- 1 = Legjobb chunk (highest blended_score)
    -- 5 = Ötödik legjobb chunk
    rank_position INTEGER,

    -- Reranking blend weight (0-1)
    -- Jelenleg fix 0.7 (70% LLM, 30% embedding)
    -- Ha később változtatjuk, visszanézhetjük mi volt a weight
    reranking_blend_weight NUMERIC(3,2) DEFAULT 0.7,

    -- RAG context record létrehozási időpont
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Rugalmas metadata tárolás (JSONB)
    -- Példa értékek:
    -- {
    --   "initial_rank": 3,           // Rank BEFORE reranking (vector search)
    --   "rank_change": -2,            // -2 = dropped 2 positions after reranking
    --   "chunk_token_count": 256,
    --   "reranking_latency_ms": 523,
    --   "document_title": "The Jungle Book Chapter 1"
    -- }
    metadata JSONB DEFAULT '{}',

    -- Constraints
    -- UNIQUE: Egy message-hez egy chunk csak egyszer lehet (no duplicates)
    CONSTRAINT unique_message_chunk UNIQUE (chat_message_id, chunk_id),

    -- CHECK: Rank position 1-5 között
    CONSTRAINT check_rank_position CHECK (rank_position >= 1 AND rank_position <= 5),

    -- CHECK: Similarity scores 0-1 között
    CONSTRAINT check_embedding_similarity CHECK (
        embedding_similarity IS NULL OR (embedding_similarity >= 0 AND embedding_similarity <= 1)
    ),
    CONSTRAINT check_llm_relevance CHECK (
        llm_relevance_score IS NULL OR (llm_relevance_score >= 0 AND llm_relevance_score <= 1)
    ),
    CONSTRAINT check_blended_score CHECK (
        blended_score IS NULL OR (blended_score >= 0 AND blended_score <= 1)
    ),

    -- CHECK: Blend weight 0-1 között
    CONSTRAINT check_blend_weight CHECK (
        reranking_blend_weight >= 0 AND reranking_blend_weight <= 1
    )
);

-- Indexek a chat_rag_context táblához
--
-- idx_rag_message_id: Message-enkénti RAG context lekérdezéséhez
-- SELECT * FROM chat_rag_context WHERE chat_message_id = ?
CREATE INDEX idx_rag_message_id ON chat_rag_context(chat_message_id);

-- idx_rag_chunk_id: Chunk-enkénti usage tracking
-- "Hány alkalommal lett felhasználva ez a chunk?"
CREATE INDEX idx_rag_chunk_id ON chat_rag_context(chunk_id);

-- idx_rag_rank_position: Rank distribution analytics
-- "Hány chunk volt top-1? Top-2? ..."
CREATE INDEX idx_rag_rank_position ON chat_rag_context(rank_position);

-- idx_rag_blended_score: Score distribution analytics
-- "Milyen eloszlásúak a blended score-ok?"
CREATE INDEX idx_rag_blended_score ON chat_rag_context(blended_score DESC);

-- idx_rag_created_at: Időrendi analytics
CREATE INDEX idx_rag_created_at ON chat_rag_context(created_at DESC);

-- Tábla komment
COMMENT ON TABLE chat_rag_context IS
'RAG metaadatok assistant message-ekhez. Tárolja mely chunk-ok lettek felhasználva és milyen relevancia score-ral (embedding + LLM reranking).';

COMMENT ON COLUMN chat_rag_context.blended_score IS
'Final score: 70% LLM relevance + 30% embedding similarity. Ez alapján történik a top-5 kiválasztása.';

COMMENT ON COLUMN chat_rag_context.rank_position IS
'Rank pozíció (1-5). 1 = legjobb chunk (highest blended_score), 5 = ötödik legjobb.';

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Trigger function: update_session_activity_on_message
-- Automatikusan frissíti a chat_sessions.last_activity_at mezőt
-- minden új message beszúrásakor.
--
-- Működés:
-- 1. Új üzenet INSERT-elődik → chat_messages
-- 2. Trigger lefut
-- 3. UPDATE chat_sessions SET last_activity_at = NOW() WHERE id = NEW.session_id
-- 4. Session "friss" marad
--
CREATE OR REPLACE FUNCTION update_session_activity_on_message()
RETURNS TRIGGER AS $$
BEGIN
    -- Frissítjük a session last_activity_at mezőjét
    UPDATE chat_sessions
    SET last_activity_at = NOW()
    WHERE id = NEW.session_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: chat_messages táblán
-- AFTER INSERT: Minden új üzenet után lefut
CREATE TRIGGER trigger_update_session_activity
    AFTER INSERT ON chat_messages
    FOR EACH ROW
    EXECUTE FUNCTION update_session_activity_on_message();

-- ============================================================================
-- ANALYTICS VIEWS
-- ============================================================================

-- View: v_session_summary
-- Session-enkénti összesítés (message count, RAG usage, stb.)
--
-- Hasznos query-k:
-- - "Top 10 legtöbb üzenetet tartalmazó session"
-- - "Sessions with high RAG usage"
-- - "Average messages per session"
--
CREATE VIEW v_session_summary AS
SELECT
    s.id AS session_id,
    s.user_id,
    s.title,
    s.started_at,
    s.last_activity_at,
    COUNT(m.id) AS total_messages,
    COUNT(m.id) FILTER (WHERE m.role = 'user') AS user_messages,
    COUNT(m.id) FILTER (WHERE m.role = 'assistant') AS assistant_messages,
    COUNT(DISTINCT r.chunk_id) AS unique_chunks_used,
    AVG(r.blended_score) AS avg_rag_score,
    s.metadata AS session_metadata
FROM chat_sessions s
LEFT JOIN chat_messages m ON s.id = m.session_id
LEFT JOIN chat_rag_context r ON m.id = r.chat_message_id
GROUP BY s.id, s.user_id, s.title, s.started_at, s.last_activity_at, s.metadata
ORDER BY s.last_activity_at DESC;

COMMENT ON VIEW v_session_summary IS
'Session-enkénti összesítő nézet: message count, RAG usage, score átlagok.';

-- View: v_chunk_usage_stats
-- Chunk-enkénti usage analytics
--
-- "Mely chunk-ok a leghasznosabbak?"
-- "Hányszor lett felhasználva egy chunk?"
-- "Átlagos rank position?"
--
CREATE VIEW v_chunk_usage_stats AS
SELECT
    dc.id AS chunk_id,
    d.title AS document_title,
    dc.chunk_index,
    COUNT(r.id) AS usage_count,
    AVG(r.blended_score) AS avg_blended_score,
    AVG(r.llm_relevance_score) AS avg_llm_score,
    AVG(r.embedding_similarity) AS avg_embedding_similarity,
    AVG(r.rank_position) AS avg_rank_position,
    MIN(r.rank_position) AS best_rank,
    MAX(r.created_at) AS last_used_at
FROM document_chunks dc
INNER JOIN documents d ON dc.document_id = d.id
LEFT JOIN chat_rag_context r ON dc.id = r.chunk_id
GROUP BY dc.id, d.title, dc.chunk_index
HAVING COUNT(r.id) > 0
ORDER BY usage_count DESC, avg_blended_score DESC;

COMMENT ON VIEW v_chunk_usage_stats IS
'Chunk usage analytics: usage count, average scores, rank distribution.';

-- ============================================================================
-- MIGRATION COMPLETION
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Chat Sessions Migration Completed!';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Created tables:';
    RAISE NOTICE '  - chat_sessions (session tracking)';
    RAISE NOTICE '  - chat_messages (message history)';
    RAISE NOTICE '  - chat_rag_context (RAG metadata)';
    RAISE NOTICE '';
    RAISE NOTICE 'Created indexes: 13 total';
    RAISE NOTICE 'Created triggers: 1 (auto-update session activity)';
    RAISE NOTICE 'Created views: 2 (session_summary, chunk_usage_stats)';
    RAISE NOTICE '';
    RAISE NOTICE 'Retention policy: Határozatlan (nincs auto-cleanup)';
    RAISE NOTICE 'Session model: Egy session = egy beszélgetés (linear)';
    RAISE NOTICE '========================================';
END $$;
