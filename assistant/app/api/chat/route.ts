/**
 * Chat API Route - RAG-alapú Streaming Chat Endpoint
 *
 * Ez a Next.js App Router API endpoint implementálja a teljes RAG pipeline-t
 * és biztosítja a streaming chat válaszokat a frontend számára.
 *
 * RAG Pipeline lépései (end-to-end):
 * 1. User üzenet fogadása a chat UI-ból (POST /api/chat)
 * 2. Utolsó user üzenet kinyerése (context retrieval-hez)
 * 3. Vector similarity search → Releváns chunk-ok lekérése
 * 4. Context assembly → Formázott kontextus készítése
 * 5. System prompt → LLM számára előkészített prompt
 * 6. GPT-4o mini hívás streaming módban
 * 7. Streaming response → Real-time válasz küldése a frontend-nek
 *
 * Technológiai stack:
 * - Next.js 14 App Router: API route kezelés
 * - Vercel AI SDK: Streaming támogatás és OpenAI integráció
 * - OpenAI GPT-4o mini: LLM (KÖTELEZŐ követelmény)
 * - Custom RAG modul: Vector search és context assembly
 *
 * Endpoint részletei:
 * - Metódus: POST
 * - URL: /api/chat
 * - Input: { messages: ChatMessage[] }
 * - Output: Streaming text response
 * - Max időtartam: 30 másodperc
 *
 * ChatMessage formátum:
 *   {
 *     role: "user" | "assistant" | "system",
 *     content: string
 *   }
 */

import { openai } from '@ai-sdk/openai';
import { streamText } from 'ai';
import { getRelevantContextWithReranking, buildSystemPrompt } from '@/lib/rag';
import {
  getOrCreateSession,
  saveMessage,
  saveRAGContextBatch,
} from '@/lib/chat-storage';
import type { CreateChatSessionDTO, CreateChatRAGContextDTO } from '@/lib/types/chat';
import { trace } from '@opentelemetry/api';
import { recordChatCompletionCost } from '@/lib/cost-metrics';

// Streaming válaszok maximális időtartama
// Next.js App Router alapértelmezetten 5 másodperc timeout-ot használ
// Ez a beállítás 30 másodpercre növeli a timeout-ot
// Ez azért fontos, mert:
// - Vector search ~1-2 mp
// - OpenAI API hívás ~2-5 mp
// - Streaming válasz generálás ~5-15 mp
// - Buffer idő rate limit és lassú hálózat esetén
export const maxDuration = 30;

/**
 * POST /api/chat - Chat kérés kezelése RAG context retrieval-lel.
 *
 * Ez a függvény a Next.js App Router API route handler.
 * Minden POST /api/chat kérés ide érkezik a frontend-ről.
 *
 * Request flow:
 * 1. Request body parse-olása (JSON)
 * 2. Validáció (messages tömb ellenőrzése)
 * 3. Utolsó user üzenet kinyerése
 * 4. RAG context retrieval (vector search)
 * 5. System prompt készítése
 * 6. GPT-4o mini hívás streaming módban
 * 7. Streaming response visszaadása
 *
 * @param req - Next.js Request objektum
 * @returns Response objektum (streaming vagy JSON error)
 *
 * Request body formátum:
 *   {
 *     "messages": [
 *       { "role": "user", "content": "Who is Mowgli?" },
 *       { "role": "assistant", "content": "Mowgli is..." },
 *       { "role": "user", "content": "Tell me more about him." }
 *     ]
 *   }
 *
 * Response formátum (streaming):
 *   A Vercel AI SDK által generált streaming response.
 *   A frontend useChat hook automatikusan feldolgozza.
 *   Formátum: Server-Sent Events (SSE) stream
 *
 * Response formátum (hiba esetén):
 *   {
 *     "error": "Failed to process chat request",
 *     "message": "Detailed error message"
 *   }
 *
 * Példa használat (frontend):
 *   const { messages, input, handleSubmit } = useChat({
 *     api: '/api/chat'
 *   });
 *
 * HTTP status kódok:
 *   - 200: Sikeres streaming response
 *   - 400: Hibás request (hiányzó vagy invalid messages)
 *   - 500: Szerver hiba (DB error, OpenAI error, stb.)
 */
export async function POST(req: Request) {
  // Get current trace for adding attributes
  const currentSpan = trace.getActiveSpan();

  try {
    // ========================================================================
    // 1. REQUEST BODY PARSE-OLÁSA
    // ========================================================================
    // A frontend által küldött JSON parse-olása
    // Várható formátum: { messages: ChatMessage[] }
    const { messages } = await req.json();

    // ========================================================================
    // 2. VALIDÁCIÓ: MESSAGES TÖMB ELLENŐRZÉSE
    // ========================================================================
    // Ellenőrizzük, hogy:
    // - messages létezik
    // - messages egy tömb
    // - messages nem üres
    //
    // Ha bármelyik feltétel nem teljesül, 400 Bad Request-et küldünk
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return new Response('Missing or invalid messages array', {
        status: 400,
      });
    }

    // ========================================================================
    // 3. UTOLSÓ USER ÜZENET KINYERÉSE
    // ========================================================================
    // A RAG context retrieval-hez az UTOLSÓ user üzenetet használjuk
    //
    // Miért az utolsó user üzenet?
    // - Ez a legfrissebb kérdés/kérés
    // - Erre kell releváns kontextust találni
    // - Korábbi üzenetek csak a beszélgetés flow-jához kellenek
    //
    // Példa messages tömb:
    // [
    //   { role: "user", content: "Who is Mowgli?" },
    //   { role: "assistant", content: "Mowgli is a human child..." },
    //   { role: "user", content: "Tell me more about him." }  // <-- Ez kell!
    // ]
    const lastUserMessage = messages
      .filter((m: any) => m.role === 'user')  // Csak user üzenetek
      .pop();                                  // Utolsó elem

    // Ha nincs user üzenet, akkor hiba
    // (Elméletben nem fordulhat elő, de biztonság kedvéért ellenőrizzük)
    if (!lastUserMessage) {
      return new Response('No user message found', { status: 400 });
    }

    // Logging: milyen kérés érkezett
    console.log('Processing chat request:', {
      messageCount: messages.length,
      lastUserMessage: lastUserMessage.content.substring(0, 100),
    });

    // ========================================================================
    // 3.5. SESSION MANAGEMENT - BACKEND SESSION TRACKING
    // ========================================================================
    // Session ID kezelése és conversation history tárolása.
    //
    // Session ID forrásai (prioritási sorrend):
    // 1. Request header: x-session-id (ha frontend küldi)
    // 2. Auto-generate: crypto.randomUUID() (új session)
    //
    // Session flow:
    // 1. Session ID lekérése vagy generálása
    // 2. Session létrehozása/lekérése adatbázisból (getOrCreateSession)
    // 3. User message mentése → chat_messages tábla
    // 4. ... RAG pipeline ...
    // 5. Assistant message mentése streaming után
    // 6. RAG context mentése → chat_rag_context tábla
    //
    // Miért backend-only?
    // - Frontend NEM módosul (nincs session kezelés UI)
    // - Backend automatikusan ment minden beszélgetést
    // - Később frontend hozzáadható (session lista, history betöltés)

    // Session ID lekérése header-ből vagy generálás
    const requestSessionId = req.headers.get('x-session-id');
    const sessionId = requestSessionId ?? crypto.randomUUID();

    console.log('Session ID:', {
      sessionId,
      isNewSession: !requestSessionId,
    });

    // Session létrehozása vagy lekérése
    const sessionDTO: CreateChatSessionDTO = {
      id: sessionId,
      user_id: null,  // Anonymous (később user authentication)
      title: null,    // Opcionális: auto-generate első message-ből
      metadata: {
        source: 'api',
        user_agent: req.headers.get('user-agent') ?? 'unknown',
        first_message: !requestSessionId ? lastUserMessage.content.substring(0, 100) : undefined,
      },
    };

    const session = await getOrCreateSession(sessionId, sessionDTO);

    // User message mentése adatbázisba
    const savedUserMessage = await saveMessage({
      session_id: session.id,
      role: 'user',
      content: lastUserMessage.content,
      metadata: {
        message_index: messages.filter((m: any) => m.role === 'user').length,
        timestamp: new Date().toISOString(),
      },
    });

    console.log('Saved user message:', savedUserMessage.id);

    // Add session_id to trace context
    if (currentSpan) {
      currentSpan.setAttributes({
        'session.id': session.id,
        'message.user_id': savedUserMessage.id,
      });
    }

    // ========================================================================
    // 4. RAG CONTEXT RETRIEVAL WITH RERANKING
    // ========================================================================
    // getRelevantContextWithReranking() végrehajtja a teljes RAG pipeline-t
    // LLM-based reranking-gel:
    // 1. Query embedding generálása (lastUserMessage.content)
    // 2. Vector similarity search PostgreSQL-ben (pgvector)
    // 3. Top-15 chunk lekérése (initial retrieval - több mint amennyi kell)
    // 4. LLM reranking: GPT-4o mini pontozza minden chunk relevanciáját
    //    - 15 párhuzamos API hívás (~500-800ms)
    //    - Blended scoring: 70% LLM + 30% embedding similarity
    // 5. Top-5 kiválasztása a reranked eredményből
    // 6. Context assembly (formázott szöveg)
    //
    // Visszaadott objektum:
    // - context: Formázott kontextus string (LLM számára)
    // - chunks: DocumentChunk[] (reranked chunk-ok similarity score-ral)
    // - hasContext: boolean (van-e legalább 1 chunk)
    //
    // Reranking előnyei:
    // - Jobb pontosság komplex kérdéseknél (+15-25%)
    // - Érti a kérdés szándékát (nem csak szó-hasonlóságot néz)
    // - Narratív logika felismerése (ok-okozat, karakterek kapcsolata)
    //
    // Cost:
    // - ~$0.0008 per request (~$0.83 per 1000 requests)
    // - Latency: +500-800ms
    const { context, chunks, hasContext } = await getRelevantContextWithReranking(
      lastUserMessage.content,
      5,  // topK: Végső chunk-ok száma (amit az LLM kap kontextusként)
      15  // initialLimit: Initial retrieval chunk száma (reranking előtt)
    );

    // Logging: milyen kontextust találtunk
    console.log('Retrieved context:', {
      chunkCount: chunks.length,
      hasContext,
      avgSimilarity:
        chunks.length > 0
          ? (
              chunks.reduce((sum, c) => sum + c.similarity, 0) / chunks.length
            ).toFixed(3)
          : 'N/A',
    });

    // ========================================================================
    // 5. SYSTEM PROMPT KÉSZÍTÉSE KONTEXTUSSAL
    // ========================================================================
    // buildSystemPrompt() készíti el a system prompt-ot
    // Ez tartalmazza:
    // - Szerep definíció (AI assistant for The Jungle Book)
    // - Utasítások (ONLY use context, be concise, stb.)
    // - Kontextus (formázott chunk-ok)
    // - Reminder (context-based answering)
    const systemPrompt = buildSystemPrompt(context);

    // ========================================================================
    // 6. MESSAGES ELŐKÉSZÍTÉSE AZ LLM SZÁMÁRA
    // ========================================================================
    // Az LLM-nek küldött messages tömb összeállítása:
    // - System message (system prompt kontextussal) MINDIG az ELSŐ helyen
    // - Majd az összes többi üzenet (user, assistant, ...)
    //
    // Fontos: A system message-nek MINDIG az első helyen kell lennie!
    // Ez határozza meg az LLM viselkedését az egész beszélgetésben.
    //
    // Ha a messages tömb már tartalmaz system message-t (első elem),
    // akkor LECSERÉLJÜK a mi system prompt-unkkal.
    // Ha nem tartalmaz, akkor HOZZÁADJUK az elejére.
    //
    // Miért cserélünk?
    // - Minden kéréshez friss kontextust kell küldeni
    // - A frontend nem tudja előre a kontextust (csak a backend)
    // - Így biztosítjuk, hogy a legfrissebb releváns chunk-ok legyenek benne
    const modelMessages = messages[0]?.role === 'system'
      ? [{ role: 'system', content: systemPrompt }, ...messages.slice(1)]
      : [{ role: 'system', content: systemPrompt }, ...messages];

    // ========================================================================
    // 7. GPT-4O MINI HÍVÁS STREAMING MÓDBAN
    // ========================================================================
    // streamText() a Vercel AI SDK függvénye streaming response-hoz
    //
    // Paraméterek:
    // - model: openai('gpt-4o-mini') - GPT-4o mini használata (KÖTELEZŐ!)
    // - messages: modelMessages - System prompt + user/assistant üzenetek
    // - temperature: 0.7 - Kreativitás szintje (0-2, ahol 0=determinisztikus, 2=nagyon kreatív)
    // - maxTokens: 1000 - Maximum generált token szám
    //
    // Temperature magyarázat:
    //   0.0-0.3: Determinisztikus, precíz válaszok (fact-based Q&A)
    //   0.4-0.7: Kiegyensúlyozott (általános chat) - MI HASZNÁLJUK
    //   0.8-1.0: Kreatív, variábilis válaszok (kreatív írás)
    //   1.0+: Nagyon kreatív, kevésbé konzisztens
    //
    // MaxTokens magyarázat:
    //   1000 token ≈ 750 szó ≈ 1-2 bekezdés
    //   Elegendő részletes válaszokhoz, de nem túl hosszú
    //
    // Streaming működés:
    //   A streamText() nem várja meg a teljes választ, hanem
    //   azonnal elkezdi küldeni a token-eket ahogy generálódnak.
    //   Ezt Server-Sent Events (SSE) formátumban küldi a frontend-nek.
    //
    // onFinish callback:
    //   A streaming befejezése után automatikusan meghívódik.
    //   Itt mentjük el az assistant message-t és a RAG context-et.
    const result = await streamText({
      model: openai('gpt-4o-mini'), // KÖTELEZŐ: GPT-4o mini (követelmény szerint)
      messages: modelMessages,
      temperature: 0.7,              // Kiegyensúlyozott kreativitás
      maxTokens: 1000,               // Maximum generált token szám

      // OpenTelemetry experimental telemetry
      experimental_telemetry: {
        isEnabled: true,
        functionId: 'chat-completion',
        metadata: {
          session_id: session.id,
          chunks_used: chunks.length,
          has_context: hasContext,
        },
      },

      // onFinish callback: Assistant message és RAG context mentése
      async onFinish({ text, finishReason, usage }) {
        try {
          console.log('Streaming finished:', {
            textLength: text.length,
            finishReason,
            usage,
          });

          // ====================================================================
          // COST TRACKING
          // ====================================================================
          // Rögzítjük a chat completion költséget OpenTelemetry metrics-ben
          if (usage) {
            recordChatCompletionCost(
              usage.promptTokens,
              usage.completionTokens,
              { session_id: session.id }
            );
          }

          // ====================================================================
          // ASSISTANT MESSAGE MENTÉSE
          // ====================================================================
          // A teljes generált választ elmentjük a chat_messages táblába
          const savedAssistantMessage = await saveMessage({
            session_id: session.id,
            role: 'assistant',
            content: text,
            metadata: {
              model: 'gpt-4o-mini',
              temperature: 0.7,
              max_tokens: 1000,
              finish_reason: finishReason,
              usage: usage,  // Token használat (prompt_tokens, completion_tokens, total_tokens)
              has_rag_context: chunks.length > 0,
              rag_chunk_count: chunks.length,
              timestamp: new Date().toISOString(),
            },
          });

          console.log('Saved assistant message:', savedAssistantMessage.id);

          // ====================================================================
          // RAG CONTEXT MENTÉSE (BATCH)
          // ====================================================================
          // Csak ha van RAG context (chunks.length > 0)
          if (chunks.length > 0) {
            // RAG context DTO-k készítése minden chunk-hoz
            const ragContextDTOs: CreateChatRAGContextDTO[] = chunks.map(
              (chunk, index) => ({
                chat_message_id: savedAssistantMessage.id,
                chunk_id: chunk.id,
                embedding_similarity: chunk.metadata?.original_embedding_similarity,
                llm_relevance_score: chunk.metadata?.llm_relevance_score,
                blended_score: chunk.similarity,  // Final blended score
                rank_position: index + 1,         // 1-5
                reranking_blend_weight: 0.7,      // Fixed: 70% LLM + 30% embedding
                metadata: {
                  document_id: chunk.document_id,
                  chunk_index: chunk.chunk_index,
                  token_count: chunk.token_count,
                },
              })
            );

            // Batch INSERT
            const savedRAGContexts = await saveRAGContextBatch(ragContextDTOs);

            console.log(
              `Saved ${savedRAGContexts.length} RAG contexts for message: ${savedAssistantMessage.id}`
            );
          } else {
            console.log('No RAG context to save (no chunks retrieved)');
          }
        } catch (error) {
          // RAG context mentési hiba NEM blokkolja a streaming választ
          // Csak logoljuk a hibát
          console.error('Error saving assistant message or RAG context:', error);
        }
      },
    });

    // ========================================================================
    // 8. STREAMING RESPONSE VISSZAADÁSA
    // ========================================================================
    // result.toAIStreamResponse() konvertálja a streamText eredményt
    // egy Next.js Response objektummá, amit a frontend useChat hook
    // automatikusan feldolgoz.
    //
    // A response formátuma: Server-Sent Events (SSE) stream
    // Példa SSE üzenetek:
    //   data: {"type":"text","value":"Mowgli"}
    //   data: {"type":"text","value":" is"}
    //   data: {"type":"text","value":" a"}
    //   data: {"type":"text","value":" human"}
    //   ...
    //   data: [DONE]
    //
    // A useChat hook a frontend-en automatikusan parse-olja ezeket
    // és real-time frissíti a UI-t.
    return result.toAIStreamResponse();

  } catch (error) {
    // ========================================================================
    // 9. HIBAKEZELÉS
    // ========================================================================
    // Ha bármilyen hiba történik (DB error, OpenAI error, stb.),
    // akkor részletes hibaüzenetet logolunk és JSON error-t küldünk.
    console.error('Error in chat API route:', error);

    // JSON error response készítése
    return new Response(
      JSON.stringify({
        error: 'Failed to process chat request',
        message: error instanceof Error ? error.message : 'Unknown error',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
