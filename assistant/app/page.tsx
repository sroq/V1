/**
 * 'use client' direktíva - Client Component jelölés
 *
 * Ez a sor KÖTELEZŐ minden Next.js komponensnél, amely:
 * - React hooks-okat használ (useState, useEffect, useRef, stb.)
 * - Browser API-kat használ (window, document, localStorage, stb.)
 * - Event handler-eket használ (onClick, onChange, stb.)
 * - Client-side interaktivitást igényel
 *
 * Next.js 14 alapértelmezés: Server Component
 * - Szerver oldalon renderelődik
 * - Nem használhat React hooks-okat
 * - SEO-barát, gyors initial load
 *
 * Client Component (amikor 'use client' van):
 * - Client oldalon renderelődik (böngészőben)
 * - Használhat React hooks-okat és browser API-kat
 * - Interaktív UI elemekhez kell
 *
 * Miért kell itt?
 * - useChat hook használata (Vercel AI SDK)
 * - useEffect és useRef használata
 * - Form submit handler
 * - Input change handler
 */
'use client';

/**
 * Chat Interface Component - Fő chat UI komponens
 *
 * Ez a komponens a RAG chat assistant teljes felhasználói felületét
 * implementálja, beleértve:
 * - Üzenet megjelenítés (user és assistant üzenetek)
 * - Streaming válaszok kezelése (real-time token-ek)
 * - Input form kezelés (kérdés beküldése)
 * - Loading és error state-ek
 * - Auto-scroll új üzenetekhez
 * - Dark mode támogatás
 *
 * Vercel AI SDK useChat hook:
 * - Automatikus streaming kezelés
 * - Message history management
 * - API kommunikáció (/api/chat)
 * - Loading és error state-ek
 * - Optimista UI update-ek
 *
 * UI felépítés:
 * 1. Header (cím, leírás)
 * 2. Messages container (chat üzenetek scrollable area)
 * 3. Loading indicator (ha éppen válaszol az AI)
 * 4. Error display (ha hiba történt)
 * 5. Input form (kérdés beküldéséhez)
 */

import { useChat } from 'ai/react';
import { useEffect, useRef } from 'react';

/**
 * ChatPage - Fő chat oldal komponens.
 *
 * Client Component, amely a Vercel AI SDK useChat hook-ját használja
 * a chat funkciók megvalósításához.
 *
 * useChat hook által biztosított funkciók:
 * - messages: ChatMessage[] - Üzenetek listája
 * - input: string - Input mező aktuális értéke
 * - handleInputChange: (e) => void - Input change handler
 * - handleSubmit: (e) => void - Form submit handler
 * - isLoading: boolean - Loading állapot (AI válaszol éppen)
 * - error: Error | undefined - Hiba objektum (ha volt hiba)
 *
 * ChatMessage típus:
 * {
 *   id: string,           // Egyedi üzenet ID
 *   role: 'user' | 'assistant' | 'system',
 *   content: string       // Üzenet szövege
 * }
 *
 * useChat működése:
 * 1. User beír egy kérdést → input state frissül
 * 2. User submit-olja a form-ot → handleSubmit()
 * 3. useChat elküldi POST /api/chat-re a messages-t
 * 4. Backend streaming választ küld (SSE)
 * 5. useChat real-time frissíti a messages tömböt
 * 6. UI automatikusan újra-renderelődik
 *
 * useRef használata:
 * - messagesEndRef: Referencia a messages container aljára
 * - Auto-scroll implementálásához kell
 * - Nem okoz re-render-t (ellentétben useState-tel)
 *
 * useEffect használata:
 * - Minden új üzenet után lefut
 * - Auto-scroll a messages container aljára
 * - Smooth scrolling animációval
 *
 * @returns JSX - Teljes chat UI
 */
export default function ChatPage() {
  // ========================================================================
  // VERCEL AI SDK USECHAT HOOK
  // ========================================================================
  // useChat hook inicializálása a chat funkciókhoz
  // Ez kezeli az összes chat logikát: API hívás, streaming, state management
  const { messages, input, handleInputChange, handleSubmit, isLoading, error } =
    useChat({
      api: '/api/chat',       // Backend API endpoint
      initialMessages: [],    // Kezdő üzenetek (üres lista = nincs history)
    });

  // ========================================================================
  // USEREF HOOK - AUTO-SCROLL REFERENCIA
  // ========================================================================
  // Referencia a messages container aljára
  // Ez egy "dummy" div, amit a messages lista végére teszünk
  // scrollIntoView() hívásával oda görgetünk
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // ========================================================================
  // USEEFFECT HOOK - AUTO-SCROLL ÚJ ÜZENETEKHEZ
  // ========================================================================
  // Függőség: messages
  // Amikor a messages tömb változik (új üzenet érkezik), lefut ez az effect
  //
  // Működés:
  // 1. User elküldi a kérdést → messages += user message
  // 2. useEffect lefut → scroll lefelé
  // 3. AI elkezd válaszolni (streaming) → messages += assistant message (folyamatosan frissül)
  // 4. Minden token érkezésekor useEffect lefut → scroll lefelé (smooth)
  //
  // scrollIntoView opciók:
  // - behavior: 'smooth' → Smooth scrolling animáció
  // - behavior: 'auto' → Instant scroll (nincs animáció)
  //
  // Optional chaining (?.):
  // - Ha messagesEndRef.current null (még nincs render-elve), nem hívja meg
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ========================================================================
  // JSX RETURN - UI RENDERELÉS
  // ========================================================================
  return (
    // Főkonténer: teljes képernyő magasság, flexbox column layout
    // Tailwind CSS class-ok:
    // - flex flex-col: Flexbox column (vertikális elrendezés)
    // - h-screen: 100vh (teljes viewport magasság)
    // - bg-gray-50 dark:bg-gray-900: Háttérszín (light/dark mode)
    <div className="flex flex-col h-screen bg-gray-50 dark:bg-gray-900">

      {/* ====================================================================
          HEADER - CÍM ÉS LEÍRÁS
          ==================================================================== */}
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4 shadow-sm">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          The Jungle Book - RAG Chat Assistant
        </h1>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Ask questions about The Jungle Book by Rudyard Kipling
        </p>
      </header>

      {/* ====================================================================
          MESSAGES CONTAINER - SCROLLABLE CHAT TERÜLET
          ==================================================================== */}
      {/*
        Tailwind CSS class-ok:
        - flex-1: Elfoglalja a rendelkezésre álló helyet (header és input között)
        - overflow-y-auto: Vertikális scrollbar ha túlcsordulás van
        - px-4 py-6: Padding (x-irány: 1rem, y-irány: 1.5rem)
        - space-y-4: Spacing a gyermek elemek között (1rem)
      */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">

        {/* ==================================================================
            WELCOME MESSAGE - HA NINCS MÉG ÜZENET
            ================================================================== */}
        {/* Conditional rendering: csak akkor jelenik meg, ha messages.length === 0 */}
        {messages.length === 0 && (
          <div className="text-center py-12">
            <div className="inline-block p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
                Welcome!
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                Start by asking a question about The Jungle Book.
              </p>
              {/* Példa kérdések lista */}
              <div className="mt-4 text-sm text-gray-500 dark:text-gray-500 space-y-1">
                <p>Example questions:</p>
                <ul className="list-disc list-inside space-y-1 text-left max-w-md mx-auto">
                  <li>Who is Mowgli?</li>
                  <li>What is the Law of the Jungle?</li>
                  <li>Tell me about Shere Khan</li>
                  <li>What happens in the story?</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================================
            ÜZENETEK RENDERELÉSE - USER ÉS ASSISTANT ÜZENETEK
            ================================================================== */}
        {/*
          messages.map() - Minden üzenetet render-elünk
          key={message.id} - React key prop (performance, update tracking)
        */}
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${
              // User üzenetek jobbra igazítva, assistant üzenetek balra
              message.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            {/* Üzenet buborék (message bubble) */}
            <div
              className={`max-w-3xl rounded-lg px-4 py-3 ${
                // User üzenetek kék háttérrel, assistant üzenetek fehér háttérrel
                message.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-700'
              }`}
            >
              {/* Flexbox: avatar + üzenet szöveg */}
              <div className="flex items-start gap-3">
                {/* Avatar (U = User, AI = Assistant) */}
                <div className="flex-shrink-0 mt-1">
                  {message.role === 'user' ? (
                    // User avatar: kék kör "U" betűvel
                    <div className="w-6 h-6 rounded-full bg-blue-700 flex items-center justify-center text-xs font-semibold">
                      U
                    </div>
                  ) : (
                    // AI avatar: zöld kör "AI" szöveggel
                    <div className="w-6 h-6 rounded-full bg-green-600 flex items-center justify-center text-xs font-semibold text-white">
                      AI
                    </div>
                  )}
                </div>
                {/* Üzenet szöveg */}
                {/*
                  Tailwind CSS class-ok:
                  - flex-1: Elfoglalja a rendelkezésre álló helyet
                  - whitespace-pre-wrap: Megtartja a whitespace-eket és sortöréseket
                  - break-words: Hosszú szavak törése (overflow megelőzése)
                */}
                <div className="flex-1 whitespace-pre-wrap break-words">
                  {message.content}
                </div>
              </div>
            </div>
          </div>
        ))}

        {/* ==================================================================
            LOADING INDICATOR - AMIKOR AZ AI VÁLASZOL
            ================================================================== */}
        {/* Conditional rendering: csak akkor jelenik meg, ha isLoading === true */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-3xl rounded-lg px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-3">
                {/* AI avatar */}
                <div className="w-6 h-6 rounded-full bg-green-600 flex items-center justify-center text-xs font-semibold text-white">
                  AI
                </div>
                {/* Animált "..." (három bouncing dot) */}
                {/*
                  animate-bounce: Tailwind CSS animáció (fel-le mozgás)
                  animationDelay: Időzített animáció (0s, 0.2s, 0.4s)
                  Ez létrehoz egy "hullám" effektust
                */}
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: '0.2s' }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: '0.4s' }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================================
            ERROR DISPLAY - HIBA MEGJELENÍTÉSE
            ================================================================== */}
        {/* Conditional rendering: csak akkor jelenik meg, ha error !== undefined */}
        {error && (
          <div className="flex justify-center">
            <div className="max-w-3xl rounded-lg px-4 py-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200">
              <p className="font-semibold">Error:</p>
              <p className="text-sm">{error.message}</p>
            </div>
          </div>
        )}

        {/* ==================================================================
            AUTO-SCROLL REFERENCIA - "DUMMY" DIV A LISTA VÉGÉN
            ================================================================== */}
        {/*
          Ez a div a messages lista végére kerül
          messagesEndRef.current erre a div-re mutat
          useEffect-ben scrollIntoView()-t hívunk rá
        */}
        <div ref={messagesEndRef} />
      </div>

      {/* ====================================================================
          INPUT FORM - KÉRDÉS BEKÜLDÉSÉHEZ
          ==================================================================== */}
      <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-4 py-4">
        {/*
          handleSubmit: Vercel AI SDK által biztosított submit handler
          Amikor a form submit-olódik:
          1. Hozzáadja a user üzenetet a messages-hez
          2. Elküldi POST /api/chat-re a messages tömböt
          3. Feldolgozza a streaming választ
          4. Frissíti a messages tömböt real-time
        */}
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex gap-2">
            {/* Input mező */}
            <input
              type="text"
              value={input}  // Vercel AI SDK által kezelt input state
              onChange={handleInputChange}  // Vercel AI SDK által biztosított handler
              placeholder="Ask a question about The Jungle Book..."
              disabled={isLoading}  // Letiltva loading közben
              className="flex-1 rounded-lg border border-gray-300 dark:border-gray-600 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
            />
            {/* Submit gomb */}
            <button
              type="submit"
              disabled={isLoading || !input.trim()}  // Letiltva ha loading vagy üres input
              className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {/* Dinamikus gomb szöveg: "Sending..." loading közben, egyébként "Send" */}
              {isLoading ? 'Sending...' : 'Send'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
