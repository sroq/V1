/**
 * Root Layout - Next.js 14 App Router gyökér layout komponens
 *
 * Ez a fájl a TELJES alkalmazás layout-ját definiálja.
 * Minden oldal és route ezen a layout-on keresztül renderelődik.
 *
 * Next.js App Router layout koncepciója:
 * - A layout komponensek közös UI elemeket definiálnak (header, footer, stb.)
 * - A layout-ok beágyazhatók (nested layouts)
 * - A root layout KÖTELEZŐ és EGYETLEN a teljes appra
 * - A layout state-je megmarad route váltásnál (nem renderelődik újra)
 *
 * Ez a root layout tartalmazza:
 * - HTML és body tag-eket
 * - Metadata-t (title, description)
 * - Globális stílusokat (globals.css)
 * - Language beállítást (lang="hu")
 *
 * Fontos: Ez a fájl Server Component (alapértelmezett Next.js 14-ben)
 * - Nem használ useState, useEffect, stb.
 * - Szerver oldalon renderelődik
 * - SEO-barát (teljes HTML a szerveren generálódik)
 */

import type { Metadata } from 'next';
import './globals.css';

/**
 * Metadata objektum az oldal SEO és böngésző címsor beállításaihoz.
 *
 * A Next.js Metadata API lehetővé teszi az oldal meta információinak
 * definiálását TypeScript-ben, típusbiztonság garantált módon.
 *
 * Támogatott metadata mezők:
 * - title: Böngésző címsor és Google találati lista cím
 * - description: Meta description (Google snippet)
 * - keywords: Keresőszavak (bár manapság kevésbé fontos)
 * - openGraph: Facebook/LinkedIn preview
 * - twitter: Twitter card preview
 * - robots: Keresőmotor indexelési utasítások
 * - viewport: Mobil viewport beállítások (alapértelmezett: responsive)
 *
 * SEO fontosság:
 * - Title: NAGYON fontos (Google ranking faktor)
 * - Description: Fontos (nem ranking faktor, de CTR-t befolyásolja)
 * - Karakterkorlát: Title ~60 karakter, Description ~160 karakter
 *
 * Példa renderelt HTML:
 *   <head>
 *     <title>RAG Chat Assistant - The Jungle Book</title>
 *     <meta name="description" content="AI-powered chat assistant..." />
 *   </head>
 */
export const metadata: Metadata = {
  title: 'RAG Chat Assistant - The Jungle Book',
  description: 'AI-powered chat assistant with RAG capabilities for The Jungle Book',
};

/**
 * RootLayout - Az alkalmazás gyökér layout komponense.
 *
 * Ez a komponens MINDEN oldalt körülvesz az alkalmazásban.
 * Server Component, nem használ React hooks-okat.
 *
 * @param children - A beágyazott oldal tartalma (page.tsx, stb.)
 * @returns JSX - Teljes HTML struktúra <html> tag-től kezdve
 *
 * Next.js App Router layout működése:
 *
 * 1. File-based routing:
 *    app/
 *    ├── layout.tsx        ← Ez a fájl (root layout)
 *    └── page.tsx          ← Home page (children prop-ként jelenik meg)
 *
 * 2. Nested layouts példa:
 *    app/
 *    ├── layout.tsx        ← Root layout (HTML, body)
 *    └── dashboard/
 *        ├── layout.tsx    ← Dashboard layout (sidebar, header)
 *        └── page.tsx      ← Dashboard home
 *
 * 3. Layout state persistence:
 *    - Route váltásnál a layout NEM renderelődik újra
 *    - Csak a children (page) renderelődik újra
 *    - Jó performance (kevesebb re-render)
 *
 * Props magyarázat:
 * - children: React.ReactNode
 *   - A beágyazott tartalom (általában page.tsx)
 *   - Next.js automatikusan injektálja a megfelelő page-et
 *
 * HTML struktúra magyarázat:
 *
 * <html lang="hu">
 *   - lang="hu": Magyar nyelvű tartalom jelzése
 *   - Fontos accessibility szempontból (screen reader-ek)
 *   - SEO szempontból is releváns (nyelv detektálás)
 *
 * <body className="antialiased">
 *   - antialiased: Tailwind CSS utility class
 *   - Simább font renderelés (anti-aliasing bekapcsolása)
 *   - Vizuálisan szebb szövegek
 *
 * globals.css import:
 *   - Tailwind CSS base, components, utilities
 *   - Custom global stílusok
 *   - Font definíciók
 *   - CSS reset/normalizálás
 *
 * Readonly típus:
 *   - TypeScript utility type
 *   - Jelzi, hogy a props nem módosíthatók
 *   - Jó practice React komponenseknél
 *
 * Példa használat:
 *   User meglátogatja: http://localhost:3000/
 *   Next.js rendereli:
 *     1. RootLayout component
 *     2. page.tsx beinjektálva a {children} prop-ba
 *     3. Teljes HTML generálása szerveroldalon
 *     4. Küldés a böngészőnek
 */
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="hu">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
