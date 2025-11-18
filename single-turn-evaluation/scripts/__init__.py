"""
Értékelési Pipeline Szkriptek.

Ez a csomag a teljes single-turn értékelési pipeline-t tartalmazza:
1. generate_golden_dataset.py - Golden Q&A dataset létrehozása
2. run_assistant.py - RAG asszisztens futtatása minden kérdésen
3. evaluate_correctness.py - Helyesség értékelése (LLM judge)
4. evaluate_relevance.py - Relevancia értékelése (LLM judge)
5. analyze_results.py - Metrikák aggregálása és riport generálás
"""
