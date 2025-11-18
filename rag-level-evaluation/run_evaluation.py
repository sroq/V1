#!/usr/bin/env python3
"""
Fő pipeline script a teljes RAG értékelés futtatásához.

Ez a script orchestrálja (összefogja) a teljes értékelési folyamatot:
1. Kérdések generálása chunk-okból (generate_questions.py)
2. RAG retrieval értékelés (evaluate_rag.py)
3. Eredmények elemzése és vizualizációk (analyze_results.py)

Ez a LEGEGYSZERŰBB módja az értékelési rendszer futtatásának!

Használat:
    python3 run_evaluation.py                    # Teljes pipeline
    python3 run_evaluation.py --skip-generation  # Meglévő kérdések használata
    python3 run_evaluation.py --regenerate       # Kérdések újragenerálása
    python3 run_evaluation.py --skip-generation --skip-evaluation  # Csak elemzés

Opciók:
    --skip-generation: Kihagyja a kérdésgenerálást (meglévő kérdéseket használ)
    --skip-evaluation: Kihagyja az értékelést (meglévő eredményeket használ)
    --regenerate: Kényszeríti a kérdések újragenerálását (felülírja a meglévőket)

Példák:
    # Teljes pipeline futtatása (első futtatás)
    python3 run_evaluation.py

    # Értékelés újrafuttatása új threshold-dal (kérdések megmaradnak)
    python3 run_evaluation.py --skip-generation

    # Csak az elemzést futtatjuk újra (pl. más vizualizációk)
    python3 run_evaluation.py --skip-generation --skip-evaluation

    # Minden újragenerálása
    python3 run_evaluation.py --regenerate
"""
import os
import sys
import argparse
from config import QUESTIONS_FILE, EVALUATION_RESULTS_FILE


def run_command(script_name: str, description: str) -> bool:
    """
    Python script futtatása és sikeres/sikertelen visszajelzés.

    Ez a függvény meghívja a megadott Python scriptet és figyeli
    a visszatérési kódját (exit code). Ha 0, sikeres volt, ha nem 0, hiba történt.

    Args:
        script_name: A futtatandó script neve (pl. "generate_questions.py")
        description: A lépés leírása (megjelenik a banner-ben)

    Returns:
        True ha sikeres volt, False ha hiba történt

    Működés:
        os.system() meghívja a script-et alshell-ben
        Exit code: 0 = siker, !=0 = hiba
    """
    # Banner a lépés kezdetekor
    print("\n" + "=" * 80)
    print(f"STEP: {description}")
    print("=" * 80)

    # Python3 parancs összeállítása
    cmd = f"python3 {script_name}"

    # Script futtatása és exit code elkapása
    exit_code = os.system(cmd)

    # Exit code ellenőrzése
    if exit_code != 0:
        print(f"\n✗ Error: {script_name} failed with exit code {exit_code}")
        return False

    print(f"\n✓ {description} completed successfully")
    return True


def main():
    """Fő pipeline orchestration függvény."""

    # ========================================================================
    # PARANCSSORI ARGUMENTUMOK FELDOLGOZÁSA
    # ========================================================================
    # argparse: Python könyvtár CLI argumentumok feldolgozásához
    parser = argparse.ArgumentParser(
        description='RAG értékelési pipeline futtatása',
        formatter_class=argparse.RawDescriptionHelpFormatter,  # Megtartja a formázást
        epilog="""
Példák:
  # Teljes értékelési pipeline
  python3 run_evaluation.py

  # Meglévő kérdések használata (újragenerálás nélkül)
  python3 run_evaluation.py --skip-generation

  # Kérdések kényszerített újragenerálása
  python3 run_evaluation.py --regenerate

  # Csak meglévő eredmények elemzése
  python3 run_evaluation.py --skip-generation --skip-evaluation
        """
    )

    # Opciók definiálása
    parser.add_argument(
        '--skip-generation',
        action='store_true',  # Boolean flag (nincs érték, True ha megadják)
        help='Kérdésgenerálás kihagyása (meglévő kérdések használata)'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Értékelés kihagyása (meglévő eredmények használata)'
    )
    parser.add_argument(
        '--regenerate',
        action='store_true',
        help='Kérdések kényszerített újragenerálása (felülírja a meglévőket)'
    )

    # Argumentumok parse-olása (feldolgozása)
    args = parser.parse_args()

    # ========================================================================
    # PIPELINE KEZDŐ BANNER
    # ========================================================================
    print("=" * 80)
    print("RAG EVALUATION PIPELINE")
    print("=" * 80)
    print()
    print("Ez a pipeline a következő lépéseket hajtja végre:")
    print("  1. Kérdések generálása dokumentum chunk-okból")
    print("  2. RAG retrieval teljesítmény értékelése")
    print("  3. Eredmények elemzése és vizualizálása")
    print()

    # ========================================================================
    # 1. LÉPÉS: KÉRDÉSGENERÁLÁS
    # ========================================================================
    # Döntés: kihagyjuk vagy futtatjuk?
    skip_generation = args.skip_generation and not args.regenerate
    questions_exist = os.path.exists(QUESTIONS_FILE)

    if skip_generation and questions_exist:
        # Kihagyjuk, mert a felhasználó kérte és léteznek kérdések
        print(f"⏭  Skipping question generation (using existing file: {QUESTIONS_FILE})")

    elif not skip_generation or args.regenerate:
        # Futtatjuk a kérdésgenerálást
        if questions_exist and not args.regenerate:
            # Van már kérdés fájl, de nem kértük explicit az újragenerálást
            # Kérdezzük meg a felhasználót
            response = input(f"\n⚠ Questions file already exists: {QUESTIONS_FILE}\n  Regenerate? (y/N): ")
            if response.lower() != 'y':
                print("  Using existing questions file.")
            else:
                # Felhasználó igent mondott → újrageneráljuk
                if not run_command('generate_questions.py', 'Question Generation'):
                    sys.exit(1)  # Hiba történt, kilépünk
        else:
            # Nincs kérdés fájl VAGY kértük az újragenerálást
            if not run_command('generate_questions.py', 'Question Generation'):
                sys.exit(1)

    elif not questions_exist:
        # Kihagytuk volna, de nincs kérdés fájl → hiba!
        print(f"\n✗ Error: Questions file not found: {QUESTIONS_FILE}")
        print("  Cannot skip generation when questions don't exist.")
        sys.exit(1)

    # ========================================================================
    # 2. LÉPÉS: RAG ÉRTÉKELÉS
    # ========================================================================
    skip_evaluation = args.skip_evaluation
    results_exist = os.path.exists(EVALUATION_RESULTS_FILE)

    if skip_evaluation and results_exist:
        # Kihagyjuk az értékelést
        print(f"\n⏭  Skipping evaluation (using existing file: {EVALUATION_RESULTS_FILE})")

    elif not skip_evaluation:
        # Futtatjuk az értékelést
        if results_exist:
            # Van már eredmény fájl → kérdezzük meg
            response = input(f"\n⚠ Evaluation results file already exists: {EVALUATION_RESULTS_FILE}\n  Re-evaluate? (y/N): ")
            if response.lower() != 'y':
                print("  Using existing evaluation results.")
            else:
                # Újraértékelés
                if not run_command('evaluate_rag.py', 'RAG Retrieval Evaluation'):
                    sys.exit(1)
        else:
            # Nincs eredmény fájl → futtatjuk
            if not run_command('evaluate_rag.py', 'RAG Retrieval Evaluation'):
                sys.exit(1)

    elif not results_exist:
        # Kihagytuk volna, de nincs eredmény fájl → hiba!
        print(f"\n✗ Error: Evaluation results file not found: {EVALUATION_RESULTS_FILE}")
        print("  Cannot skip evaluation when results don't exist.")
        sys.exit(1)

    # ========================================================================
    # 3. LÉPÉS: EREDMÉNYEK ELEMZÉSE
    # ========================================================================
    # Az elemzést MINDIG futtatjuk (kivéve ha a korábbi lépések hibáztak)
    if not run_command('analyze_results.py', 'Results Analysis and Visualization'):
        sys.exit(1)

    # ========================================================================
    # PIPELINE SIKERES BEFEJEZÉSE
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Minden értékelési lépés sikeresen lefutott!")
    print(f"  Kérdések: {QUESTIONS_FILE}")
    print(f"  Eredmények: {EVALUATION_RESULTS_FILE}")
    print(f"  Elemzés: Nézd meg a legújabb timestamped mappát a results/ könyvtárban")
    print()


# ============================================================================
# SCRIPT BELÉPÉSI PONT
# ============================================================================
if __name__ == '__main__':
    main()
