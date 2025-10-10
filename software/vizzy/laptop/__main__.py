# vizzy/laptop/__main__.py
# -----------------------------------------------------------------------------
# Entry point for the laptop side (StateManager-based orchestrator).
# Runs IDLE preview, SEARCH cycles (ScanWorker), and TaskAgent.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys

from .app import StateManager
from ..shared import config as C


def show_startup_menu():
    """Display interactive startup menu for selecting operation mode."""
    print("\n" + "="*70)
    print("  VIZZY ROBOTIC ARM - STARTUP CONFIGURATION")
    print("="*70)
    print("\nSelect operation mode:\n")
    print("  1. Full System (Normal Operation)")
    print("     - Run scan cycle to detect objects")
    print("     - Perform LLM semantic enrichment")
    print("     - Accept user commands")
    print()
    print("  2. Skip Scan Cycle (Use Existing Memory)")
    print("     - Skip object detection/scanning")
    print("     - Perform LLM semantic enrichment on existing objects")
    print("     - Accept user commands")
    print()
    print("  3. Skip Semantic Enrichment (Scan Only)")
    print("     - Run scan cycle to detect objects")
    print("     - Skip LLM semantic enrichment")
    print("     - Accept user commands")
    print()
    print("  4. Task Scheduler Only (Testing Mode)")
    print("     - Skip scan cycle and semantic enrichment")
    print("     - Use existing object memory")
    print("     - Only test LLM task scheduler")
    print("     - Save scheduler output to file")
    print()
    print("  0. Exit")
    print()
    print("="*70)
    
    while True:
        try:
            choice = input("Enter your choice (0-4): ").strip()
            if choice in ['0', '1', '2', '3', '4']:
                return choice
            else:
                print("Invalid choice. Please enter 0, 1, 2, 3, or 4.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)


def configure_mode(choice: str):
    """Configure the system based on user's menu choice."""
    if choice == '0':
        print("Exiting...")
        sys.exit(0)
    
    elif choice == '1':
        # Full System (Normal Operation)
        C.SKIP_TO_TASK_SCHEDULER = False
        C.SKIP_SCAN_CYCLE = False
        C.SKIP_SEMANTIC_ENRICHMENT = False
        print("\n[Config] Mode: FULL SYSTEM")
        print("[Config] - Scan cycle: ENABLED")
        print("[Config] - Semantic enrichment: ENABLED")
        print("[Config] - Task scheduler: ENABLED")
    
    elif choice == '2':
        # Skip Scan Cycle
        C.SKIP_TO_TASK_SCHEDULER = False
        C.SKIP_SCAN_CYCLE = True
        C.SKIP_SEMANTIC_ENRICHMENT = False
        print("\n[Config] Mode: SKIP SCAN CYCLE")
        print("[Config] - Scan cycle: SKIPPED (using existing memory)")
        print("[Config] - Semantic enrichment: ENABLED")
        print("[Config] - Task scheduler: ENABLED")
    
    elif choice == '3':
        # Skip Semantic Enrichment
        C.SKIP_TO_TASK_SCHEDULER = False
        C.SKIP_SCAN_CYCLE = False
        C.SKIP_SEMANTIC_ENRICHMENT = True
        print("\n[Config] Mode: SKIP SEMANTIC ENRICHMENT")
        print("[Config] - Scan cycle: ENABLED")
        print("[Config] - Semantic enrichment: SKIPPED")
        print("[Config] - Task scheduler: ENABLED")
    
    elif choice == '4':
        # Task Scheduler Only
        C.SKIP_TO_TASK_SCHEDULER = True
        C.SKIP_SCAN_CYCLE = True
        C.SKIP_SEMANTIC_ENRICHMENT = True
        print("\n[Config] Mode: TASK SCHEDULER ONLY (Testing)")
        print("[Config] - Scan cycle: SKIPPED")
        print("[Config] - Semantic enrichment: SKIPPED")
        print("[Config] - Task scheduler: ENABLED")
        print(f"[Config] - Output file: {C.TASK_SCHEDULER_OUTPUT_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Vizzy Laptop Orchestrator")
    parser.add_argument("--no-start-search", action="store_true",
                        help="Do not auto-start a search on launch (wait for idle timeout or user query).")
    parser.add_argument("--skip-menu", action="store_true",
                        help="Skip interactive menu and use config file settings.")
    args = parser.parse_args()

    # Show interactive menu unless skipped
    if not args.skip_menu:
        choice = show_startup_menu()
        configure_mode(choice)
    else:
        print("\n[Config] Using settings from config.py (menu skipped)")

    print("\nInitializing system...")
    sm = StateManager()

    # Optionally kick a search at startup by setting the trigger (keeps behavior flexible)
    if not args.no_start_search:
        sm.request_search("startup")

    sm.start()


if __name__ == "__main__":
    main()
