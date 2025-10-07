# vizzy/laptop/__main__.py
# -----------------------------------------------------------------------------
# Entry point for the laptop side (StateManager-based orchestrator).
# Runs IDLE preview, SEARCH cycles (ScanWorker), and TaskAgent.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse

from .app import StateManager


def main():
    parser = argparse.ArgumentParser(description="Vizzy Laptop Orchestrator")
    parser.add_argument("--no-start-search", action="store_true",
                        help="Do not auto-start a search on launch (wait for idle timeout or user query).")
    args = parser.parse_args()

    sm = StateManager()

    # Optionally kick a search at startup by setting the trigger (keeps behavior flexible)
    if not args.no_start_search:
        sm.request_search("startup")
        pass

    sm.start()


if __name__ == "__main__":
    main()
