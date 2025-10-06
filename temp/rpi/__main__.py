# vizzy/rpi/__main__.py
# -----------------------------------------------------------------------------
# Entry point for the Raspberry Pi side.
# Launches the single-client JSONL TCP server and routes messages
# using the new laptop-centric protocol.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse

from .server import serve_forever


def main():
    parser = argparse.ArgumentParser(description="Vizzy RPi server")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logs on RPi")
    args = parser.parse_args()

    serve_forever(debug=args.debug)


if __name__ == "__main__":
    main()
