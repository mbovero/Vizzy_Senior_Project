#!/usr/bin/env python3
# laptop_cmd_client.py — ultra-thin sender: type "p1 p2 p3 p4" or "rest" or "quit"

import argparse, socket, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.120.39.241")
    ap.add_argument("--port", type=int, default=65432)
    args = ap.parse_args()

    print(f"[client] connecting to {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=5) as s:
        s.settimeout(2.0)
        print("[client] connected. Enter:")
        print("  <p1> <p2> <p3> <p4>   (radians, radians, radians, pwm-µs)")
        print("  rest")
        print("  quit")

        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                line = "quit"
            if not line:
                continue
            # Send the raw line; the server parses it
            s.sendall((line + "\n").encode("utf-8"))
            if line.lower() in ("quit", "q"):
                print("[client] done.")
                break
            # Try to read a short ACK (optional)
            try:
                data = s.recv(4096)
                if data:
                    print(data.decode("utf-8").strip())
            except Exception:
                pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
