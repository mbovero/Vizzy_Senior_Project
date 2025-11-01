#!/usr/bin/env python3
# laptop_cmd_client.py — ultra-thin sender for IK protocol
# Input:
#   <x> <y> <z> <wrist>      (meters, meters, meters, radians-or-your-orient)
#   rest
#   quit / q

import argparse, socket, sys

def parse_four_numbers(line: str):
    # Allow commas or spaces; ignore extra whitespace
    norm = line.replace(",", " ").strip()
    toks = [t for t in norm.split() if t]
    if len(toks) != 4:
        return None
    try:
        return list(map(float, toks))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.120.40.160")
    ap.add_argument("--port", type=int, default=65432)
    args = ap.parse_args()

    print(f"[client] connecting to {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=5) as s:
        s.settimeout(2.0)
        print("[client] connected. Enter:")
        print("  <x> <y> <z> <wrist>   (e.g., 0.2 0.0 0.35 0.0  or  0.2, 0.0, 0.35, 0.0)")
        print("  rest")
        print("  quit")

        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                line = "quit"
            if not line:
                continue

            low = line.lower()
            if low in ("quit", "q"):
                s.sendall(b"quit\n")
                print("[client] done.")
                break
            if low == "rest":
                s.sendall(b"rest\n")
            else:
                vals = parse_four_numbers(line)
                if vals is None:
                    print("ERR: expected four numbers for x y z wrist, or 'rest' / 'quit'.")
                    continue
                x, y, z, wrist = vals
                msg = f"ik {x} {y} {z} {wrist}\n"
                s.sendall(msg.encode("utf-8"))

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
