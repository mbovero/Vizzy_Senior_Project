#!/usr/bin/env python3
# laptop_cmd_client.py — ultra-thin sender for updated IK protocol
# Protocol:
#   ik x y z pitch_deg yaw_deg O|C
#   rest
#   quit / q
#
# Examples:
#   0.20 0.00 0.35  15  -30  O
#   0.20, 0.00, 0.35, 15, -30, C

import argparse, socket, sys, re

def parse_ik_six(line: str):
    """
    Parse: x y z pitch_deg yaw_deg O|C
    Accepts comma or space separators; ignores extra whitespace.
    Returns (x,y,z,pitch,yaw,claw_char) or None on error.
    """
    norm = re.sub(r'[,\s]+', ' ', line.strip())
    toks = norm.split()
    if len(toks) != 6:
        return None
    *nums, claw = toks
    try:
        x, y, z, pitch, yaw = map(float, nums)
    except Exception:
        return None
    claw_ch = claw.upper()
    if claw_ch not in ("O", "C"):
        return None
    return x, y, z, pitch, yaw, claw_ch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.120.39.241")
    ap.add_argument("--port", type=int, default=65432)
    args = ap.parse_args()

    print(f"[client] connecting to {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=5) as s:
        s.settimeout(2.0)
        print("[client] connected. Enter one of:")
        print("  x y z pitch_deg yaw_deg O|C    e.g., 0.2 0.0 0.35 15 -30 O")
        print("                                   or   0.2, 0.0, 0.35, 15, -30, C")
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
                parsed = parse_ik_six(line)
                if parsed is None:
                    print("ERR: expected 'x y z pitch yaw O|C', or 'rest' / 'quit'.")
                    continue
                x, y, z, pitch, yaw, claw = parsed
                # Build message exactly as the server expects
                msg = f"ik {x} {y} {z} {pitch} {yaw} {claw}\n"
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
