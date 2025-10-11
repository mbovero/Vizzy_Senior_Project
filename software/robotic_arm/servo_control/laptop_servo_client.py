# -----------------------------------------------------------------------------
# laptop_servo_client.py
"""
Tiny laptop-side client to drive the single-servo RPi server.
Updated to include your PI_IP and PI_PORT defaults.

Examples:
  # Set pulse width to 1400 µs
  python3 laptop_servo_client.py set 1400

  # Query current pulse width
  python3 laptop_servo_client.py get

  # Sweep between two values (inclusive) with a step and dwell (ms)
  python3 laptop_servo_client.py sweep 1100 1900 --step 50 --dwell 150
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time

# Networking defaults (mirrors server constants)
PI_IP   = "10.120.39.241"
PI_PORT = 65432


def send_json(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))


def recv_one(sock: socket.socket, *, timeout_s: float = 2.0) -> dict:
    sock.settimeout(timeout_s)
    buf = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError("connection closed")
        buf += chunk
        i = buf.find(b"\n")
        if i != -1:
            ln, buf = buf[:i], buf[i+1:]
            return json.loads(ln.decode("utf-8"))


def do_set(host: str, port: int, pwm: int) -> None:
    with socket.create_connection((host, port), timeout=2.0) as s:
        send_json(s, {"cmd": "SET_PWM", "pwm": int(pwm)})
        print(recv_one(s))


def do_get(host: str, port: int) -> None:
    with socket.create_connection((host, port), timeout=2.0) as s:
        send_json(s, {"cmd": "GET_PWM"})
        print(recv_one(s))


def do_sweep(host: str, port: int, lo: int, hi: int, step: int, dwell_ms: int) -> None:
    with socket.create_connection((host, port), timeout=2.0) as s:
        # upward
        for v in range(lo, hi + 1, step):
            send_json(s, {"cmd": "SET_PWM", "pwm": v})
            _ = recv_one(s)
            time.sleep(dwell_ms / 1000.0)
        # downward
        for v in range(hi, lo - 1, -step):
            send_json(s, {"cmd": "SET_PWM", "pwm": v})
            _ = recv_one(s)
            time.sleep(dwell_ms / 1000.0)
        print({"done": True})


def main() -> None:
    ap = argparse.ArgumentParser(description="Client for single-servo RPi server")
    ap.add_argument("--host", default=PI_IP)
    ap.add_argument("--port", type=int, default=PI_PORT)

    sub = ap.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser("set", help="Set servo pulse width (µs)")
    p_set.add_argument("pwm", type=int)

    _ = sub.add_parser("get", help="Read current pulse width")

    p_swp = sub.add_parser("sweep", help="Sweep between limits")
    p_swp.add_argument("lo", type=int)
    p_swp.add_argument("hi", type=int)
    p_swp.add_argument("--step", type=int, default=25)
    p_swp.add_argument("--dwell", type=int, default=100, help="ms between steps")

    args = ap.parse_args()

    if args.cmd == "set":
        do_set(args.host, args.port, args.pwm)
    elif args.cmd == "get":
        do_get(args.host, args.port)
    elif args.cmd == "sweep":
        if args.lo > args.hi:
            print("lo must be <= hi", file=sys.stderr)
            sys.exit(2)
        do_sweep(args.host, args.port, args.lo, args.hi, args.step, args.dwell)
    else:
        ap.error("unknown command")


if __name__ == "__main__":
    main()
