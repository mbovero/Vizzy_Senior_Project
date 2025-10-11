#!/usr/bin/env python3
"""
Raspberry Pi Servo Server (TCP JSON, newline-delimited)
- Start pigpio daemon first: sudo systemctl enable --now pigpiod
- Run: python3 rpi_servo_server.py
- Default port: 5005
Commands (JSON per line):
  {"cmd":"center"}
  {"cmd":"stop"}
  {"cmd":"get"}
  {"cmd":"set","target":"btm|mid|top","us":1500}
Optional auth: set AUTH_TOKEN below and include {"token":"..."} in every command.
"""

import json
import socket
import threading
import pigpio
from typing import Dict, Any

# -----------------------------
# USER SETTINGS — edit to match your wiring
# -----------------------------
SERVO_PINS = {
    "btm": 0,   # BCM GPIO number
    "mid": 5,
    "top": 6,
}

# Microsecond pulse bounds (typical hobby servo: 1000–2000, center 1500)
SERVO_MIN_US    = 1000
SERVO_MAX_US    = 2000
SERVO_CENTER_US = 1500

# TCP server settings
HOST = "0.0.0.0"
PORT = 5005

# Optional simple auth token; set to None to disable
AUTH_TOKEN = None  # e.g., "secret123"

# -----------------------------
# Init pigpio
# -----------------------------
pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("pigpio daemon not running; try: sudo systemctl start pigpiod")

def clamp_us(us: int) -> int:
    return max(SERVO_MIN_US, min(SERVO_MAX_US, int(us)))

def set_servo(target: str, us: int) -> Dict[str, Any]:
    if target not in SERVO_PINS:
        return {"ok": False, "err": f"unknown target '{target}'"}
    us = clamp_us(us)
    pi.set_servo_pulsewidth(SERVO_PINS[target], us)
    return {"ok": True, "target": target, "us": us}

def center_all() -> Dict[str, Any]:
    for gpio in SERVO_PINS.values():
        pi.set_servo_pulsewidth(gpio, SERVO_CENTER_US)
    return {"ok": True, "center_us": SERVO_CENTER_US}

def stop_all() -> Dict[str, Any]:
    for gpio in SERVO_PINS.values():
        pi.set_servo_pulsewidth(gpio, 0)  # 0 = pulses off (servo unpowered)
    return {"ok": True}

def get_state() -> Dict[str, Any]:
    state = {name: pi.get_servo_pulsewidth(g) for name, g in SERVO_PINS.items()}
    return {"ok": True, "state": state}

def handle_cmd(cmd: Dict[str, Any]) -> Dict[str, Any]:
    if AUTH_TOKEN is not None and cmd.get("token") != AUTH_TOKEN:
        return {"ok": False, "err": "unauthorized"}
    op = cmd.get("cmd")
    if op == "center":
        return center_all()
    elif op == "stop":
        return stop_all()
    elif op == "get":
        return get_state()
    elif op == "set":
        return set_servo(cmd.get("target", ""), int(cmd.get("us", SERVO_CENTER_US)))
    else:
        return {"ok": False, "err": f"unknown cmd '{op}'"}

def client_thread(conn: socket.socket, addr):
    try:
        buf = b""
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    cmd = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    conn.sendall(b'{"ok": false, "err": "bad json"}\n')
                    continue
                resp = handle_cmd(cmd)
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
    finally:
        conn.close()

def main():
    # Center servos on startup for safety
    center_all()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)
        print(f"Servo server listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            t = threading.Thread(target=client_thread, args=(conn, addr), daemon=True)
            t.start()

if __name__ == "__main__":
    try:
        main()
    finally:
        # Optionally stop all pulses on exit:
        # stop_all()
        pass
