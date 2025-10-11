#!/usr/bin/env python3
"""
Raspberry Pi Servo Server — TCP JSON protocol on port 65432
Each command is a JSON line:
  {"cmd":"center"}
  {"cmd":"stop"}
  {"cmd":"get"}
  {"cmd":"set","target":"btm|mid|top","us":1500}
"""

import json, socket, threading, pigpio
from typing import Dict, Any

# ---- USER SETTINGS (BCM numbering) ----
SERVO_PINS = {
    "btm": 0,   # GPIO0
    "mid": 5,   # GPIO5
    "top": 6,   # GPIO6
}
SERVO_MIN_US, SERVO_MAX_US, SERVO_CENTER_US = 1000, 2000, 1500

HOST, PORT = "0.0.0.0", 65432
AUTH_TOKEN = None   # optional simple auth

# ---- pigpio setup ----
pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("pigpiod not running (sudo systemctl start pigpiod)")

def clamp_us(us: int) -> int:
    return max(SERVO_MIN_US, min(SERVO_MAX_US, int(us)))

def set_servo(name: str, us: int) -> Dict[str, Any]:
    if name not in SERVO_PINS:
        return {"ok": False, "err": f"bad target '{name}'"}
    us = clamp_us(us)
    pi.set_servo_pulsewidth(SERVO_PINS[name], us)
    return {"ok": True, "target": name, "us": us}

def center_all():
    for g in SERVO_PINS.values():
        pi.set_servo_pulsewidth(g, SERVO_CENTER_US)
    return {"ok": True, "center": SERVO_CENTER_US}

def stop_all():
    for g in SERVO_PINS.values():
        pi.set_servo_pulsewidth(g, 0)
    return {"ok": True}

def get_state():
    return {"ok": True,
            "state": {k: pi.get_servo_pulsewidth(v) for k, v in SERVO_PINS.items()}}

def handle(cmd: Dict[str, Any]) -> Dict[str, Any]:
    if AUTH_TOKEN and cmd.get("token") != AUTH_TOKEN:
        return {"ok": False, "err": "unauthorized"}
    op = cmd.get("cmd")
    if op == "center": return center_all()
    if op == "stop":   return stop_all()
    if op == "get":    return get_state()
    if op == "set":    return set_servo(cmd.get("target",""), int(cmd.get("us", SERVO_CENTER_US)))
    return {"ok": False, "err": f"unknown cmd '{op}'"}

def client(conn, _):
    buf = b""
    try:
        while (data := conn.recv(4096)):
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip(): continue
                try:
                    resp = handle(json.loads(line))
                except Exception as e:
                    resp = {"ok": False, "err": str(e)}
                conn.sendall((json.dumps(resp) + "\n").encode())
    finally:
        conn.close()

def main():
    center_all()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)
        print(f"Servo server listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            threading.Thread(target=client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    main()
