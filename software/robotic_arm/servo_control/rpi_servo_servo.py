#!/usr/bin/env python3
import json
import socket
import threading
import pigpio
from typing import Dict, Any

# -----------------------------
# USER SETTINGS (match your wiring)
# -----------------------------
SERVO_PINS = {
    "btm": 12,   # BCM GPIO
    "mid": 13,
    "top": 18,
}

# Pulse width bounds (microseconds)
SERVO_MIN_US    = 1000
SERVO_MAX_US    = 2000
SERVO_CENTER_US = 1500

# TCP server settings
HOST = "0.0.0.0"
PORT = 5005

# Optional shared token (set to None to disable)
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
    for name, gpio in SERVO_PINS.items():
        pi.set_servo_pulsewidth(gpio, SERVO_CENTER_US)
    return {"ok": True, "center_us": SERVO_CENTER_US}

def stop_all() -> Dict[str, Any]:
    for name, gpio in SERVO_PINS.items():
        pi.set_servo_pulsewidth(gpio, 0)  # 0 => servo off (no pulses)
    return {"ok": True}

def handle_cmd(cmd: Dict[str, Any]) -> Dict[str, Any]:
    if AUTH_TOKEN is not None and cmd.get("token") != AUTH_TOKEN:
        return {"ok": False, "err": "unauthorized"}

    op = cmd.get("cmd")
    if op == "set":
        # { "cmd":"set", "target":"btm|mid|top", "us":1500 }
        return set_servo(cmd.get("target", ""), int(cmd.get("us", SERVO_CENTER_US)))
    elif op == "center":
        return center_all()
    elif op == "stop":
        return stop_all()
    elif op == "get":
        # Return current pulsewidths
        state = {name: pi.get_servo_pulsewidth(g) for name, g in SERVO_PINS.items()}
        return {"ok": True, "state": state}
    else:
        return {"ok": False, "err": f"unknown cmd '{op}'"}

def client_thread(conn: socket.socket, addr):
    try:
        buf = b""
        # Protocol: newline-delimited JSON per command
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
    except Exception as e:
        # Basic safety: stop nothing on error; keep servos as-is
        pass
    finally:
        conn.close()

def main():
    # Power servos to center on boot
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
        # Optional: leave servos powered at last command; or stop:
        # stop_all()
        pass
