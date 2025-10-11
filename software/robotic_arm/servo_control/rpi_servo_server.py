# rpi_servo_server.py
"""
Minimal JSONL TCP server for controlling a single hobby servo on Raspberry Pi.
Now includes your requested constants, networking defaults, and an init_servos()
that powers all three servo outputs to center on startup.

- JSONL commands (all messages end with "
"):
    {"cmd": "SET_PWM", "pwm": 1500}          -> set pulse width (µs) on the primary servo
    {"cmd": "GET_PWM"}                         -> reply with current pulse width
    {"cmd": "STOP"}                            -> stop server (optional)

Responses:
    {"type": "ACK", "ok": true, "pwm": <int>}
    {"type": "PWM", "pwm": <int>}
    {"type": "ERR", "error": "..."}

Run on the Pi (requires pigpio daemon running):
    sudo systemctl enable pigpiod --now
    python3 rpi_servo_server.py --port 65432
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass

import pigpio

# ----------------------------- Constants ---------------------------------
# Servos & Sweep (RPi)
# NOTE: Using your requested GPIO pins for the three servos.
class C:
    # Servo GPIO pins (BCM numbering for pigpio)
    # @Burke use GPIO pins 0, 5, and 6 !!! these would be in place SERVO_BTM, ...
    SERVO_BTM = 0   # was 22
    SERVO_MID = 5   # was 27
    SERVO_TOP = 6   # was 17

    # Pulse width bounds (µs)
    SERVO_MIN    = 600
    SERVO_MAX    = 2500
    SERVO_CENTER = 1500

    # Networking (Laptop connects to the Pi at this host/port)
    PI_IP   = "10.120.39.241"
    PI_PORT = 65432


# Lightweight state holder
class _State:
    def __init__(self) -> None:
        self.current_horizontal = C.SERVO_CENTER
        self.current_vertical = C.SERVO_CENTER
        self.current_pwm = C.SERVO_CENTER

state = _State()


# pigpio instance and servo init
pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("pigpio daemon not running. Start with: sudo systemctl start pigpiod")


def init_servos(pi: pigpio.pi) -> None:
    """
    Initialize servo outputs at SERVO_CENTER so they are powered and not limp.
    Safe to call multiple times.
    """
    # pi.set_servo_pulsewidth(C.SERVO_BTM, C.SERVO_CENTER)
    # pi.set_servo_pulsewidth(C.SERVO_MID, C.SERVO_CENTER)
    pi.set_servo_pulsewidth(C.SERVO_TOP, C.SERVO_CENTER)
    state.current_horizontal = C.SERVO_CENTER
    state.current_vertical = C.SERVO_CENTER
    state.current_pwm = C.SERVO_CENTER


# Call once at import to bring all servos to life
init_servos(pi)


# ---------------------------- JSONL helpers -----------------------------

def send_json(sock: socket.socket, obj: dict) -> None:
    data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
    sock.sendall(data)


def recv_lines(sock: socket.socket, buf: bytes):
    """Return (messages, remaining_buffer, closed). Robust to partial reads."""
    closed = False
    try:
        chunk = sock.recv(4096)
        if not chunk:
            return [], buf, True
        buf += chunk
    except BlockingIOError:
        pass
    except ConnectionResetError:
        return [], buf, True

    msgs = []
    while True:
        i = buf.find(b"\n")
        if i == -1:
            break  # no full line yet
        line = buf[:i]
        buf = buf[i+1:]
        if not line:
            continue  # skip empty lines
        try:
            # Decode strictly; if it fails, report an error frame
            obj = json.loads(line.decode("utf-8"))
            msgs.append(obj)
        except Exception as e:
            msgs.append({"type": "ERR", "error": f"bad json ({e})", "raw": line.decode("utf-8", "replace")})
    return msgs, buf, closed


# ----------------------------- Config -----------------------------------

@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = C.PI_PORT
    primary_servo_pin: int = C.SERVO_TOP   # choose which single servo to drive via SET_PWM
    servo_min: int = C.SERVO_MIN
    servo_max: int = C.SERVO_MAX
    settle_s: float = 0.02


# ---------------------------- Server logic ------------------------------

class ServoServer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.current_pwm = C.SERVO_CENTER
        # Ensure servos are initialized (already done at import), but safe to repeat
        init_servos(pi)

    def _clamp(self, v: int) -> int:
        return max(self.cfg.servo_min, min(self.cfg.servo_max, int(v)))

    def _apply(self, pwm: int) -> None:
        pwm = self._clamp(pwm)
        pi.set_servo_pulsewidth(self.cfg.primary_servo_pin, pwm)
        self.current_pwm = pwm
        state.current_pwm = pwm

    def _handle(self, msg: dict, conn: socket.socket) -> bool:
        """Return True to request server stop."""
        if msg.get("type") == "ERR" and msg.get("error") == "bad json":
            send_json(conn, msg)
            return False
        cmd = msg.get("cmd")
        if cmd == "SET_PWM":
            try:
                pwm = int(msg.get("pwm"))
            except Exception:
                send_json(conn, {"type": "ERR", "error": "SET_PWM requires integer 'pwm'"})
                return False
            self._apply(pwm)
            time.sleep(self.cfg.settle_s)
            send_json(conn, {"type": "ACK", "ok": True, "pwm": self.current_pwm})
        elif cmd == "GET_PWM":
            send_json(conn, {"type": "PWM", "pwm": self.current_pwm})
        elif cmd == "STOP":
            send_json(conn, {"type": "ACK", "ok": True, "stopping": True})
            return True
        else:
            send_json(conn, {"type": "ERR", "error": f"unknown cmd: {cmd}"})
        return False

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.cfg.host, self.cfg.port))
            s.listen(1)
            print(f"[RPi] Listening on {self.cfg.host}:{self.cfg.port} (primary_servo_pin={self.cfg.primary_servo_pin})")
            try:
                while True:
                    print("[RPi] Waiting for laptop client…")
                    conn, addr = s.accept()
                    print(f"[RPi] Client connected: {addr[0]}:{addr[1]}")
                    init_servos(pi)
                    buf = b""
                    try:
                        with conn:
                            conn.setblocking(False)
                            while True:
                                msgs, buf, closed = recv_lines(conn, buf)
                                if closed:
                                    print("[RPi] Client disconnected.")
                                    break
                                for m in msgs:
                                    stop = self._handle(m, conn)
                                    if stop:
                                        print("[RPi] STOP requested; shutting down.")
                                        return
                                time.sleep(0.01)
                    except Exception as e:
                        print(f"[RPi] client loop error: {e}")
                        try:
                            conn.close()
                        except Exception:
                            pass
            finally:
                try:
                    init_servos(pi)
                except Exception:
                    pass
                pi.stop()
                print("[RPi] Server shut down.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-servo TCP server (JSONL)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=C.PI_PORT)
    ap.add_argument("--pin", dest="primary_servo_pin", type=int, default=C.SERVO_TOP, help="BCM pin for primary servo signal")
    ap.add_argument("--min", dest="servo_min", type=int, default=C.SERVO_MIN)
    ap.add_argument("--max", dest="servo_max", type=int, default=C.SERVO_MAX)
    ap.add_argument("--settle", dest="settle_s", type=float, default=0.02)
    args = ap.parse_args()
    cfg = Config(host=args.host, port=args.port, primary_servo_pin=args.primary_servo_pin,
                 servo_min=args.servo_min, servo_max=args.servo_max, settle_s=args.settle_s)
    ServoServer(cfg).serve_forever()


if __name__ == "__main__":
    main()


