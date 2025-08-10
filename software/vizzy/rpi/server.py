# vizzy/rpi/server.py
from __future__ import annotations
import socket, select, pigpio
from ..shared.jsonl import recv_lines
from .servo import setup_servos
from .search import run_search_cycle
from .config import LISTEN_HOST, LISTEN_PORT, SERVO_BTM, SERVO_TOP
from . import state
from .dispatch import process_messages  # <-- NEW import

def handle_client(conn, pi, debug: bool) -> None:
    conn.setblocking(False)
    buf = b""
    try:
        while True:
            readable, _, _ = select.select([conn], [], [], 0.2)
            if not readable:
                continue

            msgs, buf, closed = recv_lines(conn, buf)
            if closed:
                break

            stop, _, _, _, _ = process_messages(pi, msgs, debug=debug)
            if stop:
                return

            if state.search_active.is_set():
                run_search_cycle(pi, conn, debug=debug)
                setup_servos(pi)

    finally:
        conn.close()
        state.search_active.clear()
        state.centering_active.clear()

def main(debug: bool) -> None:
    pi = pigpio.pi()
    if not pi.connected:
        print("Failed to connect to pigpio daemon")
        return

    setup_servos(pi)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((LISTEN_HOST, LISTEN_PORT))
        s.listen()
        print(f"RPi server started (debug={'on' if debug else 'off'}), waiting for connections...")

        try:
            while True:
                conn, addr = s.accept()
                print(f"Connected by {addr}")
                try:
                    handle_client(conn, pi, debug=debug)
                except Exception as e:
                    print(f"[RPi] Client handler error: {e}")
        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            # Clean up
            pi.set_servo_pulsewidth(SERVO_BTM, 0)
            pi.set_servo_pulsewidth(SERVO_TOP, 0)
            pi.stop()
