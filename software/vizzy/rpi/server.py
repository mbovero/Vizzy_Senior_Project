# vizzy/rpi/server.py
from __future__ import annotations
import socket, select, pigpio
from typing import Tuple, Optional
from ..shared.jsonl import send_json, recv_lines
from ..shared import protocol as P
from . import state
from .servo import setup_servos, move_servos, goto_pwms
from .search import run_search_cycle
from .config import LISTEN_HOST, LISTEN_PORT

def process_messages(pi, msgs, debug: bool
                     ) -> Tuple[bool, Optional[int], Optional[bool], Optional[dict], Optional[dict]]:
    """
    Unified message handler.
    Returns: (stop_requested, center_done_cls, center_success, center_diag, yolo_results)
    """
    stop = False
    center_done_cls: Optional[int] = None
    center_success: Optional[bool] = None
    center_diag: Optional[dict] = None
    yolo_results: Optional[dict] = None

    for msg in msgs:
        mtype = msg.get("type")
        mcmd  = msg.get("cmd")

        # Laptop-driven incremental motion
        if mtype == P.TYPE_MOVE:
            if state.centering_active.is_set() or not state.search_active.is_set():
                move_servos(pi, msg.get("horizontal", 0.0), msg.get("vertical", 0.0))

        # Search toggle
        elif mtype == P.TYPE_SEARCH:
            if msg.get("active", False):
                if not state.search_active.is_set():
                    print("[RPi] Search ON")
                state.search_active.set()
            else:
                if state.search_active.is_set():
                    print("[RPi] Search OFF")
                state.search_active.clear()

        # Immediate stop
        elif mtype == P.TYPE_STOP:
            state.search_active.clear()
            stop = True

        # Center completion feedback
        elif mtype == P.TYPE_CENTER_DONE:
            try:
                center_done_cls = int(msg.get("target_cls", -1))
            except Exception:
                center_done_cls = -1
            center_success = bool(msg.get("success", False))
            center_diag = msg.get("diag", None)

        # Scan results
        elif mtype == P.TYPE_YOLO_RESULTS:
            yolo_results = msg

        # Recall: absolute go-to PWMs
        elif mcmd == P.CMD_GOTO_PWMS:
            # Only honor when not in search or centering
            if state.search_active.is_set() or state.centering_active.is_set():
                if debug:
                    print("[RPi] Ignoring GOTO_PWMS during search/centering")
                continue
            tb = int(msg.get("pwm_btm", 1500))
            tt = int(msg.get("pwm_top", 1500))
            slew = int(msg.get("slew_ms", 600))
            if debug:
                print(f"[RPi] GOTO_PWMS btm={tb} top={tt} slew_ms={slew}")
            goto_pwms(pi, tb, tt, duration_ms=slew, steps=24)

    return stop, center_done_cls, center_success, center_diag, yolo_results

def handle_client(conn, pi, debug: bool) -> None:
    """Top-level connection handler; enters search loop when activated."""
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
                setup_servos(pi)  # re-center after search

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
            # Clean up servos
            from .config import SERVO_BTM, SERVO_TOP
            pi.set_servo_pulsewidth(SERVO_BTM, 0)
            pi.set_servo_pulsewidth(SERVO_TOP, 0)
            pi.stop()
