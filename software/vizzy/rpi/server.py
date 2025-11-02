# vizzy/rpi/server.py
# -----------------------------------------------------------------------------
# Single-client JSONL TCP server for Vizzy RPi side (laptop-driven sweep).
# - Reads newline-delimited JSON messages from the laptop
# - Dispatches to protocol handlers
# -----------------------------------------------------------------------------

from __future__ import annotations

import socket
import time
from typing import Optional, Tuple

from ..shared import config as C
from ..shared.jsonl import recv_lines
from . import state
from .dispatch import process_messages


def _make_server_socket() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((C.LISTEN_HOST, int(C.LISTEN_PORT)))
    s.listen(1)
    print(f"[RPi] Listening on {C.LISTEN_HOST}:{C.LISTEN_PORT}")
    return s


def serve_forever(debug: bool = False) -> None:
    """Main server loop: accept one client and dispatch messages until STOP/close."""
    pi = None  # Placeholder for future IK/actuation backend

    server_sock = _make_server_socket()

    try:
        while True:
            print("[RPi] Waiting for laptop client...")
            conn, addr = server_sock.accept()
            print(f"[RPi] Client connected from {addr[0]}:{addr[1]}")

            # Reset state on new connection
            state.centering_active.set()   # allow nudges when idle
            with state.target_lock:
                state.current_target.update({
                    "x": float(C.REST_POSITION[0]),
                    "y": float(C.REST_POSITION[1]),
                    "z": float(C.REST_POSITION[2]),
                    "pitch": float(getattr(C, "REST_PITCH_ANGLE", 0.0)),
                })

            # Per-connection receive buffer
            buf = b""

            try:
                while True:
                    # Read any available messages (newline-delimited JSON)
                    try:
                        msgs, buf, closed = recv_lines(conn, buf)
                    except ConnectionResetError:
                        print("[RPi] Connection reset by peer.")
                        closed = True
                        msgs = []

                    if closed:
                        print("[RPi] Client disconnected.")
                        break

                    if msgs:
                        # Route messages; stop=True means TYPE_STOP received
                        stop = process_messages(pi, conn, msgs, debug=debug)
                        if stop:
                            print("[RPi] STOP requested; closing connection.")
                            try:
                                conn.shutdown(socket.SHUT_RDWR)
                            except Exception:
                                pass
                            conn.close()
                            return  # exit server

                    # Light tick to avoid a hot loop
                    time.sleep(0.01)

            except Exception as e:
                print(f"[RPi] Error in client loop: {e}")
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
                # Loop back to accept()

    finally:
        try:
            server_sock.close()
        except Exception:
            pass
        print("[RPi] Server shut down.")
