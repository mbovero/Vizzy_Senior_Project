# vizzy/rpi/server.py
# -----------------------------------------------------------------------------
# Single-client JSONL TCP server for Vizzy RPi side.
# - Reads newline-delimited JSON messages from the laptop
# - Dispatches to the new protocol handlers
# - Starts/stops the search sweep worker as search_active toggles
# -----------------------------------------------------------------------------

from __future__ import annotations

import socket
import threading
import time
from typing import Optional, Tuple

import pigpio

from ..shared import config as C
from ..shared.jsonl import recv_lines, send_json
from . import state
from . import dispatch
from .search import run_search_sweep
from .servo import init_servos

def _make_server_socket() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((C.LISTEN_HOST, int(C.LISTEN_PORT)))
    s.listen(1)
    print(f"[RPi] Listening on {C.LISTEN_HOST}:{C.LISTEN_PORT}")
    return s


def _start_sweep_thread(pi: pigpio.pi, conn: socket.socket, debug: bool = False) -> threading.Thread:
    t = threading.Thread(
        target=run_search_sweep,   # <— instead of search.run_search_sweep
        args=(pi, conn),
        kwargs={"debug": debug},
        daemon=True,
    )
    t.start()
    return t


def serve_forever(debug: bool = False) -> None:
    """Main server loop: accept one client, process messages, manage sweep worker."""
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("[RPi] pigpio daemon not running or not reachable.")

    init_servos(pi)
    print("[RPi] Servos initialized to center.")

    server_sock = _make_server_socket()

    try:
        while True:
            print("[RPi] Waiting for laptop client...")
            conn, addr = server_sock.accept()
            print(f"[RPi] Client connected from {addr[0]}:{addr[1]}")

            # Reset state on new connection
            state.search_active.clear()
            state.centering_active.clear()

            init_servos(pi)

            # Per-connection buffers and worker handle
            buf = b""
            sweep_thread: Optional[threading.Thread] = None

            try:
                while True:
                    # Read any available messages
                    try:
                        msgs, buf, closed = recv_lines(conn, buf)
                    except ConnectionResetError:
                        print("[RPi] Connection reset by peer.")
                        closed = True
                        msgs = []

                    if closed:
                        print("[RPi] Client disconnected.")
                        # Stop any ongoing sweep immediately
                        state.search_active.clear()
                        state.centering_active.clear()
                        break

                    if msgs:
                        # Route messages
                        stop = dispatch.process_messages(pi, conn, msgs, debug=debug)

                        if stop:
                            print("[RPi] STOP requested; closing connection.")
                            try:
                                conn.shutdown(socket.SHUT_RDWR)
                            except Exception:
                                pass
                            conn.close()
                            return  # exit server

                    # If search has been turned ON and no worker is running, start it
                    if state.search_active.is_set():
                        if sweep_thread is None or not sweep_thread.is_alive():
                            if debug:
                                print("[RPi] Starting search sweep worker...")
                            sweep_thread = _start_sweep_thread(pi, conn, debug=debug)
                    else:
                        # If search is OFF, let worker exit on its own; nothing to do here
                        pass

                    # Light tick
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
                # Reset flags; loop back to accept()
                state.search_active.clear()
                state.centering_active.clear()
                continue

            # Connection closed cleanly; loop back to accept()
            state.search_active.clear()
            state.centering_active.clear()

    finally:
        try:
            server_sock.close()
        except Exception:
            pass
        try:
            init_servos(pi)
            pi.stop()
        except Exception:
            pass
        print("[RPi] Server shut down.")
