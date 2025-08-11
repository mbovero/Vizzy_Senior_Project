# vizzy/rpi/server.py
# -----------------------------------------------------------------------------
# Purpose
#   This is the **main TCP server** that runs on the Raspberry Pi in the Vizzy
#   robotic arm system. It listens for incoming connections from the laptop
#   (the client), receives JSON-based commands, and coordinates execution of
#   arm control routines.
#
# Why this exists
#   - Serves as the network interface between the laptop’s higher-level vision/
#     AI code and the Pi’s low-level servo/motion control logic.
#   - Keeps the Pi "dumb" in terms of object detection: it just executes moves
#     and search cycles based on commands sent from the laptop.
#
# How it fits into the project
#   - The laptop connects over Wi-Fi/TCP and sends messages like "start search",
#     "center on object", or "move to stored pose".
#   - The Pi runs the requested operation (including calling into `search.py`)
#     and returns status updates or snapshots.
#   - This script is typically launched with `python -m vizzy.rpi` and runs
#     until interrupted.
#
# Key behaviors
#   - Listens for 1 client at a time (the laptop).
#   - Uses `select.select()` for non-blocking I/O so it can both receive
#     commands and run search cycles without freezing.
#   - Resets servo positions after each search cycle.
#   - Cleans up GPIO resources on shutdown.
#
# Implementation details
#   - Uses `pigpio` to control PWM outputs for the servos.
#   - Messages are received via `recv_lines()` (newline-delimited JSON).
#   - The actual message handling logic is in `dispatch.py` (function
#     `process_messages()`).
#   - `state.py` holds global flags for whether search/centering is active.
#
# -----------------------------------------------------------------------------

from __future__ import annotations
import socket, select, pigpio
from ..shared.jsonl import recv_lines
from .servo import setup_servos
from .search import run_search_cycle
from .config import LISTEN_HOST, LISTEN_PORT, SERVO_BTM, SERVO_TOP
from . import state
from .dispatch import process_messages  # <-- message routing logic

# -----------------------------------------------------------------------------
# Per-connection handler
# -----------------------------------------------------------------------------
def handle_client(conn, pi, debug: bool) -> None:
    """
    Handle all communication with a single connected laptop client.

    Args:
        conn: socket object representing the TCP connection.
        pi:   pigpio.pi() connection for controlling the servos.
        debug: True to enable extra diagnostic output.

    Behavior:
        - Sets the socket to non-blocking mode.
        - Loops, waiting for readable data (with a short timeout so we can
          periodically check state flags).
        - Reads any complete JSON messages and passes them to
          `process_messages()`.
        - If a "stop" flag is returned by `process_messages()`, exits.
        - If `state.search_active` is set, runs one iteration of the search
          cycle (`run_search_cycle()`).
        - After a search cycle completes, re-centers the servos with
          `setup_servos()` so the arm is ready for next command.
    """
    conn.setblocking(False)   # don't block on recv()
    buf = b""                 # leftover partial data from last recv()

    try:
        while True:
            # Wait for the socket to be ready to read, up to 0.2 seconds
            readable, _, _ = select.select([conn], [], [], 0.2)
            if not readable:
                continue  # no new data yet

            # Read and parse any available messages
            msgs, buf, closed = recv_lines(conn, buf)
            if closed:
                break  # client disconnected

            # Let dispatch layer handle messages
            stop, _, _, _, _ = process_messages(pi, msgs, debug=debug)
            if stop:
                return  # exit if instructed

            # If search mode is active, run one search cycle step
            if state.search_active.is_set():
                run_search_cycle(pi, conn, debug=debug)
                # After each cycle, return servos to center
                setup_servos(pi)

    finally:
        # Always close connection and reset flags on exit
        conn.close()
        state.search_active.clear()
        state.centering_active.clear()

# -----------------------------------------------------------------------------
# Server main loop
# -----------------------------------------------------------------------------
def main(debug: bool) -> None:
    """
    Start the Raspberry Pi TCP server and wait for incoming laptop connections.

    Args:
        debug: True to enable extra diagnostic printouts.

    Behavior:
        - Connects to pigpio daemon (must be running on the Pi).
        - Initializes servos to center position.
        - Creates a TCP socket bound to LISTEN_HOST:LISTEN_PORT.
        - Accepts incoming client connections in a loop.
        - For each client, calls `handle_client()` to process messages until
          the client disconnects or an error occurs.
        - Cleans up PWM outputs and stops pigpio on shutdown.
    """
    # Connect to pigpio daemon
    pi = pigpio.pi()
    if not pi.connected:
        print("Failed to connect to pigpio daemon")
        return

    # Initialize servos to known center position
    setup_servos(pi)

    # Create listening TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # quick restart
        s.bind((LISTEN_HOST, LISTEN_PORT))
        s.listen()
        print(f"RPi server started (debug={'on' if debug else 'off'}), waiting for connections...")

        try:
            while True:
                # Wait for a client (the laptop)
                conn, addr = s.accept()
                print(f"Connected by {addr}")
                try:
                    handle_client(conn, pi, debug=debug)
                except Exception as e:
                    print(f"[RPi] Client handler error: {e}")
        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            # Clean up servos on shutdown
            pi.set_servo_pulsewidth(SERVO_BTM, 0)
            pi.set_servo_pulsewidth(SERVO_TOP, 0)
            pi.stop()
