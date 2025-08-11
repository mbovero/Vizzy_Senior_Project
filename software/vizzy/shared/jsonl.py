# vizzy/shared/jsonl.py
# -----------------------------------------------------------------------------
# Purpose
#   Tiny helper module that implements a simple, robust “newline-delimited JSON”
#   (JSONL) framing for our TCP socket messages between the Laptop and the RPi.
#
# Why this exists
#   TCP is a byte stream with no built-in message boundaries. If we just send
#   raw JSON, the receiver can’t tell where one message ends and the next one
#   begins. By appending a '\n' (newline) after each JSON object, we create an
#   easy boundary marker that both sides understand. This file provides:
#
#   - send_json(sock, obj):      serialize a Python dict to JSON and send it
#                                with a trailing newline. One call = one message.
#   - recv_lines(sock, buf):     read whatever bytes are available, split them
#                                by newline into *complete* JSON objects, and
#                                return any leftover partial data so callers can
#                                keep it for the next read.
#
# How it fits into the project
#   - Both vizzy.laptop and vizzy.rpi import these helpers to exchange messages
#     like:
#       * Commands (e.g., "YOLO_SCAN", "CENTER_ON", "GOTO_PWMS")
#       * Events   (e.g., "YOLO_RESULTS", "CENTER_DONE", "CENTER_SNAPSHOT")
#   - The laptop’s receive thread and the RPi server loop call recv_lines()
#     repeatedly with a persistent buffer. This lets them handle messages that
#     arrive in chunks or batches (common with TCP).
#
# Key idea: persistent buffer
#   recv_lines() takes a 'buf' (bytes) that the caller maintains between calls.
#   - New bytes are appended to 'buf'.
#   - Complete JSONL records (ending with '\n') are parsed and returned.
#   - Any leftover partial JSON (no newline yet) stays in 'buf' for later.
#
# Safety / resilience
#   - If a partial or malformed JSON line is encountered, we skip it rather than
#     crashing the program. (You can add logging here in the future.)
#   - When the socket returns no data (peer closed the connection), we signal
#     that with 'closed=True'.
#
# NOTE: This module deliberately stays small and dependency-free. It’s used by
#       both sides so they agree on the exact framing and parsing rules.
# -----------------------------------------------------------------------------

from __future__ import annotations
import json
from typing import Tuple, List, Any  # 'Any' is imported for potential future use

def send_json(sock, obj: dict) -> None:
    """
    Serialize a Python dict to JSON and send it over the socket with a newline.

    Why newline?
      We use '\n' to mark the end of each message so the receiver knows where to
      split. This is the "JSON Lines" (JSONL) convention.

    Args:
      sock: a connected socket.socket object (already opened/connected).
      obj:  a JSON-serializable Python dictionary.

    Returns:
      None. (Raises if the socket send fails.)
    """
    # Convert dict to text like '{"type":"search","active":true}'
    msg = json.dumps(obj) + "\n"
    # Send encoded bytes over TCP. One call = one message frame.
    sock.sendall(msg.encode("utf-8"))

def recv_lines(sock, buf: bytes) -> Tuple[List[dict], bytes, bool]:
    """
    Read available bytes from the socket and split them into complete JSON lines.

    This function is **non-blocking-friendly**: callers typically place the
    socket in non-blocking mode (or select/poll before calling), then pass a
    persistent 'buf' that accumulates bytes across calls.

    Args:
      sock: a connected socket.socket object.
      buf:  a *persistent* bytes buffer maintained by the caller between calls.
            Pass the 'remainder_buffer' returned from the previous call back
            into the next call.

    Returns:
      (messages, remainder_buffer, closed)
        - messages: a list of Python dicts parsed from complete JSON lines
                    received in this call (could be empty).
        - remainder_buffer: any leftover bytes that did not form a complete line
                            (e.g., partial JSON without a trailing '\n').
        - closed: True if the peer closed the connection (recv returned b'').

    Behavior details:
      - We append any newly received bytes to 'buf', then split at '\n'.
      - For each complete line, we try to parse it as JSON and collect the dict.
      - If a line is malformed JSON, we skip it (no exception raised here).
      - If no bytes are available and the socket would block, we return the
        current state with closed=False.
    """
    try:
        # Read up to 4096 bytes. In non-blocking mode this returns immediately.
        data = sock.recv(4096)
        if not data:
            # Peer closed the connection cleanly.
            return [], buf, True
        # Accumulate into the rolling buffer so we can assemble complete lines.
        buf += data
    except BlockingIOError:
        # Nothing to read right now; not an error in non-blocking polling loops.
        return [], buf, False

    msgs: List[dict] = []

    # As long as there's at least one newline, we have a complete line to parse.
    while b"\n" in buf:
        # Split off the first line; keep the rest in buf for the next iteration.
        line, buf = buf.split(b"\n", 1)

        # Skip empty/whitespace-only lines (defensive).
        if not line.strip():
            continue

        # Try to decode JSON. If it fails, ignore the malformed line.
        try:
            decoded = json.loads(line.decode("utf-8"))
            msgs.append(decoded)
        except json.JSONDecodeError:
            # Malformed line; ignore rather than crashing.
            # (Optional: add logging here if you want to see bad input.)
            pass

    # Return all complete messages we parsed, plus whatever bytes remain (partial),
    # and a flag telling the caller if the connection was closed.
    return msgs, buf, False
