# vizzy/shared/jsonl.py
from __future__ import annotations
import json
from typing import Tuple, List, Any

def send_json(sock, obj: dict) -> None:
    """Send one JSON object per line (newline-delimited JSON)."""
    msg = json.dumps(obj) + "\n"
    sock.sendall(msg.encode("utf-8"))

def recv_lines(sock, buf: bytes) -> Tuple[List[dict], bytes, bool]:
    """
    Receive available bytes from socket, split into complete JSON lines.
    Returns (messages, remainder_buffer, closed).
    """
    try:
        data = sock.recv(4096)
        if not data:
            return [], buf, True
        buf += data
    except BlockingIOError:
        return [], buf, False
    msgs: List[dict] = []
    while b"\n" in buf:
        line, buf = buf.split(b"\n", 1)
        if not line.strip():
            continue
        try:
            msgs.append(json.loads(line.decode("utf-8")))
        except json.JSONDecodeError:
            # ignore malformed lines
            pass
    return msgs, buf, False
