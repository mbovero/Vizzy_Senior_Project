# vizzy/laptop/motion.py
# -----------------------------------------------------------------------------
# Motion facade used by ScanWorker and TaskAgent/EXECUTE_TASK.
# - nudge_scan(dx, dy): send normalized [-1, 1] deltas (RPi converts to mm offsets)
# - move_to_target(x, y, z, pitch): absolute Cartesian move (waits for CMD_COMPLETE)
# - request_obj_location(): fetch current commanded Cartesian target from the RPi
# -----------------------------------------------------------------------------

from __future__ import annotations

import time
from queue import Empty, Queue
from threading import Event
from typing import Any, Dict, Optional

from ..shared import config as C
from ..shared import protocol as P
from ..shared.jsonl import send_json


class Motion:
    def __init__(
        self,
        sock,
        cmd_complete_q: Queue,
        obj_loc_event: Event,
        obj_loc_payload: dict,
        *,
        abort_event: Optional[Event] = None,
    ):
        """
        Parameters
        ----------
        sock : socket.socket
            Connected TCP socket to the RPi server.
        cmd_complete_q : queue.Queue
            Queue populated by the receiver thread with TYPE_CMD_COMPLETE payloads.
        obj_loc_event : threading.Event
            Event set by the receiver thread when a TYPE_OBJ_LOC message arrives.
        obj_loc_payload : dict
            Dict updated by the receiver thread with keys 'x', 'y', 'z'.
        abort_event : threading.Event | None
            Optional event to signal that waits should abort early.
        """
        self.sock = sock
        self._cmd_complete_q = cmd_complete_q
        self._obj_loc_event = obj_loc_event
        self._obj_loc_payload = obj_loc_payload
        self._abort_event = abort_event

    # ------------------------------------------------------------------ utils

    def _send(self, payload: Dict[str, Any]) -> None:
        """Best-effort send; reconnects are orchestrator's responsibility."""
        try:
            send_json(self.sock, payload)
        except Exception:
            # Don’t crash worker threads on transient socket errors.
            pass

    def _drain_cmd_queue(self) -> None:
        """Remove any stale command completion acknowledgements."""
        if self._cmd_complete_q is None:
            return
        try:
            while True:
                self._cmd_complete_q.get_nowait()
        except Empty:
            pass
        except Exception:
            pass

    def _wait_cmd_complete(self, expected_cmd: str, timeout_s: float) -> bool:
        """Wait for TYPE_CMD_COMPLETE matching expected_cmd."""
        if self._cmd_complete_q is None:
            raise RuntimeError("Motion.move_to_target requires a cmd_complete queue")

        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self._abort_event is not None and getattr(self._abort_event, "is_set", lambda: False)():
                return False
            remaining = max(0.01, deadline - time.monotonic())
            try:
                msg = self._cmd_complete_q.get(timeout=min(0.1, remaining))
            except Empty:
                continue
            except Exception:
                continue

            if not isinstance(msg, dict):
                continue

            cmd = str(msg.get("cmd", "")).upper()
            if cmd != expected_cmd:
                # Unexpected completion (likely belongs to another flow); drop it.
                continue

            status = str(msg.get("status", "success")).lower()
            return status == "success"

        return False

    # ----------------------------------------------------------- public API

    def nudge_scan(self, dx: float, dy: float) -> None:
        """
        Send a SCAN_MOVE with normalized deltas in [-1, 1].
        The RPi converts these values to millimetre offsets.
        """
        h = max(-1.0, min(1.0, float(dx)))
        v = max(-1.0, min(1.0, float(dy)))
        self._send({"type": P.TYPE_SCAN_MOVE, "horizontal": h, "vertical": v})

    def move_to_target(
        self,
        x: float,
        y: float,
        z: float,
        pitch: float,
        *,
        timeout_s: Optional[float] = None,
    ) -> bool:
        """
        Send a MOVE_TO command to the RPi and wait for TYPE_CMD_COMPLETE.
        Returns True on success, False on timeout/abort/failure.
        """
        timeout = float(timeout_s or C.PRIMITIVE_CMD_TIMEOUT)

        self._drain_cmd_queue()
        payload: Dict[str, Any] = {
            "cmd": P.CMD_MOVE_TO,
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "pitch": float(pitch),
        }
        self._send(payload)
        return self._wait_cmd_complete(P.CMD_MOVE_TO, timeout)

    def request_obj_location(self, timeout_s: float = 0.5) -> Optional[Dict[str, float]]:
        """
        Request the current commanded Cartesian target from the RPi.

        Returns
        -------
        dict | None
            {"x": float, "y": float, "z": float} on success, or None on timeout/parse error.
        """
        if self._obj_loc_event is None:
            raise RuntimeError("Motion.request_obj_location requires an obj_loc_event")

        try:
            self._obj_loc_event.clear()
        except Exception:
            pass

        self._send({"cmd": P.CMD_GET_OBJ_LOC})
        ok = self._obj_loc_event.wait(timeout_s)
        if not ok:
            return None
        try:
            return {
                "x": float(self._obj_loc_payload["x"]),
                "y": float(self._obj_loc_payload["y"]),
                "z": float(self._obj_loc_payload["z"]),
            }
        except Exception:
            return None
