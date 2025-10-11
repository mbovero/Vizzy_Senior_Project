# vizzy/laptop/motion.py
# -----------------------------------------------------------------------------
# Motion facade used by ScanWorker and TaskAgent/EXECUTE_TASK.
# - nudge_scan(dx, dy): send normalized [-1, 1] deltas (RPi scales & clamps)
# - goto_pose_pwm(pwm_btm, pwm_top, pose_id): absolute move (RPi echoes pose_id)
# - get_pwms(timeout_s): request current PWM pose from the RPi
# -----------------------------------------------------------------------------

from __future__ import annotations

import time
from queue import Queue, Empty
from threading import Event
from typing import Optional, Dict, Any

from ..shared import protocol as P
from ..shared import config as C
from ..shared.jsonl import send_json


class Motion:
    def __init__(
        self,
        sock,
        pose_ready_q: Queue,
        pwms_event: Event,
        pwms_payload: dict,
        *,
        abort_event: Optional[Event] = None,
    ):
        """
        Parameters
        ----------
        sock : socket.socket
            Connected TCP socket to the RPi server.
        pose_ready_q : queue.Queue
            Queue of pose-ready acks populated by the receiver thread.
        pwms_event : threading.Event
            Event set by the receiver thread when a PWMS message arrives.
        pwms_payload : dict
            Dict updated by the receiver thread with keys 'pwm_btm', 'pwm_top'.
        abort_event : threading.Event | None
            Optional event to signal that waiting for pose ready should abort early.
        """
        self.sock = sock
        self._pose_ready_q = pose_ready_q
        self._pwms_event = pwms_event
        self._pwms_payload = pwms_payload
        self._abort_event = abort_event

    # ------------------------------------------------------------------ utils

    def _send(self, payload: Dict[str, Any]) -> None:
        """Best-effort send; reconnects are orchestrator's responsibility."""
        try:
            send_json(self.sock, payload)
        except Exception:
            # Don’t crash worker threads on transient socket errors.
            pass

    def _drain_pose_ready(self) -> None:
        """Remove any stale pose-ready acknowledgements."""
        if self._pose_ready_q is None:
            return
        try:
            while True:
                self._pose_ready_q.get_nowait()
        except Empty:
            pass
        except Exception:
            pass

    def _wait_pose_ready(self, expected_id: Optional[int], slew_ms: int) -> bool:
        """
        Wait for a TYPE_POSE_READY ack. If expected_id is provided, only
        consider matching ids. Returns True on ack, False on timeout/abort.
        """
        if self._pose_ready_q is None:
            raise RuntimeError("Motion.goto_pose_pwm requires a pose_ready queue")

        settle = float(getattr(C, "POSE_SETTLE_S", 0.3))
        deadline = time.monotonic() + (slew_ms / 1000.0) + settle + 1.5
        while time.monotonic() < deadline:
            if self._abort_event is not None and getattr(self._abort_event, "is_set", lambda: False)():
                return False
            try:
                pose_id = self._pose_ready_q.get(timeout=0.05)
            except Empty:
                continue
            except Exception:
                continue

            try:
                pose_id = int(pose_id)
            except Exception:
                continue

            if expected_id is None or pose_id == int(expected_id):
                return True

        return False

    # ----------------------------------------------------------- public API

    def nudge_scan(self, dx: float, dy: float) -> None:
        """
        Send a SCAN_MOVE with normalized deltas in [-1, 1].
        The RPi applies MOVE_SCALE_US and clamps to servo limits.
        """
        # Clamp to [-1, 1]
        h = max(-1.0, min(1.0, float(dx)))
        v = max(-1.0, min(1.0, float(dy)))
        self._send({"type": P.TYPE_SCAN_MOVE, "horizontal": h, "vertical": v})

    def goto_pose_pwm(
        self,
        pwm_btm: int,
        pwm_top: int,
        pose_id: Optional[int] = None,
    ) -> bool:
        """
        Send an absolute PWM pose move to the RPi and block until POSE_READY.
        Returns True if the acknowledgement arrives, False on timeout/abort.
        """
        slew_ms = int(C.GOTO_POSE_SLEW_MS)

        if self._pose_ready_q is None:
            raise RuntimeError("Motion.goto_pose_pwm requires a pose_ready queue")

        self._drain_pose_ready()

        payload: Dict[str, Any] = {
            "cmd": P.CMD_GOTO_PWMS,
            "pwm_btm": int(pwm_btm),
            "pwm_top": int(pwm_top),
            "slew_ms": int(slew_ms),
        }
        if pose_id is not None:
            payload["pose_id"] = int(pose_id)

        self._send(payload)
        return self._wait_pose_ready(pose_id, slew_ms)

    def get_pwms(self, timeout_s: float = 0.3) -> Optional[Dict[str, int]]:
        """
        Request current PWM pose from the RPi and wait briefly for PWMS.

        Returns
        -------
        dict | None
            {"pwm_btm": int, "pwm_top": int} on success, or None on timeout/parse error.
        """
        try:
            self._pwms_event.clear()
        except Exception:
            pass

        self._send({"cmd": P.CMD_GET_PWMS})
        ok = self._pwms_event.wait(timeout_s)
        if not ok:
            return None
        try:
            return {
                "pwm_btm": int(self._pwms_payload["pwm_btm"]),
                "pwm_top": int(self._pwms_payload["pwm_top"]),
            }
        except Exception:
            return None
