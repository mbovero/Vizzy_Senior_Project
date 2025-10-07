# vizzy/laptop/motion.py
# -----------------------------------------------------------------------------
# Motion facade used by ScanWorker and TaskAgent/EXECUTE_TASK.
# - nudge_scan(dx, dy): send normalized [-1, 1] deltas (RPi scales & clamps)
# - goto_pose_pwm(pwm_btm, pwm_top, slew_ms): absolute move for recall / tasks
# - get_pwms(timeout_s): request current PWM pose from the RPi
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Dict, Any

from ..shared import protocol as P
from ..shared.jsonl import send_json


class Motion:
    def __init__(self, sock, pwms_event, pwms_payload: dict):
        """
        Parameters
        ----------
        sock : socket.socket
            Connected TCP socket to the RPi server.
        pwms_event : threading.Event
            Event set by the receiver thread when a PWMS message arrives.
        pwms_payload : dict
            Dict updated by the receiver thread with keys 'pwm_btm', 'pwm_top'.
        """
        self.sock = sock
        self._pwms_event = pwms_event
        self._pwms_payload = pwms_payload

    # ------------------------------------------------------------------ utils

    def _send(self, payload: Dict[str, Any]) -> None:
        """Best-effort send; reconnects are orchestrator's responsibility."""
        send_json(self.sock, payload)

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

    def goto_pose_pwm(self, pwm_btm: int, pwm_top: int, *, slew_ms: int = 600) -> None:
        """
        Send an absolute PWM pose move to the RPi (ignored during search/centering).
        """
        self._send({
            "cmd": P.CMD_GOTO_PWMS,
            "pwm_btm": int(pwm_btm),
            "pwm_top": int(pwm_top),
            "slew_ms": int(slew_ms),
        })

    def get_pwms(self, timeout_s: float = 0.3) -> Optional[Dict[str, int]]:
        """
        Request current PWM pose from the RPi and wait briefly for PWMS.

        Returns
        -------
        dict | None
            {"pwm_btm": int, "pwm_top": int} on success, or None on timeout/parse error.
        """
        self._pwms_event.clear()
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
