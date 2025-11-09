# vizzy/laptop/motion.py
# -----------------------------------------------------------------------------
# Motion facade used by ScanWorker and TaskAgent/EXECUTE_TASK.
# - nudge_scan(dx, dy): send normalized [-1, 1] deltas (RPi converts to mm offsets)
# - move_to_target(x, y, z, pitch): absolute Cartesian move (waits for ACK from server)
# - request_obj_location(): fetch current commanded Cartesian target from the RPi
# 
# NOTE: This now sends text-based 'ik' commands to rpi_control_server.py
# Commands are in format: "ik x y z pitch_rad yaw_rad O|C\n"
# Server responds with: "ACK ik\n" or "ERR ...\n"
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
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

    def _send_text(self, text: str) -> None:
        """Send text command to rpi_control_server (text-based protocol)."""
        try:
            self.sock.sendall(text.encode("utf-8"))
        except Exception as e:
            # Don't crash worker threads on transient socket errors, but log them
            print(f"[Motion] ERROR sending command: {e}")
            raise

    def _send(self, payload: Dict[str, Any]) -> None:
        """Best-effort send JSON (kept for backward compatibility, but not used for search)."""
        try:
            send_json(self.sock, payload)
        except Exception:
            # Don't crash worker threads on transient socket errors.
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
        """
        Wait for command completion from the server.
        For rpi_control_server, this waits for "ACK ik" or "ERR ..." messages
        that are parsed by the receiver thread and put into the cmd_complete_q.
        """
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

            # Handle both JSON protocol (old) and text protocol (new)
            if isinstance(msg, dict):
                cmd = str(msg.get("cmd", "")).upper()
                if cmd != expected_cmd:
                    continue
                status = str(msg.get("status", "success")).lower()
                return status == "success"
            elif isinstance(msg, str):
                # Text protocol: "ACK ik" or "ERR ..."
                # For "IK" commands, server responds with "ACK ik" or "ERR ..."
                if expected_cmd == "IK":
                    if msg.strip().upper() == "ACK IK" or msg.strip().upper().startswith("ACK IK"):
                        return True
                    elif msg.strip().upper().startswith("ERR"):
                        return False
                else:
                    # For other commands, accept any ACK
                    if msg.strip().upper().startswith("ACK"):
                        return True
                    elif msg.strip().upper().startswith("ERR"):
                        return False

        return False

    # ----------------------------------------------------------- public API

    def nudge_scan(self, dx: float, dy: float) -> None:
        """
        Send a SCAN_MOVE with normalized deltas in [-1, 1].
        NOTE: This is not used with rpi_control_server. For search, we use move_to_target
        with absolute coordinates. This is kept for backward compatibility but does nothing
        with the new server architecture.
        """
        # For rpi_control_server, nudges should be converted to absolute moves
        # This method is kept for API compatibility but doesn't send commands
        # The centering code should call move_to_target with updated absolute coordinates
        pass

    def move_to_target(
        self,
        x: float,
        y: float,
        z: float,
        pitch: float,
        *,
        timeout_s: Optional[float] = None,
        yaw: float = 0.0,
        claw: str = "O",
    ) -> bool:
        """
        Send an 'ik' command to rpi_control_server and wait for ACK.
        
        Parameters
        ----------
        x, y, z : float
            Target position in MILLIMETERS (converted to meters for server)
        pitch : float
            Pitch angle in DEGREES (converted to radians for server)
        yaw : float
            Yaw angle in DEGREES (converted to radians for server). Default 0.0.
        claw : str
            Claw state: "O" (open) or "C" (closed). Default "O".
        timeout_s : float, optional
            Timeout in seconds. Defaults to C.PRIMITIVE_CMD_TIMEOUT.
        
        Returns
        -------
        bool
            True on success (ACK received), False on timeout/abort/failure.
        """
        timeout = float(timeout_s or C.PRIMITIVE_CMD_TIMEOUT)

        # Convert units: mm -> meters, degrees -> radians
        x_m = float(x) / 1000.0
        y_m = float(y) / 1000.0
        z_m = float(z) / 1000.0
        pitch_rad = math.radians(float(pitch))
        yaw_rad = math.radians(float(yaw))
        
        # Validate claw state
        claw_upper = str(claw).strip().upper()
        if claw_upper not in ("O", "C"):
            claw_upper = "O"

        self._drain_cmd_queue()
        
        # Format: "ik x y z pitch_rad yaw_rad O|C\n"
        cmd_text = f"ik {x_m:.6f} {y_m:.6f} {z_m:.6f} {pitch_rad:.6f} {yaw_rad:.6f} {claw_upper}\n"
        print(f"[Motion] Sending ik command: x={x:.2f}mm y={y:.2f}mm z={z:.2f}mm pitch={pitch:.3f}° ({pitch_rad:.6f}rad) yaw={yaw:.3f}° claw={claw_upper}")
        print(f"[Motion] Command string: {cmd_text.strip()}")
        self._send_text(cmd_text)
        
        result = self._wait_cmd_complete("IK", timeout)
        if result:
            print(f"[Motion] Command acknowledged by server")
        else:
            print(f"[Motion] Command timeout or error (timeout={timeout}s)")
        return result

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
