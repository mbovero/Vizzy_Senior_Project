# vizzy/rpi/dispatch.py
# -----------------------------------------------------------------------------
# Purpose
#   This module defines the **central message dispatcher** for the Raspberry Pi
#   side of the Vizzy robotic arm project.
#
# Why this exists
#   - All messages from the laptop (received over the TCP socket) come through
#     here for interpretation and action.
#   - This function decides what each message means and which part of the Pi's
#     code should handle it.
#
# How it fits into the project
#   - The Pi's server receives raw JSON messages from the laptop (via jsonl.py).
#   - These messages are passed here to `process_messages`, which:
#       * Updates global state flags (search/centering)
#       * Moves servos directly when appropriate
#       * Handles centering completion notifications
#       * Saves YOLO scan results
#       * Processes "go to position" commands
#   - This is the main **control switchboard** for the Pi.
#
# Key points for understanding:
#   - Messages can either be "type" messages (P.TYPE_...) or "command" messages
#     (P.CMD_...).
#   - Some commands are ignored when the Pi is busy searching or centering.
#   - The function returns a tuple of information so the caller can act on it:
#       (stop_requested, center_done_cls, center_success, center_diag, yolo_results)
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Tuple, Optional
from ..shared import protocol as P
from . import state
from .servo import move_servos, goto_pwms

def process_messages(pi, msgs, debug: bool
                     ) -> Tuple[bool, Optional[int], Optional[bool], Optional[dict], Optional[dict]]:
    """
    Process a list of incoming messages from the laptop.

    Args:
        pi: pigpio instance controlling the servos.
        msgs: List of decoded JSON messages from the socket.
        debug: If True, print extra details about actions taken.

    Returns:
        A tuple of:
          stop (bool)               - True if a stop command was received.
          center_done_cls (int|None)- Class ID of object just centered on, if any.
          center_success (bool|None)- True if centering was verified as successful.
          center_diag (dict|None)   - Optional diagnostic data from centering.
          yolo_results (dict|None)  - Latest YOLO scan results, if any.
    """
    stop = False  # flag: should the search loop stop?
    center_done_cls: Optional[int] = None
    center_success: Optional[bool] = None
    center_diag: Optional[dict] = None
    yolo_results: Optional[dict] = None

    # Loop through each message in the batch
    for msg in msgs:
        mtype = msg.get("type")  # event type (e.g., TYPE_MOVE)
        mcmd  = msg.get("cmd")   # request command (e.g., CMD_GOTO_PWMS)

        # -------------------------------------------------
        # TYPE_MOVE: joystick-like incremental servo control
        # -------------------------------------------------
        if mtype == P.TYPE_MOVE:
            # Only allow servo motion if we're centering OR not searching
            if state.centering_active.is_set() or not state.search_active.is_set():
                move_servos(pi,
                            msg.get("horizontal", 0.0),  # horizontal move amount (-1..1)
                            msg.get("vertical", 0.0))    # vertical move amount (-1..1)

        # -------------------------------------------------
        # TYPE_SEARCH: start or stop the Pi's search mode
        # -------------------------------------------------
        elif mtype == P.TYPE_SEARCH:
            if msg.get("active", False):
                if not state.search_active.is_set():
                    print("[RPi] Search ON")
                state.search_active.set()
            else:
                if state.search_active.is_set():
                    print("[RPi] Search OFF")
                state.search_active.clear()

        # -------------------------------------------------
        # TYPE_STOP: immediate halt of all search/centering
        # -------------------------------------------------
        elif mtype == P.TYPE_STOP:
            state.search_active.clear()
            stop = True

        # -------------------------------------------------
        # TYPE_CENTER_DONE: laptop reports centering result
        # -------------------------------------------------
        elif mtype == P.TYPE_CENTER_DONE:
            try:
                center_done_cls = int(msg.get("target_cls", -1))
            except Exception:
                center_done_cls = -1  # invalid class ID
            center_success = bool(msg.get("success", False))
            center_diag = msg.get("diag", None)  # diagnostic info (thresholds, observed stats)

        # -------------------------------------------------
        # TYPE_YOLO_RESULTS: laptop sends scan detections
        # -------------------------------------------------
        elif mtype == P.TYPE_YOLO_RESULTS:
            yolo_results = msg

        # -------------------------------------------------
        # CMD_GOTO_PWMS: move servos to absolute positions
        # -------------------------------------------------
        elif mcmd == P.CMD_GOTO_PWMS:
            # Ignore if busy with search or centering
            if state.search_active.is_set() or state.centering_active.is_set():
                if debug:
                    print("[RPi] Ignoring GOTO_PWMS during search/centering")
                continue
            tb = int(msg.get("pwm_btm", 1500))  # bottom servo PWM
            tt = int(msg.get("pwm_top", 1500))  # top servo PWM
            slew = int(msg.get("slew_ms", 600)) # move duration
            if debug:
                print(f"[RPi] GOTO_PWMS btm={tb} top={tt} slew_ms={slew}")
            goto_pwms(pi, tb, tt, duration_ms=slew, steps=24)

    return stop, center_done_cls, center_success, center_diag, yolo_results
