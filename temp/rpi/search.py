# vizzy/rpi/search.py
# -----------------------------------------------------------------------------
# Laptop-centric sweep FSM:
#   For each grid pose: move -> settle -> POSE_READY -> enable centering window
#   Wait for laptop to send POSE_DONE (SUCCESS|SKIP|FAIL), then advance.
#   Accept SCAN_MOVE only during the centering window.
#   Stop immediately on SEARCH {active:false} (interrupt).
#   On final pose completion, emit SEARCH {active:false} (completed).
# -----------------------------------------------------------------------------

from __future__ import annotations

import time
from typing import Iterable, List, Tuple

from ..shared import config as C
from ..shared import protocol as P
from ..shared.jsonl import send_json

from . import state
from .servo import goto_pwms
from .dispatch import process_messages, take_pose_done


def _build_grid() -> List[Tuple[int, int]]:
    """
    Build a grid of (pwm_btm, pwm_top) pose targets from shared config.
    Uses trimmed ranges [MIN+MIN_OFFSET, MAX-MAX_OFFSET] with given steps.
    Traversal is 'snake' across rows to minimize long reversals.
    """
    b_lo = C.SERVO_MIN + C.SEARCH_MIN_OFFSET
    b_hi = C.SERVO_MAX - C.SEARCH_MAX_OFFSET
    t_lo = C.SERVO_MIN + C.SEARCH_MIN_OFFSET
    t_hi = C.SERVO_MAX - C.SEARCH_MAX_OFFSET

    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo
    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo

    # Build inclusive ranges with step > 0
    def arange_inclusive(lo: int, hi: int, step: int) -> List[int]:
        if step <= 0:
            step = 1
        vals = []
        x = lo
        if lo <= hi:
            while x <= hi:
                vals.append(int(x))
                x += step
        else:
            while x >= hi:
                vals.append(int(x))
                x -= step
        return vals

    btms = arange_inclusive(b_lo, b_hi, int(C.SEARCH_H_STEP))
    tops = arange_inclusive(t_lo, t_hi, int(C.SEARCH_V_STEP))

    grid: List[Tuple[int, int]] = []
    reverse = False
    for top in tops:
        row = [(b, top) for b in (reversed(btms) if reverse else btms)]
        grid.extend(row)
        reverse = not reverse
    return grid


def run_search_sweep(pi, conn, *, debug: bool = False) -> None:
    """
    Run the search sweep while state.search_active is set.
    Per pose:
      - goto pose
      - wait settle
      - send POSE_READY {pose_id}
      - enable centering window
      - wait for POSE_DONE or interrupt
      - advance
    On completion (last pose), send SEARCH {active:false} once.
    """
    grid = _build_grid()
    if debug:
        print(f"[RPi] Sweep grid size: {len(grid)} poses")

    # Walk the grid until interrupted or complete
    pose_id = 0
    for (pwm_btm, pwm_top) in grid:
        # If interrupted, exit immediately
        if not state.search_active.is_set():
            if debug:
                print("[RPi] Sweep interrupted before pose move.")
            return

        # Move to target pose smoothly
        if debug:
            print(f"[RPi] -> Pose {pose_id}: goto btm={pwm_btm} top={pwm_top}")
        goto_pwms(pi, pwm_btm, pwm_top, duration_ms=500, steps=24)

        # Mechanical settle
        time.sleep(float(C.POSE_SETTLE_S))

        # Announce pose ready to laptop
        try:
            send_json(conn, {"type": P.TYPE_POSE_READY, "pose_id": int(pose_id)})
            if debug:
                print(f"[RPi] <- POSE_READY {{pose_id:{pose_id}}}")
        except Exception as e:
            if debug:
                print(f"[RPi] Failed to send POSE_READY: {e}")
            # Connection issue => abort sweep
            return

        # Enable centering window: allow SCAN_MOVE until POSE_DONE
        state.centering_active.set()

        # Wait for POSE_DONE or interrupt
        # We service inbound messages frequently to allow SCAN_MOVE, GET_PWMS, etc.
        while True:
            # Interrupt: stop immediately
            if not state.search_active.is_set():
                if debug:
                    print("[RPi] Sweep interrupted during pose.")
                state.centering_active.clear()
                return

            # Non-blocking pump of any pending messages
            try:
                # The server loop feeds batches of messages; here we do not read the socket
                # directly; instead, server will call process_messages(...) and we
                # simply check for POSE_DONE via take_pose_done().
                pass
            except Exception:
                # Nothing to do here; server is responsible for reading
                pass

            # Check if POSE_DONE has arrived
            pid, status = take_pose_done()
            if pid is not None:
                # Accept only if this is for our current pose_id; otherwise ignore/stash semantics:
                if int(pid) == int(pose_id):
                    if debug:
                        print(f"[RPi] Pose {pose_id} done with status={status}")
                    state.centering_active.clear()
                    break
                else:
                    # Not for this pose; ignore and continue waiting
                    if debug:
                        print(f"[RPi] Ignored POSE_DONE for pid={pid} (current={pose_id})")

            # Gentle tick; actual servo nudge happens inside dispatch on SCAN_MOVE
            time.sleep(0.02)

        pose_id += 1

    # Completed all poses without interruption -> default stop
    if debug:
        print("[RPi] Sweep complete; sending SEARCH {active:false}")
    try:
        send_json(conn, {"type": P.TYPE_SEARCH, "active": False})
    except Exception as e:
        if debug:
            print(f"[RPi] Failed to send SEARCH completion: {e}")
