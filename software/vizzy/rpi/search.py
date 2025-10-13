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

