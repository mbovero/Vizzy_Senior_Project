# vizzy/rpi/servo.py
# -----------------------------------------------------------------------------
# Low-level servo control functions for the Raspberry Pi side of Vizzy.
# - move_servos(): apply small normalized deltas ([-1, 1]) scaled by MOVE_SCALE_US
# - goto_pwms():   smooth absolute move with clamping to SERVO_MIN/MAX
# Keeps state.current_horizontal / state.current_vertical in sync with hardware.
# -----------------------------------------------------------------------------

from __future__ import annotations

import time
from typing import Tuple

from ..shared import config as C
from . import state

# pigpio.pi() instance is passed in by caller (server/dispatch/search)

# --- add near top, after imports and state import ---
def init_servos(pi) -> None:
    """
    Initialize servo outputs at SERVO_CENTER so they are powered and not limp.
    Safe to call multiple times.
    """
    pi.set_servo_pulsewidth(C.SERVO_BTM, C.SERVO_CENTER)
    pi.set_servo_pulsewidth(C.SERVO_MID, C.SERVO_CENTER)
    pi.set_servo_pulsewidth(C.SERVO_TOP, C.SERVO_CENTER)
    state.current_horizontal = C.SERVO_CENTER
    state.current_vertical = C.SERVO_CENTER


def _clamp(v: int, vmin: int, vmax: int) -> int:
    """Clamp integer v to [vmin, vmax]."""
    return max(vmin, min(vmax, v))


def _apply(pi, pwm_btm: int, pwm_top: int) -> None:
    """Send pulse widths to hardware and update shared state."""
    pi.set_servo_pulsewidth(C.SERVO_BTM, pwm_btm)
    pi.set_servo_pulsewidth(C.SERVO_TOP, pwm_top)
    state.current_horizontal = pwm_btm
    state.current_vertical = pwm_top


def move_servos(pi, dx: float, dy: float, *, scale_us: int | float = None) -> Tuple[int, int]:
    """
    Apply small *normalized* deltas to the current PWM positions.
    The laptop sends SCAN_MOVE with dx, dy in [-1, 1].
    We scale by MOVE_SCALE_US and clamp to [SERVO_MIN, SERVO_MAX].

    Returns (new_pwm_btm, new_pwm_top).
    """
    if scale_us is None:
        scale_us = C.MOVE_SCALE_US

    # Clamp deltas to [-1, 1]
    dx = max(-1.0, min(1.0, float(dx)))
    dy = -max(-1.0, min(1.0, float(dy)))

    # Convert to microseconds and compute new targets
    delta_btm = int(round(dx * float(scale_us)))
    delta_top = int(round(dy * float(scale_us)))

    nb = _clamp(state.current_horizontal + delta_btm, C.SERVO_MIN, C.SERVO_MAX)
    nt = _clamp(state.current_vertical   + delta_top, C.SERVO_MIN, C.SERVO_MAX)

    if nb == state.current_horizontal and nt == state.current_vertical:
        # No change after clamping; avoid unnecessary pigpio calls
        return nb, nt

    _apply(pi, nb, nt)
    return nb, nt


def goto_pwms(
    pi,
    pwm_btm: int,
    pwm_top: int,
    *,
    duration_ms: int = 500,
    steps: int = 24,
) -> None:
    """
    Smooth absolute move from current pose to (pwm_btm, pwm_top).
    Clamps targets to [SERVO_MIN, SERVO_MAX], interpolates over `steps`
    across `duration_ms`, and updates shared state each substep.
    """
    tb = _clamp(int(pwm_btm), C.SERVO_MIN, C.SERVO_MAX)
    tt = _clamp(int(pwm_top), C.SERVO_MIN, C.SERVO_MAX)

    sb = int(state.current_horizontal)
    st = int(state.current_vertical)

    # Degenerate case: already there
    if tb == sb and tt == st:
        _apply(pi, tb, tt)
        return

    # Steps/time
    steps = max(1, int(steps))
    dt = (max(0, int(duration_ms)) / 1000.0) / steps if duration_ms > 0 else 0.0

    # Linear interpolation
    for i in range(1, steps + 1):
        nb = int(round(sb + (tb - sb) * (i / steps)))
        nt = int(round(st + (tt - st) * (i / steps)))
        _apply(pi, nb, nt)
        if dt > 0:
            time.sleep(dt)

    # Ensure exact target at end
    if state.current_horizontal != tb or state.current_vertical != tt:
        _apply(pi, tb, tt)
