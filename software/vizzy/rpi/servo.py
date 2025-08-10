# vizzy/rpi/servo.py
from __future__ import annotations
import time
from . import state
from .config import SERVO_BTM, SERVO_TOP, SERVO_MID, SERVO_MIN, SERVO_MAX, SERVO_CENTER

def clamp(v: int, vmin: int, vmax: int) -> int:
    return max(vmin, min(vmax, v))

def setup_servos(pi) -> None:
    """Initialize servos to center position."""
    pi.set_servo_pulsewidth(SERVO_MID, SERVO_CENTER)
    pi.set_servo_pulsewidth(SERVO_BTM, SERVO_CENTER)
    pi.set_servo_pulsewidth(SERVO_TOP, SERVO_CENTER)
    state.current_horizontal = SERVO_CENTER
    state.current_vertical   = SERVO_CENTER
    time.sleep(1)

def move_servos(pi, horizontal: float, vertical: float) -> None:
    """
    Incremental move: normalized inputs (-1..1) scaled by ~200 µs per step.
    Vertical is inverted to match camera frame.
    """
    h_change = int(horizontal * 200)
    v_change = int(vertical   * 200)
    new_h = clamp(state.current_horizontal + h_change, SERVO_MIN, SERVO_MAX)
    new_v = clamp(state.current_vertical   - v_change, SERVO_MIN, SERVO_MAX)
    pi.set_servo_pulsewidth(SERVO_BTM, new_h)
    pi.set_servo_pulsewidth(SERVO_TOP, new_v)
    state.current_horizontal = new_h
    state.current_vertical   = new_v

def goto_pwms(pi, target_btm: int, target_top: int, duration_ms: int = 600, steps: int = 24) -> None:
    """
    Smoothly slew to absolute PWM targets over duration_ms using linear interpolation.
    """
    tb = clamp(int(target_btm), SERVO_MIN, SERVO_MAX)
    tt = clamp(int(target_top),  SERVO_MIN, SERVO_MAX)

    sb = state.current_horizontal
    st = state.current_vertical

    if steps <= 1 or duration_ms <= 0:
        pi.set_servo_pulsewidth(SERVO_BTM, tb)
        pi.set_servo_pulsewidth(SERVO_TOP, tt)
        state.current_horizontal, state.current_vertical = tb, tt
        return

    dt = duration_ms / 1000.0 / steps
    for i in range(1, steps + 1):
        nb = int(sb + (tb - sb) * (i / steps))
        nt = int(st + (tt - st) * (i / steps))
        pi.set_servo_pulsewidth(SERVO_BTM, nb)
        pi.set_servo_pulsewidth(SERVO_TOP, nt)
        state.current_horizontal, state.current_vertical = nb, nt
        time.sleep(dt)
