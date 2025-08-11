# vizzy/rpi/servo.py
# -----------------------------------------------------------------------------
# Purpose
#   Low-level servo control functions for the Raspberry Pi side of Vizzy.
#   These functions handle **direct pulse-width commands** to the Pi's PWM
#   controller (via the `pigpio` library) and maintain the current servo
#   position in `state.py`.
#
# Why this exists
#   - Centralizes all physical servo movement code.
#   - Prevents scattering raw `pi.set_servo_pulsewidth(...)` calls across the
#     codebase.
#   - Keeps the shared servo position state (`state.current_horizontal`,
#     `state.current_vertical`) updated for any other module that needs it.
#
# How it fits into the project
#   - Called by search mode logic to move between scan positions.
#   - Called by centering routines to make fine adjustments toward an object.
#   - Called by the "goto pose" command from the laptop to restore saved poses.
#
# Implementation notes
#   - All pulse widths are in microseconds (µs), as required by the pigpio API.
#   - Constants like `SERVO_MIN`, `SERVO_MAX`, and `SERVO_CENTER` come from
#     `config.py` and are specific to your servo hardware.
#   - We store current servo pulse widths in `state.py` so multiple modules
#     can read/write them consistently.
#
# -----------------------------------------------------------------------------

from __future__ import annotations
import time
from . import state
from .config import SERVO_BTM, SERVO_TOP, SERVO_MID, SERVO_MIN, SERVO_MAX, SERVO_CENTER

# -----------------------------------------------------------------------------
# Utility: clamp a value to a range
# -----------------------------------------------------------------------------
def clamp(v: int, vmin: int, vmax: int) -> int:
    """
    Keep `v` within the range [vmin, vmax].
    If v < vmin, return vmin; if v > vmax, return vmax.
    """
    return max(vmin, min(vmax, v))

# -----------------------------------------------------------------------------
# Servo setup / initialization
# -----------------------------------------------------------------------------
def setup_servos(pi) -> None:
    """
    Initialize all servos to their "center" position.
    - `pi` is an active pigpio.pi() connection.
    - Sets mid, bottom, and top servos to SERVO_CENTER pulse width.
    - Updates global state with new positions.
    - Sleeps 1s to allow the arm to physically move and settle.
    """
    pi.set_servo_pulsewidth(SERVO_MID, SERVO_CENTER)
    pi.set_servo_pulsewidth(SERVO_BTM, SERVO_CENTER)
    pi.set_servo_pulsewidth(SERVO_TOP, SERVO_CENTER)
    state.current_horizontal = SERVO_CENTER
    state.current_vertical   = SERVO_CENTER
    time.sleep(1)

# -----------------------------------------------------------------------------
# Incremental movement
# -----------------------------------------------------------------------------
def move_servos(pi, horizontal: float, vertical: float) -> None:
    """
    Move servos incrementally based on normalized direction inputs.

    Args:
        horizontal: float in range [-1.0, 1.0]
        vertical:   float in range [-1.0, 1.0]

    Behavior:
        - Each axis is scaled to ~200 µs per "full" input.
        - Vertical input is inverted to match the camera frame (so pushing
          "up" moves the arm up visually).
        - New positions are clamped to [SERVO_MIN, SERVO_MAX] to protect hardware.
        - Updates both the hardware (via pigpio) and the `state` variables.
    """
    # Convert normalized input to PWM pulse width change
    h_change = int(horizontal * 200)
    v_change = int(vertical   * 200)

    # Calculate new pulse widths with clamping
    new_h = clamp(state.current_horizontal + h_change, SERVO_MIN, SERVO_MAX)
    new_v = clamp(state.current_vertical   - v_change, SERVO_MIN, SERVO_MAX)

    # Send pulse widths to hardware
    pi.set_servo_pulsewidth(SERVO_BTM, new_h)
    pi.set_servo_pulsewidth(SERVO_TOP, new_v)

    # Update global state
    state.current_horizontal = new_h
    state.current_vertical   = new_v

# -----------------------------------------------------------------------------
# Absolute movement with smoothing
# -----------------------------------------------------------------------------
def goto_pwms(pi, target_btm: int, target_top: int,
              duration_ms: int = 600, steps: int = 24) -> None:
    """
    Smoothly move servos to specific pulse-width targets.

    Args:
        target_btm:  target pulse width for bottom servo (µs)
        target_top:  target pulse width for top servo (µs)
        duration_ms: total movement time in milliseconds
        steps:       number of small increments to divide the motion into

    Behavior:
        - Clamps targets to safe servo range.
        - If steps <= 1 or duration_ms <= 0, jumps directly to target.
        - Otherwise:
            * Interpolates linearly from current to target pulse widths.
            * Sends each intermediate position with a short delay (`dt`)
              so movement is smooth instead of jerky.
        - Updates `state` positions after each increment.
    """
    # Clamp target positions
    tb = clamp(int(target_btm), SERVO_MIN, SERVO_MAX)
    tt = clamp(int(target_top),  SERVO_MIN, SERVO_MAX)

    # Starting positions
    sb = state.current_horizontal
    st = state.current_vertical

    # Immediate jump if no smoothing requested
    if steps <= 1 or duration_ms <= 0:
        pi.set_servo_pulsewidth(SERVO_BTM, tb)
        pi.set_servo_pulsewidth(SERVO_TOP, tt)
        state.current_horizontal, state.current_vertical = tb, tt
        return

    # Time per step (in seconds)
    dt = duration_ms / 1000.0 / steps

    # Perform smooth linear interpolation
    for i in range(1, steps + 1):
        nb = int(sb + (tb - sb) * (i / steps))  # next bottom position
        nt = int(st + (tt - st) * (i / steps))  # next top position
        pi.set_servo_pulsewidth(SERVO_BTM, nb)
        pi.set_servo_pulsewidth(SERVO_TOP, nt)
        state.current_horizontal, state.current_vertical = nb, nt
        time.sleep(dt)
