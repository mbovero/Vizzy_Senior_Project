# vizzy/rpi/state.py
# -----------------------------------------------------------------------------
# Shared runtime state for the Raspberry Pi side.
# - search_active:     True while the sweep FSM should run
# - centering_active:  True between POSE_READY and POSE_DONE (allows SCAN_MOVE)
# - current_*:         Live PWM positions (µs) for bottom/top servos
# -----------------------------------------------------------------------------

from __future__ import annotations

import threading

from ..shared import config as C

# Flags
search_active = threading.Event()
centering_active = threading.Event()

# Live servo pose (µs). Initialize to center so first goto_pwms() is well-defined.
current_horizontal: int = int(C.SERVO_CENTER)
current_vertical: int = int(C.SERVO_CENTER)
