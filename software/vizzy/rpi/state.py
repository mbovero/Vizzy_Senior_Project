# vizzy/rpi/state.py
# -----------------------------------------------------------------------------
# Shared runtime state for the Raspberry Pi side.
# - search_active:     True while the sweep FSM should run
# - centering_active:  Enables/disables TYPE_SCAN_MOVE processing
# - current_target:    Last commanded Cartesian pose (x, y, z, pitch)
# -----------------------------------------------------------------------------

from __future__ import annotations

import threading

from ..shared import config as C

# Flags
search_active = threading.Event()
centering_active = threading.Event()
centering_active.set()

# Cartesian target state shared across handlers
target_lock = threading.Lock()
current_target = {
    "x": float(C.REST_POSITION[0]),
    "y": float(C.REST_POSITION[1]),
    "z": float(C.REST_POSITION[2]),
    "pitch": float(C.REST_PITCH_ANGLE),
}
