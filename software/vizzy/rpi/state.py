# vizzy/rpi/state.py
from __future__ import annotations
from threading import Event

# Mode flags
search_active    = Event()
centering_active = Event()   # lets laptop drive servos during centering

# Current PWM positions
current_horizontal: int = 1500
current_vertical:   int = 1500
