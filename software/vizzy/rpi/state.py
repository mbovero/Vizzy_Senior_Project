# vizzy/rpi/state.py
# -----------------------------------------------------------------------------
# Purpose
#   Holds **shared state variables** for the Raspberry Pi server side of Vizzy.
#   These variables represent the current operating mode and servo positions,
#   and are imported by multiple RPi modules (e.g., the search logic, server
#   message loop, and centering functions).
#
# Why this exists
#   The RPi codebase has several threads/tasks running in parallel:
#     - The TCP server thread handles incoming commands from the laptop.
#     - The search mode logic drives the arm through its scanning pattern.
#     - The centering logic adjusts servos based on laptop YOLO feedback.
#   These tasks need to share some **global state**, such as:
#     - Whether search mode is currently active.
#     - Whether the RPi is in the middle of a centering sequence.
#     - The current PWM values of the base and top servos.
#
#   By putting these in one small, central module, we:
#     - Avoid circular imports (each file can import `state` instead of each
#       other).
#     - Keep a single source of truth for mode flags and servo positions.
#     - Make it clear which variables are meant to be shared globally.
#
# How it fits into the project
#   - `search_active` is set/cleared when search mode starts or stops.
#   - `centering_active` is set when the laptop has control of servos for a
#     centering operation; cleared afterwards so search or manual movement can
#     resume.
#   - `current_horizontal` and `current_vertical` track the *most recent*
#     pulse-width values sent to the horizontal and vertical servos. This
#     allows us to:
#       * Save/restore arm poses.
#       * Report current position to the laptop if needed.
#       * Keep arm movement functions in sync with the physical state.
#
# Thread-safety
#   - The Event objects (`search_active`, `centering_active`) are thread-safe
#     synchronization primitives from Python’s `threading` module. Multiple
#     threads can set(), clear(), or check them without additional locks.
#   - The integer PWM variables are simple globals; if multiple threads will
#     write to them frequently, consider adding a lock.
#
# -----------------------------------------------------------------------------

from __future__ import annotations
from threading import Event

# -----------------------------
# Mode flags (thread-safe)
# -----------------------------

# True when the RPi is running its programmed search pattern and
# requesting YOLO scans from the laptop.
search_active    = Event()

# True when the RPi is allowing the laptop to directly control servos
# for object centering. Prevents other motion code from interfering.
centering_active = Event()

# -----------------------------
# Current PWM positions (µs)
# -----------------------------
# These track the *actual* last pulse widths sent to the servos, so
# any module can know the current arm pose.
# Default to neutral (1500 µs) at startup.

current_horizontal: int = 1500  # Base rotation servo pulse width
current_vertical:   int = 1500  # Arm tilt servo pulse width
