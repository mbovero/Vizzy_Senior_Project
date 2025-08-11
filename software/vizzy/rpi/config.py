# vizzy/rpi/config.py
# -----------------------------------------------------------------------------
# Purpose
#   This file contains **all configurable constants** and a small argument
#   parser for the Raspberry Pi side of the Vizzy robotic arm project.
#
# Why this exists
#   - Centralizes hardware pin numbers, servo limits, scanning grid parameters,
#     timing settings, and network configuration in one place.
#   - Makes it easy to tune behavior without having to dig through multiple
#     code files.
#
# How it fits into the project
#   - Every module on the Pi imports values from here to ensure consistent
#     servo control, search patterns, and networking behavior.
#   - The `parse_args()` function allows starting the Pi server in a debug
#     mode that prints detailed diagnostics after centering attempts.
#
# Key points for understanding:
#   - Constants like `SERVO_MIN` and `SERVO_MAX` define **absolute limits** for
#     servo movement in microseconds of PWM signal.
#   - Search grid settings (`SEARCH_*`) define how far and how finely the arm
#     scans when looking for objects.
#   - Timing constants (`POSE_SETTLE_S`, etc.) define pauses and durations for
#     scanning, centering, and servo slews.
#   - Network constants (`LISTEN_HOST`, `LISTEN_PORT`) define how the Pi listens
#     for the laptop connection.
# -----------------------------------------------------------------------------

from __future__ import annotations
import argparse

def parse_args():
    """
    Parse command-line arguments for the RPi server.
    Currently supports:
      --debug : enable extra diagnostic output when centering finishes.
    """
    p = argparse.ArgumentParser(description="RPi Arm Server")
    p.add_argument('--debug', action='store_true',
                   help='Print detailed diagnostics when centering completes')
    return p.parse_args()

# --------------------------
# Servo GPIO pin assignments
# --------------------------
# These are BCM (Broadcom) GPIO numbers used by pigpio.
SERVO_BTM   = 22  # Base rotation servo
SERVO_MID   = 27  # Middle joint servo (unused in current design, reserved)
SERVO_TOP   = 17  # Vertical tilt servo

# --------------------------------------
# Servo PWM pulse width limits (µseconds)
# --------------------------------------
# Defines mechanical safe range for servos.
SERVO_MIN   = 1000   # Fully one direction
SERVO_MAX   = 2000   # Fully opposite direction
SERVO_CENTER= 1500   # Neutral center position

# ---------------------------------
# Search grid sweep configuration
# ---------------------------------
# Offsets keep the scan inside safe limits.
SEARCH_MIN_OFFSET   = 200   # Minimum sweep offset from SERVO_MIN
SEARCH_MAX_OFFSET   = 200   # Maximum sweep offset from SERVO_MAX
SEARCH_H_STEP       = 250   # Step size (µs) for horizontal servo in search
SEARCH_V_STEP       = 100   # Step size (µs) for vertical servo in search

# ---------------------------------------
# Timing settings for scanning/centering
# ---------------------------------------
POSE_SETTLE_S       = 0.35   # Pause after moving to let arm settle before scan
SCAN_DURATION_MS    = 1500   # How long YOLO scan runs at each grid point
CENTER_DURATION_MS  = 3000   # How long to allow laptop to center object
CENTER_EPSILON_PX   = 25     # Pixel error tolerance for centering success
MAX_CENTERS_PER_POSE= 1      # Max number of centering attempts before moving on

# -----------------------------------------
# Target selection thresholds from YOLO scan
# -----------------------------------------
CONF_THRESHOLD      = 0.65   # Minimum average detection confidence to target

# ----------------------
# Networking parameters
# ----------------------
LISTEN_HOST         = '0.0.0.0'  # Listen on all network interfaces
LISTEN_PORT         = 65432      # TCP port to accept laptop connection
