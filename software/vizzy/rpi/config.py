# vizzy/rpi/config.py
from __future__ import annotations
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="RPi Arm Server")
    p.add_argument('--debug', action='store_true',
                   help='Print detailed diagnostics when centering completes')
    return p.parse_args()

# Servo pins
SERVO_BTM   = 22
SERVO_MID   = 27
SERVO_TOP   = 17

# Servo pulsewidths (µs)
SERVO_MIN   = 1000
SERVO_MAX   = 2000
SERVO_CENTER= 1500

# Search grid
SEARCH_MIN_OFFSET   = 200   # min = SERVO_MIN + this
SEARCH_MAX_OFFSET   = 200   # max = SERVO_MAX - this
SEARCH_H_STEP       = 250
SEARCH_V_STEP       = 100

# Orchestration timings
POSE_SETTLE_S       = 0.35
SCAN_DURATION_MS    = 1500
CENTER_DURATION_MS  = 3000
CENTER_EPSILON_PX   = 25
MAX_CENTERS_PER_POSE= 1

# Target selection (from scan window)
CONF_THRESHOLD      = 0.65   # average confidence within scan window

# Networking
LISTEN_HOST         = '0.0.0.0'
LISTEN_PORT         = 65432
