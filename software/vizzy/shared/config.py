# vizzy/shared/config.py
# -----------------------------------------------------------------------------
# Single source of truth for both Laptop and RPi configuration.
# Keep protocol identifiers in shared/protocol.py.
# All code should import from this module instead of module-local configs.
# -----------------------------------------------------------------------------

# -----------------------------
# Vision / Camera (Laptop)
# -----------------------------
YOLO_MODEL = "yolo11m-seg.engine"
CAM_INDEX = 4
DISPLAY_SCALE = 1.3
YOLO_VERBOSE = False

# Duration knobs (ms)
SCAN_DURATION_MS   = 1750   # Per-pose scan window
CENTER_DURATION_MS = 3000   # Max time to attempt centering

# Explicit scan gates (selection before attempting to center)
# (Use these to filter scan results; centering thresholds remain separate.)
SCAN_MIN_CONF   = 0.60
SCAN_MIN_FRAMES = 4

# Centering verification thresholds (used during closed-loop centering)
CENTER_CONF        = 0.60     # Per-frame minimum confidence
CENTER_EPSILON_PX  = 25       # Pixel error tolerance for success
CENTER_MOVE_NORM   = 0.035    # Normalized motion stability
CENTER_FRAMES      = 12       # Number of good frames (not necessarily consecutive)
CENTER_DEADZONE    = 30       # HUD/visual deadzone (px); also helps avoid micro-hunting

# Retry / safety
MAX_FAILS_PER_POSE = 2        # Prevent infinite failed centering loop at a single pose

# Object memory
MEM_FILE = "object_memory.json"

# -----------------------------
# Networking
# -----------------------------
# Laptop connects to the Pi at this host/port.
PI_IP   = "10.120.39.241"
PI_PORT = 65432

# RPi server bind (RPi will ignore PI_IP and bind to LISTEN_HOST:LISTEN_PORT)
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 65432

# -----------------------------
# Servos & Sweep (RPi)
# -----------------------------
# Servo GPIO pins (BCM numbering for pigpio)
# Adjust to match your wiring.
SERVO_BTM = 22
SERVO_MID = 27
SERVO_TOP = 17

# Pulse width bounds (µs)
SERVO_MIN    = 1000
SERVO_MAX    = 2000
SERVO_CENTER = 1500

# Normalized scan moves (Laptop -> RPi) scaling to pulse width (µs)
# The laptop sends SCAN_MOVE with values in [-1, 1]; the RPi multiplies by this scale.
MOVE_SCALE_US = 40

# Return-to-pose behavior (laptop -> RPi)
GOTO_POSE_SLEW_MS = 600    # how quickly to snap back to the baseline pose after a centering attempt
RETURN_TO_POSE_DWELL_S = 0.25   # small pause to let the arm settle before the next scan window


# Search grid definition (computed on RPi)
# Range is reduced by MIN/MAX offsets to avoid hard mechanical limits.
SEARCH_MIN_OFFSET = 200  # µs trimmed from the low end
SEARCH_MAX_OFFSET = 200  # µs trimmed from the high end
SEARCH_H_STEP     = 250  # µs horizontal step between poses
SEARCH_V_STEP     = 100  # µs vertical step between rows

# Time to allow the arm to settle mechanically at each pose before scanning
POSE_SETTLE_S = 0.30

# Idle/auto-search behavior (laptop)
IDLE_TIMEOUT_S = 20.0   # seconds of inactivity before auto SEARCH