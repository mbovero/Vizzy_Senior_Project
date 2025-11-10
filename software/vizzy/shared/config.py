# vizzy/shared/config.py
# -----------------------------------------------------------------------------
# Single source of truth for both Laptop and RPi configuration.
# Keep protocol identifiers in shared/protocol.py.
# All code should import from this module instead of module-local configs.
# -----------------------------------------------------------------------------

from pathlib import Path

PKG_ROOT   = Path(__file__).resolve().parent.parent
LAPTOP_DIR = PKG_ROOT / "laptop"

# -----------------------------
# Vision / Camera (Laptop)
# -----------------------------
YOLO_MODEL = str(LAPTOP_DIR / "yolo11m-seg.engine")
CAM_INDEX = 4
DISPLAY_SCALE = 1.3
YOLO_VERBOSE = False
OBJ_BLACKLIST = [
    "person",
    "chair",
    "tv",
    "laptop",
    "keyboard",
    "dining table",
]

# Duration knobs (ms) - reduced for faster iteration
SCAN_DURATION_MS   = 500   # Per-pose scan window (reduced from 1750)
CENTER_DURATION_MS = 10000  # Max time to attempt centering (10 seconds)

# Centering movement calculation (matching object_centering.py)
PIXEL_TO_MM = 1.0 / 2.90  # mm per pixel
WORKING_DISTANCE_MM = 600.0  # mm (typical working distance for arm operations)
MOVEMENT_SCALE_FACTOR = 1.2  # Scale factor for movement calculation

# Explicit scan gates (selection before attempting to center)
# (Use these to filter scan results; centering thresholds remain separate.)
SCAN_MIN_CONF   = 0.80  # confidence threshold for object detection
SCAN_MIN_FRAMES = 2  

# Centering verification thresholds (used during closed-loop centering)
CENTER_CONF        = 0.40     # Per-frame minimum confidence
CENTER_EPSILON_PX  = 25       # Pixel error tolerance for success
CENTER_MOVE_NORM   = 0.035    # Normalized motion stability
CENTER_FRAMES      = 12       # Number of good frames (not necessarily consecutive)
CENTER_MIN_MOVEMENT_MM = 3.0  # Minimum movement threshold: if movement < 5mm, consider centered
CENTER_MEASURE_WAIT_TIME_S = 1.0  # Time to wait after movement command before measuring (arm must be stopped)
CENTER_TIMEOUT_S = 3.0  # Maximum time per movement cycle before canceling (seconds). Timer resets after each movement toward object.

# Retry / safety
MAX_FAILS_PER_POSE = 2        # Prevent infinite failed centering loop at a single pose

# Object memory
MEM_FILE = str(LAPTOP_DIR / "object_memory.json")

# Valid objects to center on during search (fork, cup, and knife)
SEARCH_VALID_CLASS_NAMES = ["fork", "spoon", "cup", "knife"]  # Only center on these objects

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
# Cartesian Search & Arm Geometry
# -----------------------------
# Workspace bounds used to build the search path (millimetres, laptop-side)
# NOTE: Commands are sent to server in meters (mm/1000.0)
# The search path is now built from SEARCH_PATH_POINTS below (all points are valid)
SEARCH_X_MIN_MM = 0.0    # Starting x: 0mm (x cannot be negative)
SEARCH_X_MAX_MM = 500.0  # Maximum x: 500mm (constrained by max magnitude)
SEARCH_X_STEP_MM = 50.0  # Step size: 50mm (reduced by 50% from 100mm)

SEARCH_Y_MIN_MM = -500.0  # Minimum y: -500mm (y can be negative)
SEARCH_Y_MAX_MM = 500.0   # Maximum y: 500mm (constrained by max magnitude)
SEARCH_Y_STEP_MM = 50.0   # Step size: 50mm (reduced by 50% from 100mm)

# Z is fixed at 275mm (0.275m) for the entire scan cycle
SEARCH_Z_FIXED_MM = 275.0
# Legacy z range (not used when SEARCH_Z_FIXED_MM is set)
SEARCH_Z_MIN_MM = 275.0
SEARCH_Z_MAX_MM = 275.0
SEARCH_Z_STEP_MM = 50.0

# Default pitch for search poses (degrees)
SEARCH_PITCH_DEG = 5

# Explicit search path points (x_mm, y_mm) - all valid poses the arm can move to
SEARCH_PATH_POINTS = [
    
    # Pass 1: negative diagonal (x+25, y+25)
    (175.0, -125.0),
    (225.0, -175.0),
    (275.0, -225.0),
    (325.0, -275.0),
    (375.0, -325.0),

    # Pass 1.5: between Pass 1 and Pass 2, reversed (x+25, y+25)
    (400.0, -150.0),
    (350.0, -125.0),
    (300.0, -100.0),
    (250.0, -75.0),
    (200.0, -50.0),

    # Pass 2: middle row (x +25, y unchanged)
    (225.0, 0.0),
    (275.0, 0.0),
    (325.0, 0.0),
    (375.0, 0.0),
    (425.0, 0.0),

    # Pass 2.5: between Pass 2 and Pass 3, reversed (x+25, y+25)
    (400.0, 200.0),
    (350.0, 175.0),
    (300.0, 150.0),
    (250.0, 125.0),
    (200.0, 100.0),

    # Pass 3: positive diagonal (x+25, y+25)
    (175.0, 175.0),
    (225.0, 225.0),
    (275.0, 275.0),
    (325.0, 325.0),
    (375.0, 375.0),

    
    
    (250.0, 0.0),  
    

]

# Relative nudge scaling (converted from normalized [-1,1] commands on the RPi)
SCAN_NUDGE_STEP_MM = 5.0

# Target settle/dwell times (reduced for faster iteration)
MOVE_SETTLE_S = .05          # allow time after a commanded move before accepting nudges (reduced from 0.30)
RETURN_TO_POSE_DWELL_S = 0.10 # dwell after returning to baseline before next scan window (reduced from 0.25)
POSE_CV_DELAY_S = 0         # extra delay at each pose to give CV model time to identify objects

# Physical servo PWM bounds for the new arm (documentation for IK output clamping)
SERVO_PITCH_CENTER_US = 1500
SERVO_PITCH_MIN_US    = 1000
SERVO_PITCH_MAX_US    = 2000

SERVO_YAW_CENTER_US   = 1700
SERVO_YAW_MIN_US      = 1000
SERVO_YAW_MAX_US      = 2000

SERVO_CLAW_CENTER_US  = 1500
SERVO_CLAW_MIN_US     = 1000
SERVO_CLAW_MAX_US     = 2000

# Idle/auto-search behavior (laptop)
IDLE_TIMEOUT_S = 20.0   # seconds of inactivity before auto SEARCH

# -----------------------------
# LLM / Semantic Enrichment
# -----------------------------
# OpenAI model for semantic enrichment of captured objects
IMAGE_PROCESS_MODEL = "gpt-5-nano"

# OpenAI model for task scheduling / planning
TASK_SCHEDULER_MODEL = "gpt-5-nano"

# Number of concurrent LLM worker threads for semantic enrichment
LLM_WORKERS = 5

# Directory to save captured images (for LLM processing)
IMAGE_DIR = str (LAPTOP_DIR /"captured_images")

# Timeout for waiting for LLM enrichment to complete before processing user queries (seconds)
# Set to None to wait forever, or a number for timeout
LLM_COMPLETION_TIMEOUT = 30.0

# Operation mode flags (can be set via interactive menu or manually)
SKIP_TO_TASK_SCHEDULER = False  # Skip to task scheduler only (testing mode)
SKIP_SCAN_CYCLE = False          # Skip scan cycle, use existing memory
SKIP_SEMANTIC_ENRICHMENT = False # Skip LLM semantic enrichment

# Output file for LLM task scheduler results (for verification)
TASK_SCHEDULER_OUTPUT_FILE = str(LAPTOP_DIR / "task_scheduler_output.json")

# -----------------------------
# Task Execution - Rest Position
# -----------------------------
# Rest/home position for robotic arm (all coordinates in millimeters)
REST_POSITION = [250, 0, 300]    # [x_mm, y_mm, z_mm]
REST_YAW_ANGLE = 0.0             # degrees
REST_PITCH_ANGLE = 0.0           # degrees

# Vertical offset for approach/retract moves (millimeters)
APPROACH_OFFSET_Z = 350.0        # mm above object for safe approach

# Timeout for primitive command execution (seconds)
PRIMITIVE_CMD_TIMEOUT = 30.0

# -----------------------------
# Orientation Calculation
# -----------------------------
# Note: Orientation is calculated from frames collected during centering (CENTER_FRAMES)
# No additional frame capture needed - reuses centering frames for efficiency

# -----------------------------
# Camera to Gripper Offset
# -----------------------------
# Physical offset from camera center to gripper center (millimeters)
# When camera is centered on object, gripper is offset by this distance along the radius
CAMERA_TO_GRIPPER_OFFSET_MM = 34.5  # mm
