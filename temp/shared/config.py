# Computer Vision parameters
YOLO_MODEL = 'yolo11m-seg.engine'
CAM_INDEX = 4

# Raspberry Pi connections parameters
PI_IP = '192.168.1.30'
PI_PORT = 65432

MEM_FILE = 'object_memory.json'

DISPLAY_SCALE = 1.3

# Object centering parameters
CENTER_DEADZONE = 30
SERVO_SPEED = 0.2 # TODO: this should be baked into motor/servo control API; remove later on
CENTER_CONF = 0.60          # per-frame minimum confidence
CENTER_MOVE_NORM = 0.035    # normalized motion stability
CENTER_FRAMES = 12          # number of “good” frames (not necessarily consecutive)


SCAN_DURATION_MS    = 1750   # How long YOLO scan runs at each grid point
CENTER_DURATION_MS  = 3000   # How long to allow laptop to center object
