# shared/protocol.py

# -----------------------------
# Types (events)
# -----------------------------
TYPE_SCAN_MOVE    = "TYPE_SCAN_MOVE"      # Laptop -> RPi; normalized nudges in [-1,1]
TYPE_OBJ_LOC      = "TYPE_OBJ_LOC"        # RPi -> Laptop; current Cartesian target (x,y,z)
TYPE_STOP         = "TYPE_STOP"           # Laptop -> RPi; shutdown/close
TYPE_CMD_COMPLETE = "CMD_COMPLETE"        # RPi -> Laptop; primitive command completed
TYPE_CMD_ERROR    = "CMD_ERROR"           # RPi -> Laptop; primitive command failed

# -----------------------------
# Commands (requests)
# -----------------------------
CMD_GET_OBJ_LOC   = "CMD_GET_OBJ_LOC"     # Laptop -> RPi; request current Cartesian target

# -----------------------------
# Primitive Commands - Task Execution System
# -----------------------------
# Note: Values are simple names without CMD_ prefix to match LLM output
CMD_GRAB          = "GRAB"                # Laptop -> RPi; close gripper
CMD_RELEASE       = "RELEASE"             # Laptop -> RPi; open gripper
CMD_ROT_YAW       = "ROT_YAW"             # Laptop -> RPi; rotate yaw axis
CMD_ROT_PITCH     = "ROT_PITCH"           # Laptop -> RPi; rotate pitch axis
CMD_MOVE_TO       = "MOVE_TO"             # Laptop -> RPi; move to XYZ coordinates

# High-level commands (expanded by laptop, never sent to RPi)
CMD_PICK          = "PICK"                # LLM output; expanded to primitives
CMD_PLACE         = "PLACE"               # LLM output; expanded to primitives

# -----------------------------
# Payload notes
# -----------------------------
# TYPE_SCAN_MOVE:
#   { "type": TYPE_SCAN_MOVE, "horizontal": float, "vertical": float } # [-1,1], RPi scales
#
# CMD_GET_OBJ_LOC:
#   { "cmd": CMD_GET_OBJ_LOC }
#     - RPi replies with TYPE_OBJ_LOC
#
# TYPE_OBJ_LOC:
#   { "type": TYPE_OBJ_LOC, "x": float, "y": float, "z": float }
#     - Encodes the arm's current commanded Cartesian target; orientation is tracked on the RPi
#
# TYPE_STOP:
#   { "type": TYPE_STOP }
#
# ===== NEW PRIMITIVE COMMAND SYSTEM =====
#
# CMD_MOVE_TO:
#   { "cmd": CMD_MOVE_TO, "x": float, "y": float, "z": float, "pitch": float }
#
# CMD_GRAB:
#   { "cmd": CMD_GRAB }
#
# CMD_RELEASE:
#   { "cmd": CMD_RELEASE }
#
# CMD_ROT_YAW:
#   { "cmd": CMD_ROT_YAW, "angle": float }  # degrees
#
# CMD_ROT_PITCH:
#   { "cmd": CMD_ROT_PITCH, "angle": float }  # degrees
#
# TYPE_CMD_COMPLETE:
#   { "type": TYPE_CMD_COMPLETE, "cmd": str, "status": "success" }
#     - Sent by RPi when a primitive command completes successfully (replaces TYPE_POSE_READY)
#
# TYPE_CMD_ERROR:
#   { "type": TYPE_CMD_ERROR, "cmd": str, "status": "error", "message": str? }
#     - Sent by RPi when a primitive command fails
