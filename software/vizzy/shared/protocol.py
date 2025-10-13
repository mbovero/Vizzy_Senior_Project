# shared/protocol.py

# -----------------------------
# Types (events)
# -----------------------------
TYPE_POSE_READY   = "TYPE_POSE_READY"     # RPi -> Laptop; move complete (optional pose_id)
TYPE_SCAN_MOVE    = "TYPE_SCAN_MOVE"      # Laptop -> RPi; normalized nudges in [-1,1]
TYPE_PWMS         = "TYPE_PWMS"           # RPi -> Laptop; current PWM snapshot
TYPE_STOP         = "TYPE_STOP"           # Laptop -> RPi; shutdown/close
TYPE_CMD_COMPLETE = "CMD_COMPLETE"        # RPi -> Laptop; primitive command completed
TYPE_CMD_ERROR    = "CMD_ERROR"           # RPi -> Laptop; primitive command failed

# -----------------------------
# Commands (requests) - Scan System
# -----------------------------
CMD_GET_PWMS      = "CMD_GET_PWMS"        # Laptop -> RPi; request current PWM pose
CMD_GOTO_PWMS     = "CMD_GOTO_PWMS"       # Laptop -> RPi; absolute move (now supports pose_id)

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
# CMD_GOTO_PWMS:
#   { "cmd": CMD_GOTO_PWMS, "pwm_btm": int, "pwm_top": int, "slew_ms": int, "pose_id": int? }
#     - "pose_id" optional; when present, RPi should echo it in TYPE_POSE_READY
#
# TYPE_POSE_READY:
#   { "type": TYPE_POSE_READY, "pose_id": int? }
#     - "pose_id" optional; echo from CMD_GOTO_PWMS if provided, else omit or 0
#
# TYPE_SCAN_MOVE:
#   { "type": TYPE_SCAN_MOVE, "horizontal": float, "vertical": float } # [-1,1], RPi scales
#
# TYPE_PWMS:
#   { "type": TYPE_PWMS, "pwm_btm": int, "pwm_top": int }
#
# TYPE_STOP:
#   { "type": TYPE_STOP }
#
# ===== NEW PRIMITIVE COMMAND SYSTEM =====
#
# CMD_MOVE_TO:
#   { "cmd": CMD_MOVE_TO, "x": float, "y": float, "z": float }
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
#     - Sent by RPi when a primitive command completes successfully
#
# TYPE_CMD_ERROR:
#   { "type": TYPE_CMD_ERROR, "cmd": str, "status": "error", "message": str? }
#     - Sent by RPi when a primitive command fails
