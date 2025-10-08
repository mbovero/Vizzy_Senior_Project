# shared/protocol.py

# -----------------------------
# Types (events)
# -----------------------------
TYPE_POSE_READY   = "TYPE_POSE_READY"     # RPi -> Laptop; move complete (optional pose_id)
TYPE_SCAN_MOVE    = "TYPE_SCAN_MOVE"      # Laptop -> RPi; normalized nudges in [-1,1]
TYPE_PWMS         = "TYPE_PWMS"           # RPi -> Laptop; current PWM snapshot
TYPE_STOP         = "TYPE_STOP"           # Laptop -> RPi; shutdown/close

# TODO Deprecated (kept for compatibility during transition; will be removed)
TYPE_SEARCH       = "TYPE_SEARCH"         # (deprecated) start/complete search cycle
TYPE_POSE_DONE    = "TYPE_POSE_DONE"      # (deprecated) per-pose status

# -----------------------------
# Commands (requests)
# -----------------------------
CMD_GET_PWMS      = "CMD_GET_PWMS"        # Laptop -> RPi; request current PWM pose
CMD_GOTO_PWMS     = "CMD_GOTO_PWMS"       # Laptop -> RPi; absolute move (now supports pose_id)

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
