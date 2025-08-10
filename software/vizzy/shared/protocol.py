# vizzy/shared/protocol.py
from __future__ import annotations

# Message "type" (events/data)
TYPE_MOVE           = "move"
TYPE_SEARCH         = "search"
TYPE_STOP           = "stop"
TYPE_YOLO_RESULTS   = "YOLO_RESULTS"
TYPE_CENTER_DONE    = "CENTER_DONE"
TYPE_CENTER_SNAPSHOT= "CENTER_SNAPSHOT"

# Message "cmd" (requests)
CMD_YOLO_SCAN       = "YOLO_SCAN"
CMD_CENTER_ON       = "CENTER_ON"
CMD_GOTO_PWMS       = "GOTO_PWMS"
