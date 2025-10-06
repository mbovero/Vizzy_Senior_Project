from __future__ import annotations
import json, os, time
from typing import Dict, Any, List

# TODO: remove pwm positions and replace with object XYZ coordinates and claw grasping orientation; 
# also remove avg_conf as I don't see us using this in the future
class ObjectMemory:
    """
    Persistent memory keyed by YOLO class_id (stored as string in JSON).
    Each entry stores:
      - cls_id (int)        : YOLO class ID
      - cls_name (str)      : Human-readable class name
      - pwm_btm (int)       : Servo pulse width for bottom servo
      - pwm_top (int)       : Servo pulse width for top servo
      - last_seen_ts (float): UNIX timestamp when last centered
      - updated_this_session (int): 1 if updated in this session, else 0
    """

    def __init__(self, path: str):
        """
        Create an ObjectMemory instance and immediately load existing data
        from the given file path if it exists.
        """
        self.path = path
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """
        Load JSON data from disk into memory.
        If the file does not exist or fails to parse, start with empty data.
        """
        try:
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            else:
                self.data = {}
        except Exception as e:
            print(f"[Memory] Failed to load {self.path}: {e}")
            self.data = {}

    def save(self) -> None:
        """
        Save current data to disk atomically:
        - Write to a temporary file first
        - Replace the original file to avoid partial writes
        """
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[Memory] Failed to save {self.path}: {e}")

    def reset_session_flags(self) -> None:
        """
        Reset the 'updated_this_session' flag for all entries to 0.
        This should be called at the start of a new search session.
        """
        for k, v in self.data.items():
            v["updated_this_session"] = 0
        self.save()

    def prune_not_updated(self) -> None:
        """
        Remove any entries that were not updated in the current session
        (i.e., 'updated_this_session' != 1).
        """
        before = len(self.data)
        self.data = {
            k: v for k, v in self.data.items()
            if int(v.get("updated_this_session", 0)) == 1
        }
        after = len(self.data)
        if before != after:
            print(f"[Memory] Pruned {before - after} stale entries")
        self.save()

    def update_entry(
        self,
        cls_id: int,
        cls_name: str,
        pwm_btm: int,
        pwm_top: int,
    ) -> None:
        """
        Create or update a memory entry for a given class_id.
        Also records the current timestamp and marks as updated in this session.

        Args:
            cls_id   : YOLO class ID
            cls_name : Human-readable name
            pwm_btm  : Bottom servo PWM position
            pwm_top  : Top servo PWM position
        """
        k = str(int(cls_id))
        now = time.time()
        entry = self.data.get(k, {})
        entry.update({
            "cls_id": int(cls_id),
            "cls_name": cls_name,
            "pwm_btm": int(pwm_btm),
            "pwm_top": int(pwm_top),
            "last_seen_ts": now,
            "updated_this_session": 1
        })
        self.data[k] = entry
        self.save()

    def entries_sorted(self) -> List[dict]:
        """
        Return all entries sorted by integer class_id.
        """
        return [
            self.data[k]
            for k in sorted(self.data.keys(), key=lambda x: int(x))
        ]
