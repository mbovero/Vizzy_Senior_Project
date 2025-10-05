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
      - avg_conf (float, optional): Average detection confidence
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
        avg_conf: float | None = None
    ) -> None:
        """
        Create or update a memory entry for a given class_id.
        Also records the current timestamp and marks as updated in this session.

        Args:
            cls_id   : YOLO class ID
            cls_name : Human-readable name
            pwm_btm  : Bottom servo PWM position
            pwm_top  : Top servo PWM position
            avg_conf : Optional average detection confidence
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
        if avg_conf is not None:
            entry["avg_conf"] = float(avg_conf)
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

    def print_table(self) -> None:
        """
        Print a formatted table of all stored objects to stdout.
        Displays class name, IDs, PWM positions, average confidence,
        and the last seen timestamp in human-readable form.
        """
        entries = self.entries_sorted()
        if not entries:
            print("[Memory] (empty)")
            return

        print("\n[Memory] Stored objects:")
        print("  idx  name            id   pwm_btm  pwm_top   avg_conf   last_seen")
        for idx, e in enumerate(entries):
            ts = e.get("last_seen_ts")
            import time as _t
            ts_s = _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(ts)) if ts else "-"
            avg_conf = e.get("avg_conf")
            avg_conf_s = f"{avg_conf:.2f}" if isinstance(avg_conf, (int, float)) else "-"
            print(f"  {idx:>3}  {e.get('cls_name','?'):<14} {e.get('cls_id'):>3}   "
                  f"{e.get('pwm_btm'):>7}  {e.get('pwm_top'):>7}   {avg_conf_s:>8}   {ts_s}")
        print()
