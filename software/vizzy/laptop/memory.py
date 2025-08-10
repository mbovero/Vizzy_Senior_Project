# vizzy/laptop/memory.py
from __future__ import annotations
import json, os, time
from typing import Dict, Any, List

class ObjectMemory:
    """
    Persistent memory keyed by class_id (as string).
    Stores: cls_id, cls_name, pwm_btm, pwm_top, last_seen_ts, updated_this_session, avg_conf?
    """
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
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
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[Memory] Failed to save {self.path}: {e}")

    def reset_session_flags(self) -> None:
        for k, v in self.data.items():
            v["updated_this_session"] = 0
        self.save()

    def prune_not_updated(self) -> None:
        before = len(self.data)
        self.data = {k: v for k, v in self.data.items()
                     if int(v.get("updated_this_session", 0)) == 1}
        after = len(self.data)
        if before != after:
            print(f"[Memory] Pruned {before - after} stale entries")
        self.save()

    def update_entry(self, cls_id: int, cls_name: str,
                     pwm_btm: int, pwm_top: int, avg_conf: float | None = None) -> None:
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
        return [self.data[k] for k in sorted(self.data.keys(), key=lambda x: int(x))]

    def print_table(self) -> None:
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
