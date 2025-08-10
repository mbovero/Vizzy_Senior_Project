# vizzy/laptop/config.py
from __future__ import annotations
import argparse
from typing import Optional

def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Laptop client for YOLO-driven robotic arm")
    p.add_argument('--class-id', type=int, default=-1)
    p.add_argument('--class-name', type=str, default=None)
    p.add_argument('--ip', type=str, default='192.168.1.30')
    p.add_argument('--port', type=int, default=65432)
    p.add_argument('--engine', type=str, default='yolo11m-seg.engine')
    p.add_argument('--camera-index', type=int, default=4)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--mem-file', type=str, default='object_memory.json')
    p.add_argument('--display-scale', type=float, default=1.5)
    return p

def resolve_class_id_from_name(names_map, class_id: int, class_name: Optional[str]) -> int:
    """
    Return resolved class id:
      - if class_name provided and found in model.names → that id
      - elif class_id >= 0 → that id
      - else -1 (ALL)
    """
    if class_name:
        lname = class_name.strip().lower()
        try:
            if isinstance(names_map, dict):
                for k, v in names_map.items():
                    if str(v).lower() == lname:
                        return int(k)
            else:
                for i, v in enumerate(names_map):
                    if str(v).lower() == lname:
                        return int(i)
        except Exception:
            pass
        print(f"[WARN] Class name '{class_name}' not found; using ALL CLASSES.")
        return -1
    if class_id is not None and class_id >= 0:
        return int(class_id)
    return -1
