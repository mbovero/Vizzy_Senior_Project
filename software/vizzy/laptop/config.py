# -----------------------------------------------------------------------------
# vizzy/laptop/config.py
#
# Purpose:
#   Defines command-line argument parsing and helper functions for resolving
#   YOLO class IDs based on either user-provided IDs or human-readable names.
#
# Why this exists:
#   - The laptop client needs a flexible way to configure connection settings,
#     camera index, YOLO engine path, debug flags, and object targeting
#     parameters without hardcoding them in the source code.
#   - Also provides a way to map between user-friendly object class names
#     (e.g., "bottle") and the integer class IDs used internally by YOLO.
#
# How it fits into the bigger picture:
#   - Called from the laptop’s `main` entry point to parse CLI arguments.
#   - Ensures the robotic arm can be run with different configurations for
#     different setups, datasets, or testing conditions.
#   - The `resolve_class_id_from_name()` helper supports both workflows:
#       • User specifies class ID directly
#       • User specifies class name, which gets resolved via YOLO's `model.names`
#
# Notes for new developers:
#   - Uses Python’s built-in `argparse` for command-line parsing.
#   - The `names_map` argument in `resolve_class_id_from_name()` can be either:
#       • A dict {class_id: name}
#       • A list where index = class_id, value = name
# -----------------------------------------------------------------------------

from __future__ import annotations
import argparse
from typing import Optional

def make_arg_parser() -> argparse.ArgumentParser:
    """
    Create and return an ArgumentParser configured for the laptop client.

    The arguments include:
        --class-id       : Target YOLO class ID (-1 = ALL classes)
        --class-name     : Target YOLO class name (overrides --class-id if valid)
        --ip             : IP address of the Raspberry Pi server
        --port           : TCP port of the Raspberry Pi server
        --engine         : Path to YOLO engine file (.engine format)
        --camera-index   : OpenCV camera index (integer, often 0–N)
        --debug          : Enable verbose debug output
        --mem-file       : Path to persistent object memory file
        --display-scale  : Scale factor for displayed frames
    """
    p = argparse.ArgumentParser(description="Laptop client for YOLO-driven robotic arm")
    p.add_argument('--class-id', type=int, default=-1)
    p.add_argument('--class-name', type=str, default=None)
    p.add_argument('--ip', type=str, default='192.168.1.30')
    p.add_argument('--port', type=int, default=65432)
    p.add_argument('--engine', type=str, default='yolo11m-seg.engine')
    p.add_argument('--camera-index', type=int, default=4)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--mem-file', type=str, default='object_memory.json')
    p.add_argument('--display-scale', type=float, default=1.3)
    return p

def resolve_class_id_from_name(
    names_map,
    class_id: int,
    class_name: Optional[str]
) -> int:
    """
    Determine the YOLO class ID to target based on the provided arguments.

    Resolution logic:
        1. If `class_name` is provided:
            - Search for a case-insensitive match in `names_map`.
            - Return the corresponding integer ID if found.
            - If not found, print a warning and return -1 (ALL).
        2. If `class_id` is >= 0:
            - Return `class_id` directly.
        3. Otherwise:
            - Return -1 (ALL classes).

    Args:
        names_map : Mapping of YOLO class IDs to names.
                    Can be a dict {id: name} or a list where index = id.
        class_id  : Numeric ID specified by the user (may be -1).
        class_name: Optional string name specified by the user.

    Returns:
        int: Resolved class ID (-1 means no filtering; detect all classes).
    """
    if class_name:
        lname = class_name.strip().lower()
        try:
            if isinstance(names_map, dict):
                # Search in dict form: {id: name}
                for k, v in names_map.items():
                    if str(v).lower() == lname:
                        return int(k)
            else:
                # Search in list form: index = id
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
