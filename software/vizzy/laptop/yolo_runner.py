# vizzy/laptop/yolo_runner.py
# -----------------------------------------------------------------------------
# Purpose
#   This module provides a small wrapper around the Ultralytics YOLO API for
#   initializing and running object detection in the Vizzy project.
#
# Why this exists
#   - Keeps all YOLO-related logic isolated from the rest of the laptop code.
#   - Makes it easy to swap models or change inference behavior without
#     modifying unrelated code.
#
# How it fits into the project
#   - The laptop uses YOLO to detect and localize objects in the camera feed.
#   - Detected objects are sent to the Raspberry Pi for centering and memory
#     storage.
#   - This module abstracts away device selection (CPU/GPU), logging control,
#     and class filtering so the rest of the code just calls `infer_all()`.
#
# Key points for understanding:
#   - `YOLO(engine_path)` loads a model from a `.pt` file or a TensorRT `.engine`
#     file.
#   - Setting `model.overrides['verbose'] = False` silences YOLO’s console spam
#     during inference unless debug mode is enabled.
#   - YOLO’s `stream=True` option yields results for each frame, which lets
#     the laptop process them incrementally.
# -----------------------------------------------------------------------------

from __future__ import annotations
from ultralytics import YOLO
import torch
from typing import Any, Optional

def init_model(engine_path: str, debug: bool) -> Any:
    """
    Load a YOLO model for inference.

    Args:
        engine_path : Path to the YOLO model file (.pt or TensorRT .engine).
        debug       : If True, keep YOLO’s verbose logging enabled.

    Returns:
        A YOLO model object ready for inference.
    """
    # Choose GPU (cuda) if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the YOLO model from the provided path
    model = YOLO(engine_path)

    # Silence unnecessary logs when not debugging
    if not debug:
        try:
            model.overrides['verbose'] = False
        except Exception:
            # Not all YOLO builds support .overrides; ignore errors
            pass
    return model

def infer_all(model, frame, classes: Optional[list[int]] = None):
    """
    Run inference on an image frame and return a generator of results.

    Args:
        model   : The YOLO model instance returned by `init_model`.
        frame   : The image frame (e.g., from OpenCV) to analyze.
        classes : Optional list of class IDs to filter on.
                  If None, detects all classes.

    Returns:
        A generator of YOLO results for each detection in the frame.
    """
    return model(frame, classes=classes, stream=True)

def clear_class_filter(model) -> None:
    """
    Remove any "sticky" class filter that may have been set earlier.

    YOLO's predictor can remember a class filter from a previous call,
    so this ensures future inferences see all classes unless explicitly filtered.
    """
    try:
        if hasattr(model, "predictor") and hasattr(model.predictor, "args"):
            model.predictor.args.classes = None
    except Exception:
        # Fail silently if YOLO internals differ from expected
        pass
