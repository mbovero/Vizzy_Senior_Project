# vizzy/laptop/yolo_runner.py
from __future__ import annotations
from ultralytics import YOLO
import torch
from typing import Any, Optional

def init_model(engine_path: str, debug: bool) -> Any:
    """Load YOLO model; quiet logs if not in debug."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = YOLO(engine_path)
    if not debug:
        # Silence Ultralytics verbose logs unless debugging
        try:
            model.overrides['verbose'] = False
        except Exception:
            pass
    return model

def infer_all(model, frame, classes: Optional[list[int]] = None):
    """
    Run one inference call returning a generator of results.
    classes=None → ALL classes, otherwise [ids].
    """
    return model(frame, classes=classes, stream=True)

def clear_class_filter(model) -> None:
    """Clear any sticky class filter in the predictor (defensive)."""
    try:
        if hasattr(model, "predictor") and hasattr(model.predictor, "args"):
            model.predictor.args.classes = None
    except Exception:
        pass
