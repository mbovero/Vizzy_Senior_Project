from __future__ import annotations

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
