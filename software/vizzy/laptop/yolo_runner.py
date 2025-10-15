from __future__ import annotations

from typing import Iterable, Mapping, Sequence, List, Dict

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


def build_allowed_class_ids(
    names: Mapping[int, str] | Sequence[str],
    blacklist: Iterable[str],
) -> List[int]:
    """
    Convert a human-readable blacklist of YOLO class labels into the
    complement of allowed class IDs.
    """
    def _normalize(value: str) -> str:
        return str(value).strip().lower()

    if isinstance(names, Mapping):
        id_to_name: Dict[int, str] = {int(idx): str(label) for idx, label in names.items()}
    else:
        id_to_name = {int(idx): str(label) for idx, label in enumerate(names)}

    all_ids = sorted(id_to_name.keys())
    normalized_labels = {_normalize(label): idx for idx, label in id_to_name.items()}
    blacklisted_ids = set()
    missing_labels = []

    for label in blacklist:
        key = _normalize(label)
        idx = normalized_labels.get(key)
        if idx is None:
            missing_labels.append(label)
            continue
        blacklisted_ids.add(idx)

    if missing_labels:
        print(f"[YOLO] Warning: blacklist labels not found: {', '.join(missing_labels)}")

    return [cls_id for cls_id in all_ids if cls_id not in blacklisted_ids]
