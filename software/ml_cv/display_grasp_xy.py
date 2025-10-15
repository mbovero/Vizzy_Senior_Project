#!/usr/bin/env python3
"""
display_grasp_xy.py

Show images from --images and overlay red dots from semantics.grasp_xy
in --memory (object_memory.json). Navigate with A/← and D/→. Quit with Q/ESC.
"""

from __future__ import annotations
import os
# Force Qt to X11/XWayland to avoid Wayland plugin errors
if os.environ.get("QT_QPA_PLATFORM", "").lower() == "wayland" or "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import argparse
import json
import signal
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SCRIPT_DIR = Path(__file__).resolve().parent
SOFTWARE_DIR = SCRIPT_DIR.parent
DEFAULT_IMAGES_DIR = SOFTWARE_DIR / "vizzy" / "laptop" / "captured_images"
DEFAULT_MEMORY_PATH = SOFTWARE_DIR / "vizzy" / "laptop" / "object_memory.json"

def on_sigint(signum, frame):
    print("\n[info] Ctrl+C — closing viewer.")
    try:
        cv2.destroyAllWindows()
    finally:
        raise SystemExit(0)

signal.signal(signal.SIGINT, on_sigint)

def build_grasp_index(memory_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    idx: Dict[str, List[Tuple[int, int]]] = {}
    if not memory_path.exists():
        print(f"[warn] memory file not found: {memory_path}")
        return idx
    with open(memory_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for obj in (data.get("objects") or {}).values():
        fname = Path(obj.get("image_path", "")).name
        xy = (obj.get("semantics") or {}).get("grasp_xy")
        if fname and isinstance(xy, list) and len(xy) == 2:
            try:
                idx.setdefault(fname, []).append((int(xy[0]), int(xy[1])))
            except Exception:
                pass
    return idx

def scan_images(folder: Path) -> List[Path]:
    if not folder.exists():
        print(f"[warn] images folder not found: {folder}")
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

def draw_hud(img: np.ndarray, text: str, scale: float = 0.55, pad: int = 8, thick: int = 1):
    """Simple translucent header ribbon with text (smaller by default)."""
    h, w = img.shape[:2]
    bar_h = int(24 * scale) + 12
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    # Avoid clipping: y is top margin; offset by approximate ascent
    ascent = int(16 * scale) + 8
    cv2.putText(img, text, (pad, ascent), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thick, cv2.LINE_AA)

def show_with_grasps(img_path: Path, grasps: List[Tuple[int, int]] | None, scale: float,
                     hud_scale: float, hud_thick: int) -> np.ndarray:
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        vis = np.zeros((300, 600, 3), dtype=np.uint8)
        draw_hud(vis, f"Failed to read: {img_path.name}", scale=hud_scale, thick=hud_thick)
        return vis

    vis = bgr.copy()
    h, w = vis.shape[:2]
    if grasps:
        for (x, y) in grasps:
            x0 = max(0, min(w - 1, int(x)))
            y0 = max(0, min(h - 1, int(y)))
            cv2.circle(vis, (x0, y0), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.drawMarker(vis, (x0, y0), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
        label = f"{img_path.name} | {len(grasps)} grasp point(s)"
    else:
        label = f"{img_path.name} | no grasp_xy in memory"
    draw_hud(vis, label, scale=hud_scale, thick=hud_thick)

    if scale != 1.0:
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return vis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images",
        type=str,
        default=str(DEFAULT_IMAGES_DIR),
        help="Folder containing images (default: vizzy/laptop/captured_images)"
    )
    ap.add_argument(
        "--memory",
        type=str,
        default=str(DEFAULT_MEMORY_PATH),
        help="Path to object memory JSON (default: vizzy/laptop/object_memory.json)"
    )
    ap.add_argument("--scale", type=float, default=1.0, help="Display scale (e.g., 0.75)")
    ap.add_argument("--hud-scale", type=float, default=0.55, help="HUD text scale (smaller default)")
    ap.add_argument("--hud-thick", type=int, default=1, help="HUD text thickness")
    args = ap.parse_args()

    img_dir = Path(args.images)
    mem_path = Path(args.memory)
    scale = float(args.scale)
    hud_scale = float(args.hud_scale)
    hud_thick = int(args.hud_thick)

    grasp_index = build_grasp_index(mem_path)
    images = scan_images(img_dir)
    if not images:
        print(f"[info] No images found in {img_dir.resolve()}")
        return

    idx = 0
    win = "Grasp Viewer"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    KEY_LEFTS  = {ord('a'), ord('A'), 81, 2424832}   # 'a', Left arrow
    KEY_RIGHTS = {ord('d'), ord('D'), 83, 2555904}   # 'd', Right arrow
    KEY_QUIT   = {ord('q'), ord('Q'), 27}            # 'q', ESC

    # render once before entering the loop
    vis = show_with_grasps(images[idx], grasp_index.get(images[idx].name), scale, hud_scale, hud_thick)
    cv2.imshow(win, vis)

    while True:
        key = cv2.waitKey(20) & 0xFF  # <-- correct mask; -1 stays 255 here on many backends

        # Allow closing by window close button
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            break

        if key in KEY_QUIT:
            break
        elif key in KEY_RIGHTS:
            idx = (idx + 1) % len(images)
            vis = show_with_grasps(images[idx], grasp_index.get(images[idx].name), scale, hud_scale, hud_thick)
            cv2.imshow(win, vis)
        elif key in KEY_LEFTS:
            idx = (idx - 1) % len(images)
            vis = show_with_grasps(images[idx], grasp_index.get(images[idx].name), scale, hud_scale, hud_thick)
            cv2.imshow(win, vis)
        # else: no-op on all other keys / no key

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
