# vizzy/laptop/hud.py
from __future__ import annotations
import cv2

HUD_FONT   = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE  = 0.7
HUD_THICK  = 2
HUD_GAP    = 6

def draw_wrapped_text(img, text: str, x: int, y: int, max_width: int,
                      scale: float = HUD_SCALE, color=(0,0,255), thick: int = HUD_THICK) -> int:
    """Word-wrap text to fit max_width at (x,y); returns new y."""
    words = text.split()
    if not words: return y
    line = ""
    ascent = cv2.getTextSize("Ag", HUD_FONT, scale, thick)[0][1]
    line_h = int(ascent + HUD_GAP)
    for w in words:
        test = w if not line else (line + " " + w)
        (tw, _), _ = cv2.getTextSize(test, HUD_FONT, scale, thick)
        if tw <= max_width:
            line = test
        else:
            cv2.putText(img, line, (x, y), HUD_FONT, scale, color, thick, cv2.LINE_AA)
            y += line_h
            line = w
    if line:
        cv2.putText(img, line, (x, y), HUD_FONT, scale, color, thick, cv2.LINE_AA)
        y += line_h
    return y
