from __future__ import annotations
import cv2

# TODO: move constants to config vars

# Default HUD text style constants
HUD_FONT   = cv2.FONT_HERSHEY_SIMPLEX  # Font face
HUD_SCALE  = 0.7                       # Relative text size
HUD_THICK  = 2                         # Stroke thickness
HUD_GAP    = 6                         # Vertical gap between lines
HUD_TOP_PAD = 4  # extra pixels to avoid clipping above the first line

def draw_wrapped_text(
    img,
    text: str,
    x: int,
    y: int,
    max_width: int,
    scale: float = HUD_SCALE,
    color=(0, 0, 255),
    thick: int = HUD_THICK
) -> int:
    """
    Draw multi-line, word-wrapped text on an image.

    NOTE: Here, `y` is treated as the *top* margin. The function offsets the
    first line by the font ascent, so nothing is clipped above the image.
    """
    words = text.split()
    if not words:
        return y

    # Font metrics
    (sample_w, sample_h), _ = cv2.getTextSize("Ag", HUD_FONT, scale, thick)
    ascent = sample_h  # OpenCV returns text height; good proxy for ascent
    line_h = int(ascent + HUD_GAP)

    # Start drawing at a baseline that’s safely below the top edge
    baseline_y = int(y + ascent + HUD_TOP_PAD)

    line = ""
    for w in words:
        test = w if not line else (line + " " + w)
        (tw, _), _ = cv2.getTextSize(test, HUD_FONT, scale, thick)

        if tw <= max_width:
            line = test
        else:
            cv2.putText(img, line, (x, baseline_y), HUD_FONT, scale, color, thick, cv2.LINE_AA)
            baseline_y += line_h
            line = w

    if line:
        cv2.putText(img, line, (x, baseline_y), HUD_FONT, scale, color, thick, cv2.LINE_AA)
        baseline_y += line_h

    return baseline_y