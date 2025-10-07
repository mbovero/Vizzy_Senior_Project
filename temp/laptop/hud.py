from __future__ import annotations
import cv2

# TODO: move constants to config vars

# Default HUD text style constants
HUD_FONT   = cv2.FONT_HERSHEY_SIMPLEX  # Font face
HUD_SCALE  = 0.7                       # Relative text size
HUD_THICK  = 2                         # Stroke thickness
HUD_GAP    = 6                         # Vertical gap between lines

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

    Args:
        img       : OpenCV image (numpy array) to draw onto.
        text      : The text string to render.
        x, y      : Starting coordinates (in pixels) for the first line's baseline.
        max_width : Maximum width (in pixels) allowed for each line before wrapping.
        scale     : Font scaling factor (relative size).
        color     : Text color in BGR tuple format.
        thick     : Stroke thickness for the text.

    Returns:
        int : The new y-coordinate after the last drawn line,
              so the caller can continue drawing below.

    Behavior:
        - Splits the text into words.
        - Iteratively builds lines word-by-word until adding a word
          would exceed `max_width`, at which point the current line is drawn.
        - Moves `y` down after each line by line height + gap.
        - Draws any leftover text after the loop.
    """
    words = text.split()
    if not words:
        return y  # Nothing to draw; return unchanged y position

    # Precompute height of one line (based on a typical text sample)
    ascent = cv2.getTextSize("Ag", HUD_FONT, scale, thick)[0][1]
    line_h = int(ascent + HUD_GAP)

    line = ""  # Current line buffer

    for w in words:
        # Test adding this word to the current line
        test = w if not line else (line + " " + w)
        (tw, _), _ = cv2.getTextSize(test, HUD_FONT, scale, thick)

        if tw <= max_width:
            # Word fits → append to current line
            line = test
        else:
            # Word doesn't fit → draw current line, then start a new one
            cv2.putText(img, line, (x, y), HUD_FONT, scale, color, thick, cv2.LINE_AA)
            y += line_h
            line = w  # Start new line with the current word

    # Draw any remaining text in the buffer
    if line:
        cv2.putText(img, line, (x, y), HUD_FONT, scale, color, thick, cv2.LINE_AA)
        y += line_h

    return y
