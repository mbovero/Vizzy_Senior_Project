#!/usr/bin/env python3
"""
vizzy manual control – Left Half-Plane Click UI

- Positive X points to the LEFT. We only use/display x ∈ [0, +0.6].
- Click in the canvas to choose (x, y); shows the point and crosshair.
- Sliders: Z (0..0.8 m), Rotation (-180..180 deg).
- Pitch: LEFT (0), UP (~+π/2), DOWN (~-π/2) with rounding to thousandths.
- Claw toggle button: Open/Close.
- Red LEFT-facing semicircle centered at the origin with user-set radius (meters),
  drawn with equal pixel radii so it appears as a perfect half-circle on screen.
- Output line order: x y z pitch claw rotation
- PRINTS to console **only when you press the “S” key** (s or S).
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox

# ---------------------- Tunable Parameters ----------------------
PARAMS = {
    # World-coordinate bounds (meters)
    # Left-half-plane only: x ∈ [0, X_MAX], y ∈ [Y_MIN, Y_MAX]
    "X_MAX": 0.60,                 # meters (positive x to LEFT)
    "Y_RANGE": (-0.60, 0.60),      # meters (min_y, max_y)

    # Canvas pixels
    "CANVAS_W": 600,               # px (horizontal)
    "CANVAS_H": 600,               # px (vertical)

    # Z slider (meters)
    "Z_MIN": 0.00,
    "Z_MAX": 0.80,
    "Z_INIT": 0.25,

    # Rotation slider (degrees)
    "ROT_MIN": -180,
    "ROT_MAX": 180,
    "ROT_INIT": 0,

    # Reach/Arc radius input (meters)
    "R_MIN": 0.05,
    "R_MAX": 0.60,
    "R_INIT": 0.3,

    # Drawing
    "ARC_WIDTH": 3,                # pixels for the red arc
}

# Only three pitch options now: LEFT(0), UP(π/2), DOWN(-π/2)
PITCH_CHOICES = [
    ("←", "left"),
    ("↑", "up"),
    ("↓", "down"),
]


class VizzyManualControl(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("vizzy manual control")

        # Internal state
        self.x_m = 0.0       # meters (>= 0, positive to the LEFT on screen)
        self.y_m = 0.0       # meters
        self.z_m = tk.DoubleVar(value=PARAMS["Z_INIT"])
        self.rot_deg = tk.DoubleVar(value=PARAMS["ROT_INIT"])
        self.pitch_sel = tk.StringVar(value="left")   # 'left' | 'up' | 'down'
        self.claw_open = tk.BooleanVar(value=False)
        self.r_m = tk.DoubleVar(value=PARAMS["R_INIT"])

        self._build_ui()
        self._draw_axes()
        self._update_output()

        # Print ONLY when S is pressed (bind both lowercase and uppercase)
        self.bind_all("<s>", self._on_s_key)
        self.bind_all("<S>", self._on_s_key)

    # ---------- Coordinate Mapping (World <-> Canvas) ----------
    def world_to_canvas(self, x, y):
        X_MAX = PARAMS["X_MAX"]
        Y_MIN, Y_MAX = PARAMS["Y_RANGE"]
        W, H = PARAMS["CANVAS_W"], PARAMS["CANVAS_H"]

        cx = W * (1.0 - (x / X_MAX))              # +x goes LEFT
        cy = H * (1.0 - (y - Y_MIN) / (Y_MAX - Y_MIN))
        return cx, cy

    def canvas_to_world(self, cx, cy):
        X_MAX = PARAMS["X_MAX"]
        Y_MIN, Y_MAX = PARAMS["Y_RANGE"]
        W, H = PARAMS["CANVAS_W"], PARAMS["CANVAS_H"]

        x = X_MAX * (1.0 - (cx / W))
        y = Y_MIN + (1.0 - (cy / H)) * (Y_MAX - Y_MIN)

        # Clamp to left half-plane and Y limits
        x = max(0.0, min(X_MAX, x))
        y = max(Y_MIN, min(Y_MAX, y))
        return x, y

    # ----------------------------- UI -----------------------------
    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Left: Canvas
        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            left,
            width=PARAMS["CANVAS_W"],
            height=PARAMS["CANVAS_H"],
            bg="#101318",
            highlightthickness=1,
            highlightbackground="#3a3f47",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.canvas.bind("<Button-1>", self._on_click)

        # Right: Controls
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="ns")

        # Z Slider
        ttk.Label(right, text="Z Height (m)").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            right, from_=PARAMS["Z_MIN"], to=PARAMS["Z_MAX"],
            orient="horizontal", variable=self.z_m, command=self._on_any_change
        ).grid(row=1, column=0, sticky="ew", pady=(0, 8))

        # Rotation Slider
        ttk.Label(right, text="Rotation (deg)").grid(row=2, column=0, sticky="w")
        ttk.Scale(
            right, from_=PARAMS["ROT_MIN"], to=PARAMS["ROT_MAX"],
            orient="horizontal", variable=self.rot_deg, command=self._on_any_change
        ).grid(row=3, column=0, sticky="ew", pady=(0, 8))

        # Reach Radius Input (meters)
        radius_frame = ttk.Frame(right)
        radius_frame.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(radius_frame, text="Reach Radius (m)").pack(side="left")
        self.r_entry = ttk.Entry(radius_frame, width=8)
        self.r_entry.insert(0, f"{self.r_m.get():.3f}")
        self.r_entry.pack(side="left", padx=(6, 6))
        set_btn = ttk.Button(radius_frame, text="Set", command=self._on_radius_set)
        set_btn.pack(side="left")
        self.r_entry.bind("<Return>", lambda e: self._on_radius_set())

        # Pitch Orientation (LEFT, UP, DOWN)
        ttk.Label(right, text="Pitch Orientation").grid(row=5, column=0, sticky="w", pady=(12, 0))
        pitch_row = ttk.Frame(right)
        pitch_row.grid(row=6, column=0, sticky="w", pady=(2, 8))
        for symbol, value in PITCH_CHOICES:
            ttk.Radiobutton(
                pitch_row, text=symbol, value=value, variable=self.pitch_sel,
                command=self._on_any_change
            ).pack(side="left", padx=2)

        # Claw Toggle Button
        self.claw_button = ttk.Button(
            right, text="Claw: CLOSED", command=self._toggle_claw
        )
        self.claw_button.grid(row=7, column=0, sticky="ew", pady=(8, 8))

        # Current XY readout
        self.xy_label = ttk.Label(right, text="x=0.000  y=0.000", font=("Consolas", 11))
        self.xy_label.grid(row=8, column=0, sticky="w", pady=(4, 2))

        # Output line (label updates live; printing happens on S press)
        ttk.Label(right, text="Output").grid(row=9, column=0, sticky="w", pady=(8, 0))
        self.output_label = ttk.Label(right, text="", font=("Consolas", 11))
        self.output_label.grid(row=10, column=0, sticky="w")

        # Make right column expand a bit
        right.columnconfigure(0, weight=1)

    # ---------------------------- Drawing ----------------------------
    def _draw_axes(self):
        """Draw the left half-plane grid, axes, semicircle, and current point."""
        self.canvas.delete("grid")
        self.canvas.delete("arc")
        self.canvas.delete("point")

        W, H = PARAMS["CANVAS_W"], PARAMS["CANVAS_H"]
        X_MAX = PARAMS["X_MAX"]
        Y_MIN, Y_MAX = PARAMS["Y_RANGE"]

        # Grid lines
        divisions = 4
        for i in range(divisions + 1):
            xw = X_MAX * (i / divisions)
            cx, _ = self.world_to_canvas(xw, 0)
            self.canvas.create_line(cx, 0, cx, H, fill="#2b3240", tags="grid")

        for i in range(divisions + 1):
            yw = Y_MIN + (Y_MAX - Y_MIN) * (i / divisions)
            _, cy = self.world_to_canvas(0, yw)
            self.canvas.create_line(0, cy, W, cy, fill="#2b3240", tags="grid")

        # Axis labels
        self.canvas.create_text(W - 8, H // 2, anchor="e", fill="#8aa0b7",
                                text="x=0 →", tags="grid")
        self.canvas.create_text(12, 14, anchor="w", fill="#8aa0b7",
                                text=f"+x to left, x∈[0,{X_MAX}]", tags="grid")
        self.canvas.create_text(12, 32, anchor="w", fill="#8aa0b7",
                                text=f"y∈[{Y_MIN},{Y_MAX}]", tags="grid")

        # Red left-facing semicircle centered at origin, radius r_m
        self._draw_left_semicircle()

        # Current point
        self._draw_point()

    def _draw_left_semicircle(self):
        """Draw a red left-facing semicircle centered at (0,0) with radius r_m (meters).
        We force equal pixel radii so it renders as a perfect half-circle visually.
        """
        self.canvas.delete("arc")
        W, H = PARAMS["CANVAS_W"], PARAMS["CANVAS_H"]
        X_MAX = PARAMS["X_MAX"]
        Y_MIN, Y_MAX = PARAMS["Y_RANGE"]

        # Center at world origin (0,0)
        cx0, cy0 = self.world_to_canvas(0.0, 0.0)

        # Pixel-per-meter scales
        px_per_m_x = W / X_MAX
        px_per_m_y = H / (Y_MAX - Y_MIN)

        # Use a unified pixel scale so the arc is a circle in screen space
        px_per_m = min(px_per_m_x, px_per_m_y)

        Rm = float(self.r_m.get())
        Rpx = Rm * px_per_m  # same for x and y to look circular

        # Bounding box in canvas coords
        x1, y1 = cx0 - Rpx, cy0 - Rpx
        x2, y2 = cx0 + Rpx, cy0 + Rpx

        # Left-facing semicircle = arc from 90° to 270° (Tk: 0° is right, CCW positive)
        self.canvas.create_arc(
            x1, y1, x2, y2,
            start=90, extent=180,
            style="arc",
            outline="#e74c3c",
            width=PARAMS["ARC_WIDTH"],
            tags="arc"
        )

    def _draw_point(self):
        self.canvas.delete("point")
        cx, cy = self.world_to_canvas(self.x_m, self.y_m)
        r = 5
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                outline="#f2a65a", width=2, tags="point")
        self.canvas.create_line(cx - 10, cy, cx + 10, cy, fill="#f2a65a", tags="point")
        self.canvas.create_line(cx, cy - 10, cx, cy + 10, fill="#f2a65a", tags="point")

    # -------------------------- Event Handlers -------------------------
    def _on_click(self, event):
        xw, yw = self.canvas_to_world(event.x, event.y)
        self.x_m, self.y_m = xw, yw
        self.xy_label.config(text=f"x={self.x_m:.3f}  y={self.y_m:.3f}")
        self._draw_point()
        self._update_output()

    def _toggle_claw(self):
        self.claw_open.set(not self.claw_open.get())
        self.claw_button.config(text=f"Claw: {'OPEN' if self.claw_open.get() else 'CLOSED'}")
        self._update_output()

    def _on_any_change(self, *args):
        self._update_output()

    def _on_radius_set(self):
        txt = self.r_entry.get().strip()
        try:
            val = float(txt)
        except ValueError:
            messagebox.showerror("Invalid radius", "Please enter a numeric radius in meters.")
            return

        # Clamp to allowed range
        val = max(PARAMS["R_MIN"], min(PARAMS["R_MAX"], val))
        self.r_m.set(val)
        self.r_entry.delete(0, tk.END)
        self.r_entry.insert(0, f"{val:.3f}")

        # Redraw with new radius
        self._draw_axes()
        self._update_output()

    def _on_s_key(self, event=None):
        """Print ONLY when 's' or 'S' is pressed."""
        print(self._compose_output())

    # ---------------------------- Helpers ----------------------------
    def _pitch_radians(self) -> float:
        """Map selection to radians, rounding π/2 choices to nearest thousandth."""
        sel = self.pitch_sel.get()
        if sel == "up":
            return round(math.pi / 2, 3)       # +1.571
        if sel == "down":
            return round(-math.pi / 2, 3)      # -1.571
        return 0.0                              # left

    def _compose_output(self) -> str:
        # Order: x y z pitch claw rotation
        x = self.x_m
        y = self.y_m
        z = self.z_m.get()
        pitch_rad = self._pitch_radians()
        claw = "OPEN" if self.claw_open.get() else "CLOSED"
        rot = self.rot_deg.get()
        return (
            f"x={x:.3f}  y={y:.3f}  z={z:.3f}  "
            f"pitch={pitch_rad:.3f} rad  "
            f"claw={claw}  rotation={rot:.1f}°"
        )

    def _update_output(self):
        """Update label only; printing happens on S press."""
        self.output_label.config(text=self._compose_output())


if __name__ == "__main__":
    app = VizzyManualControl()
    app.mainloop()
