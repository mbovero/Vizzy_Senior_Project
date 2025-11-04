#!/usr/bin/env python3
"""
vizzy_client_gui.py — Real-time GUI sender: ik x y z pitch yaw O|C

Changes (click/slider/pitch/claw) send automatically (no 'S' needed), debounced to ~20 Hz.
- Z starts at 0.60 m.
- Positive X points LEFT; x ∈ [0, X_MAX], y ∈ [Y_MIN, Y_MAX].
- Sliders: Z (m), Yaw (rad −π..+π). Pitch via radio (0, ±π/2). Claw => O|C.
- Prints exactly what's being sent and any ACK.

Keys:
  R => "rest"
  Q => "quit"
"""

import argparse
import math
import socket
import sys
import tkinter as tk
from tkinter import ttk, messagebox

# ---------------------- Tunables ----------------------
PARAMS = {
    "X_MAX": 0.60,
    "Y_RANGE": (-0.60, 0.60),
    "CANVAS_W": 600,
    "CANVAS_H": 600,

    # Z slider (meters) — start at 0.60
    "Z_MIN": 0.00,
    "Z_MAX": 0.80,
    "Z_INIT": 0.60,

    # Yaw in radians (send as float)
    "YAW_MIN": -math.pi,
    "YAW_MAX": +math.pi,
    "YAW_INIT": 0.0,

    # Reach circle (draw only)
    "R_MIN": 0.05,
    "R_MAX": 0.60,
    "R_INIT": 0.30,

    # Drawing
    "ARC_WIDTH": 3,

    # Live send debounce (ms) — 50ms ≈ 20 Hz
    "LIVE_SEND_DEBOUNCE_MS": 50,
}

PITCH_CHOICES = [("←", "left"), ("↑", "up"), ("↓", "down")]

class VizzyClientGUI(tk.Tk):
    def __init__(self, host: str, port: int):
        super().__init__()
        self.title("vizzy client GUI (real-time)")

        # Connection
        self.host, self.port = host, port
        self.sock: socket.socket | None = None

        # State
        self.x_m = 0.0
        self.y_m = 0.0
        self.z_m = tk.DoubleVar(value=PARAMS["Z_INIT"])
        self.yaw_rad = tk.DoubleVar(value=PARAMS["YAW_INIT"])
        self.pitch_sel = tk.StringVar(value="left")   # left/up/down
        self.claw_open = tk.BooleanVar(value=False)   # False='C', True='O'
        self.r_m = tk.DoubleVar(value=PARAMS["R_INIT"])

        # Debounce handle
        self._send_after_id: str | None = None

        self._build_ui()
        self._draw_axes()
        self._update_previews()

        # Keybinds (rest/quit)
        self.bind_all("<r>", self._on_rest_key)
        self.bind_all("<R>", self._on_rest_key)
        self.bind_all("<q>", self._on_quit_key)
        self.bind_all("<Q>", self._on_quit_key)

        # Connect once
        self.after(50, self._connect_once)

    # ---------------- Networking ----------------
    def _connect_once(self):
        if self.sock is not None:
            return
        try:
            self.status_var.set(f"Connecting to {self.host}:{self.port} ...")
            s = socket.create_connection((self.host, self.port), timeout=3.0)
            s.settimeout(0.25)
            self.sock = s
            self.status_var.set(f"Connected to {self.host}:{self.port}")
            print(f"[NET] Connected to {self.host}:{self.port}")
        except Exception as e:
            self.status_var.set(f"Conn error: {e}")
            print(f"[NET] Connect error: {e}")
            self.after(1500, self._connect_once)

    def _send_line(self, line: str):
        tokens = line.strip().split()
        sanitized = " ".join(tokens)
        payload = sanitized + "\n"

        print(f"SEND {sanitized}")
        print(f"BYTES {payload.encode('utf-8')!r}")

        if self.sock is None:
            self.status_var.set("Not connected; retrying...")
            print("[NET] Not connected; attempting reconnect...")
            self._connect_once()
            return
        try:
            self.sock.sendall(payload.encode("utf-8"))
            try:
                data = self.sock.recv(4096)
                if data:
                    msg = data.decode("utf-8", errors="replace").strip()
                    self.status_var.set(f"ACK: {msg}")
                    print(f"ACK {msg}")
                else:
                    self.status_var.set("Sent (no ACK)")
                    print("ACK <none>")
            except Exception:
                self.status_var.set("Sent (no ACK)")
                print("ACK <none/timeout>")
        except (BrokenPipeError, ConnectionResetError) as e:
            self.status_var.set(f"Send error: {e}; reconnecting...")
            print(f"[NET] Send error: {e}; reconnecting...")
            try:
                if self.sock: self.sock.close()
            except Exception:
                pass
            self.sock = None
            self._connect_once()
        except Exception as e:
            self.status_var.set(f"Send error: {e}")
            print(f"[NET] Send error: {e}")

    # ---------------- Mapping ----------------
    def world_to_canvas(self, x, y):
        X_MAX = PARAMS["X_MAX"]
        Y_MIN, Y_MAX = PARAMS["Y_RANGE"]
        W, H = PARAMS["CANVAS_W"], PARAMS["CANVAS_H"]
        cx = W * (1.0 - (x / X_MAX))            # +x goes LEFT
        cy = H * (1.0 - (y - Y_MIN) / (Y_MAX - Y_MIN))
        return cx, cy

    def canvas_to_world(self, cx, cy):
        X_MAX = PARAMS["X_MAX"]
        Y_MIN, Y_MAX = PARAMS["Y_RANGE"]
        W, H = PARAMS["CANVAS_W"], PARAMS["CANVAS_H"]
        x = X_MAX * (1.0 - (cx / W))
        y = Y_MIN + (1.0 - (cy / H)) * (Y_MAX - Y_MIN)
        x = max(0.0, min(X_MAX, x))
        y = max(Y_MIN, min(Y_MAX, y))
        return x, y

    # ---------------- UI ----------------
    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Left panel: canvas
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

        # Click to set XY
        self.canvas.bind("<Button-1>", self._on_click)
        # Optional: drag to move continuously
        self.canvas.bind("<B1-Motion>", self._on_drag)

        # Right panel: controls
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="ns")
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Z Height (m)").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            right, from_=PARAMS["Z_MIN"], to=PARAMS["Z_MAX"],
            orient="horizontal", variable=self.z_m, command=self._on_any_change
        ).grid(row=1, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(right, text="Yaw (rad)").grid(row=2, column=0, sticky="w")
        ttk.Scale(
            right, from_=PARAMS["YAW_MIN"], to=PARAMS["YAW_MAX"],
            orient="horizontal", variable=self.yaw_rad, command=self._on_any_change
        ).grid(row=3, column=0, sticky="ew", pady=(0, 8))

        radius_frame = ttk.Frame(right)
        radius_frame.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(radius_frame, text="Reach Radius (m)").pack(side="left")
        self.r_entry = ttk.Entry(radius_frame, width=8)
        self.r_entry.insert(0, f"{self.r_m.get():.3f}")
        self.r_entry.pack(side="left", padx=(6, 6))
        ttk.Button(radius_frame, text="Set", command=self._on_radius_set).pack(side="left")
        self.r_entry.bind("<Return>", lambda e: self._on_radius_set())

        ttk.Label(right, text="Pitch").grid(row=5, column=0, sticky="w", pady=(12, 0))
        pitch_row = ttk.Frame(right)
        pitch_row.grid(row=6, column=0, sticky="w", pady=(2, 8))
        for symbol, value in PITCH_CHOICES:
            ttk.Radiobutton(
                pitch_row, text=symbol, value=value, variable=self.pitch_sel,
                command=self._on_any_change
            ).pack(side="left", padx=2)

        self.claw_button = ttk.Button(right, text="Claw: CLOSED (C)", command=self._toggle_claw)
        self.claw_button.grid(row=7, column=0, sticky="ew", pady=(8, 8))

        self.xy_label = ttk.Label(right, text="x=0.000  y=0.000", font=("Consolas", 11))
        self.xy_label.grid(row=8, column=0, sticky="w", pady=(4, 2))

        ttk.Label(right, text="Preview").grid(row=9, column=0, sticky="w", pady=(8, 0))
        self.preview_label = ttk.Label(right, text="", font=("Consolas", 11), justify="left")
        self.preview_label.grid(row=10, column=0, sticky="w")

        ttk.Label(right, text="Exact line (auto-send)").grid(row=11, column=0, sticky="w", pady=(8, 0))
        self.command_label = ttk.Label(right, text="", font=("Consolas", 11), justify="left")
        self.command_label.grid(row=12, column=0, sticky="w")

        self.status_var = tk.StringVar(value="Not connected yet")
        ttk.Label(right, textvariable=self.status_var, foreground="#8aa0b7").grid(row=13, column=0, sticky="w", pady=(12, 0))

    # ---------------- Drawing ----------------
    def _draw_axes(self):
        self.canvas.delete("grid"); self.canvas.delete("arc"); self.canvas.delete("point")
        W, H = PARAMS["CANVAS_W"], PARAMS["CANVAS_H"]
        X_MAX = PARAMS["X_MAX"]; Y_MIN, Y_MAX = PARAMS["Y_RANGE"]

        # Grid
        divisions = 4
        for i in range(divisions + 1):
            xw = X_MAX * (i / divisions)
            cx, _ = self.world_to_canvas(xw, 0)
            self.canvas.create_line(cx, 0, cx, H, fill="#2b3240", tags="grid")
        for i in range(divisions + 1):
            yw = Y_MIN + (Y_MAX - Y_MIN) * (i / divisions)
            _, cy = self.world_to_canvas(0, yw)
            self.canvas.create_line(0, cy, W, cy, fill="#2b3240", tags="grid")

        # Labels
        self.canvas.create_text(W - 8, H // 2, anchor="e", fill="#8aa0b7", text="x=0 →", tags="grid")
        self.canvas.create_text(12, 14, anchor="w", fill="#8aa0b7",
                                text=f"+x to left, x∈[0,{X_MAX}]", tags="grid")
        self.canvas.create_text(12, 32, anchor="w", fill="#8aa0b7",
                                text=f"y∈[{Y_MIN},{Y_MAX}]", tags="grid")

        # Reach semicircle and current point
        self._draw_left_semicircle()
        self._draw_point()

    def _draw_left_semicircle(self):
        self.canvas.delete("arc")
        W, H = PARAMS["CANVAS_W"], PARAMS["CANVAS_H"]
        X_MAX = PARAMS["X_MAX"]; Y_MIN, Y_MAX = PARAMS["Y_RANGE"]
        cx0, cy0 = self.world_to_canvas(0.0, 0.0)
        px_per_m = min(W / X_MAX, H / (Y_MAX - Y_MIN))
        Rpx = float(self.r_m.get()) * px_per_m
        x1, y1 = cx0 - Rpx, cy0 - Rpx
        x2, y2 = cx0 + Rpx, cy0 + Rpx
        self.canvas.create_arc(x1, y1, x2, y2, start=90, extent=180,
                               style="arc", outline="#e74c3c",
                               width=PARAMS["ARC_WIDTH"], tags="arc")

    def _draw_point(self):
        self.canvas.delete("point")
        cx, cy = self.world_to_canvas(self.x_m, self.y_m)
        r = 5
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                outline="#f2a65a", width=2, tags="point")
        self.canvas.create_line(cx - 10, cy, cx + 10, cy, fill="#f2a65a", tags="point")
        self.canvas.create_line(cx, cy - 10, cx, cy + 10, fill="#f2a65a", tags="point")

    # ---------------- Events -> Live send ----------------
    def _on_click(self, event):
        self.x_m, self.y_m = self.canvas_to_world(event.x, event.y)
        self.xy_label.config(text=f"x={self.x_m:.3f}  y={self.y_m:.3f}")
        self._draw_point()
        self._update_previews()
        self._schedule_live_send()

    def _on_drag(self, event):
        self.x_m, self.y_m = self.canvas_to_world(event.x, event.y)
        self.xy_label.config(text=f"x={self.x_m:.3f}  y={self.y_m:.3f}")
        self._draw_point()
        self._update_previews()
        self._schedule_live_send()

    def _toggle_claw(self):
        self.claw_open.set(not self.claw_open.get())
        state = "OPEN (O)" if self.claw_open.get() else "CLOSED (C)"
        self.claw_button.config(text=f"Claw: {state}")
        self._update_previews()
        self._schedule_live_send()

    def _on_any_change(self, *args):
        self._update_previews()
        self._schedule_live_send()

    def _on_radius_set(self):
        txt = self.r_entry.get().strip()
        try:
            val = float(txt)
        except ValueError:
            messagebox.showerror("Invalid radius", "Please enter a numeric radius in meters.")
            return
        val = max(PARAMS["R_MIN"], min(PARAMS["R_MAX"], val))
        self.r_m.set(val)
        self.r_entry.delete(0, tk.END); self.r_entry.insert(0, f"{val:.3f}")
        self._draw_axes()
        self._update_previews()
        # radius is visual only; no send needed

    # ---------------- Debounced live send ----------------
    def _schedule_live_send(self):
        if self._send_after_id is not None:
            try:
                self.after_cancel(self._send_after_id)
            except Exception:
                pass
            self._send_after_id = None
        delay = PARAMS["LIVE_SEND_DEBOUNCE_MS"]
        self._send_after_id = self.after(delay, self._send_latest)

    def _send_latest(self):
        self._send_after_id = None
        self._send_line(self._compose_command_line())

    # ---------------- Helpers ----------------
    def _pitch_radians(self) -> float:
        sel = self.pitch_sel.get()
        if sel == "up":   return round(math.pi / 2, 3)   # +1.571
        if sel == "down": return -1.52  # -1.571
        return 0.1

    def _target_char(self) -> str:
        return "O" if self.claw_open.get() else "C"

    def _compose_command_line(self) -> str:
        """
        EXACT payload:
            ik x y z pitch yaw O|C
        (single spaces; floats to 3 decimals)
        """
        x = f"{self.x_m:.3f}"
        y = f"{self.y_m:.3f}"
        z = f"{self.z_m.get():.3f}"
        pitch = f"{self._pitch_radians():.3f}"
        yaw = f"{self.yaw_rad.get():.3f}"
        tgt = self._target_char()
        return " ".join(["ik", x, y, z, pitch, yaw, tgt])

    def _compose_human_preview(self) -> str:
        return (
            f"x={self.x_m:.3f}  y={self.y_m:.3f}  z={self.z_m.get():.3f}\n"
            f"pitch={self._pitch_radians():.3f} rad  yaw={self.yaw_rad.get():.3f} rad  target={self._target_char()}"
        )

    def _update_previews(self):
        self.preview_label.config(text=self._compose_human_preview())
        self.command_label.config(text=self._compose_command_line())

    # ---------------- Keys: rest/quit ----------------
    def _on_rest_key(self, event=None):
        print("SEND rest"); print(f"BYTES {('rest\\n').encode('utf-8')!r}")
        self._send_line("rest")

    def _on_quit_key(self, event=None):
        print("SEND quit"); print(f"BYTES {('quit\\n').encode('utf-8')!r}")
        self._send_line("quit")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.120.39.241")
    ap.add_argument("--port", type=int, default=65432)
    args = ap.parse_args()
    app = VizzyClientGUI(args.host, args.port)
    app.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
