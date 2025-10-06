# Vizzy — Laptop-Centric Search & Centering (README)

This doc explains how the system works **after** the refactor that pushes scanning/centering to the **laptop** and leaves the **RPi** to run the sweep path and servo control. It also documents the protocol, threading model, config, and how to run + troubleshoot.

---

## TL;DR

- **Laptop (GPU):** YOLO, scan selection, centering loop, and **all memory**.
- **RPi (servos):** Grid sweep timing + motion. Announces when settled, accepts fine nudges, advances only on `POSE_DONE`, and auto-stops at the end.

---

## Repo layout (relevant parts)

```
vizzy/
  laptop/
    __main__.py           # entry; launches laptop/client.py
    client.py             # networking + UI + scan/select/center + memory updates
    scanning.py           # per-pose scan window (YOLO aggregation)
    centering.py          # closed-loop centering (calls SCAN_MOVE deltas)
    memory.py             # JSON store keyed by class id
    hud.py                # small on-frame overlays

  rpi/
    __main__.py           # entry; launches rpi/server.py
    server.py             # single-client JSONL TCP server + sweep worker manager
    dispatch.py           # message router; SCAN_MOVE, GET_PWMS, POSE_DONE, etc.
    search.py             # sweep FSM (move → settle → POSE_READY → wait for POSE_DONE)
    servo.py              # init, goto_pwms, move_servos (scales [-1,1] to µs)
    state.py              # Events/flags + live PWM state

  shared/
    config.py             # single source of truth (camera, thresholds, grid, pins, scaling, ports)
    protocol.py           # UPPERCASE message names (minimal payloads)
    jsonl.py              # newline-delimited JSON helpers
```

---

## Protocol (UPPERCASE, minimal payloads)

**Events (type):**
- `SEARCH { "active": true|false }` – Toggle search.  
  - Laptop→RPi: start/interrupt  
  - RPi→Laptop: default completion at end of grid
- `POSE_READY { "pose_id": int }` – RPi settled at next grid pose.
- `POSE_DONE { "pose_id": int, "status": "SUCCESS"|"SKIP"|"FAIL" }` – Laptop finished that pose.
- `SCAN_MOVE { "horizontal": float, "vertical": float }` – Laptop fine nudges in **[-1,1]** each axis. RPi scales/clamps.
- `PWMS { "pwm_btm": int, "pwm_top": int }` – RPi replies with current servo pulse widths.
- `STOP {}` – Laptop requests clean shutdown of the connection.

**Commands (cmd):**
- `GET_PWMS {}` – Laptop asks RPi for current PWMs (used **after success** while still centered).
- `GOTO_PWMS { "pwm_btm": int, "pwm_top": int, "slew_ms": int }` – Recall absolute move (disabled while searching/centering).

**Removed legacy:** `YOLO_SCAN`, `CENTER_ON`, `YOLO_RESULTS`, `CENTER_DONE`, `CENTER_SNAPSHOT`, and old `MOVE`.

---

## Shared config highlights (`vizzy/shared/config.py`)

- **Vision / camera (laptop):** `YOLO_MODEL`, `CAM_INDEX`, `DISPLAY_SCALE`
- **Scan & centering:**
  - Scan gates: `SCAN_MIN_CONF`, `SCAN_MIN_FRAMES`
  - Centering thresholds: `CENTER_CONF`, `CENTER_EPSILON_PX`, `CENTER_MOVE_NORM`, `CENTER_FRAMES`, `CENTER_DEADZONE`
  - Durations: `SCAN_DURATION_MS`, `CENTER_DURATION_MS`
  - Retry guard: `MAX_FAILS_PER_POSE`
- **Networking:** `PI_IP`, `PI_PORT`, `LISTEN_HOST`, `LISTEN_PORT`
- **Servos & sweep (RPi):**
  - Pins: `SERVO_BTM`, `SERVO_TOP`
  - Range: `SERVO_MIN`, `SERVO_MAX`, `SERVO_CENTER`
  - Nudge scale: `MOVE_SCALE_US`  ← scales `SCAN_MOVE` from `[-1,1]` to ±µs
  - Grid: `SEARCH_MIN_OFFSET`, `SEARCH_MAX_OFFSET`, `SEARCH_H_STEP`, `SEARCH_V_STEP`, `POSE_SETTLE_S`

> Tip: If centering overshoots, reduce `MOVE_SCALE_US` (e.g., 60–150). If servo pins differ from your wiring, change `SERVO_BTM`/`SERVO_TOP`.

---

## Full lifecycle

### 1) Startup
- **RPi**: `python -m vizzy.rpi --debug`
  - Starts server on `LISTEN_HOST:LISTEN_PORT`
  - Connects to `pigpio`, **initializes servos to center** (powered, not limp)
- **Laptop**: `python -m vizzy.laptop`
  - Loads YOLO, opens camera, connects to RPi, shows live detections

### 2) Enter search (press `s` on laptop)
- Laptop sets `search_mode=True`, calls `ObjectMemory.reset_session_flags()`, sends:  
  `SEARCH {active:true}`
- RPi sets `search_active` and starts **sweep worker** (if not running)

### 3) Per-pose handshake
- **RPi sweep worker** (from shared grid):
  1. `goto_pwms(...)` to pose; sleep `POSE_SETTLE_S`
  2. `POSE_READY {pose_id}` and set `centering_active`
- **Laptop main/UI:**
  1. On `POSE_READY`, run `run_scan_window(...)` and pick best class by `SCAN_MIN_*` gates, skipping entries with `updated_this_session == 1`
  2. If none → `POSE_DONE {pose_id, "SKIP"}`
  3. Else → `center_on_class(...)`  
     - per frame, if needed, send `SCAN_MOVE {horizontal,vertical}` in `[-1,1]`
     - success if enough good frames meet `CENTER_*` thresholds within `CENTER_DURATION_MS`
  4. On **success (still centered)**:
     - `GET_PWMS {}` → wait ~0.3 s → expect `PWMS {...}`
     - `ObjectMemory.update_entry(cls_id, cls_name, pwm_btm, pwm_top)` (sets `updated_this_session=1`)
     - `POSE_DONE {pose_id, "SUCCESS"}`
     On **fail**: `POSE_DONE {pose_id, "FAIL"}`
- **RPi sweep worker**:
  - Polls a latch set by `dispatch` to detect `POSE_DONE`
  - Clears `centering_active` and advances to next pose

### 4) Exiting search
- **Default completion** (end of grid): RPi → Laptop: `SEARCH {active:false}`
  - Laptop receiver sets `search_mode=False` and flags completion
  - Laptop main loop prunes: `ObjectMemory.prune_not_updated()`
- **Manual interrupt** (press `s` again): Laptop → RPi: `SEARCH {active:false}`
  - RPi immediately clears `search_active` and `centering_active` (worker exits)
  - Laptop **immediately** prunes (we added this)

### 5) Recall mode
- With search off, press **`m`** to toggle recall
- Use **`a`/`d`** to cycle; on selection, laptop sends `GOTO_PWMS`
- RPi runs `goto_pwms()` unless searching/centering is active

---

## Threading model

### Laptop
- **Main/UI thread**
  - Hotkeys (`s`, `m/a/d`, `q`)
  - Live annotated preview
  - Pose processing (`process_pose`): scan → select → center → `GET_PWMS` → memory → `POSE_DONE`
- **Receiver thread**
  - Reads JSONL and updates:
    - `pose_ready_q: Queue[int]` for `POSE_READY`
    - `pwms_event` + `pwms_payload` for `PWMS`
    - `sweep_completed_flag` (Event) for `SEARCH {active:false}`
  - Also flips `search_mode=False` on sweep completion so recall is unblocked

### RPi
- **Server thread (single client)**
  - Accepts client, calls `recv_lines`, routes to `dispatch.process_messages(...)`
  - Starts sweep worker when `search_active` turns on
- **Sweep worker thread**
  - Runs grid; per pose: move → settle → `POSE_READY` → enable `centering_active` → **wait for `POSE_DONE`** → advance
  - Sends `SEARCH {active:false}` once when the grid finishes
- **Shared state (`rpi/state.py`)**
  - `search_active`, `centering_active` (Events)
  - `current_horizontal`, `current_vertical` live PWMs

---

## Running

**RPi:**
```bash
# pigpio daemon must be running
sudo pigpiod

# start server
python -m vizzy.rpi --debug
```

**Laptop:**
```bash
python -m vizzy.laptop
# hotkeys:
#  s = start/stop search (toggle)
#  m = toggle recall mode (when not searching)
#  a/d = cycle recall entries
#  q = quit
```

---

## Tuning & troubleshooting

- **No motion at boot:** Verify RPi logs “Servos initialized to center.” Confirm `SERVO_BTM`/`SERVO_TOP` match wiring.
- **Overshoot or wrong direction while centering:**
  - Lower `MOVE_SCALE_US` (e.g., 60–150) in shared config.
  - If image mirrored or axes swapped, we can add `INVERT_H`, `INVERT_V`, `SWAP_AXES` toggles in `shared/config.py` and apply in `servo.move_servos()` (ask and we’ll wire them in).
- **Recall locked out after sweep:** Laptop must clear `search_mode=False` on RPi’s `SEARCH {active:false}` (already patched).
- **Memory not updating:** Ensure `PWMS` is received before `POSE_DONE` is sent. We wait ~0.3 s; if needed, increase that timeout slightly.

---

## Design choices (why this split?)

- **Low chatter:** RPi no longer instructs the laptop per step. It just announces poses and accepts nudges.
- **Deterministic sweep:** RPi keeps timing smooth and owns default completion.
- **Tight control loop:** Laptop runs centering at camera/frame rate and emits simple normalized deltas; RPi handles scaling and safety.
- **Robustness:** Receiver thread keeps sockets responsive; sweep worker never blocks on I/O.

---

## Quick API facts (for code readers)

- Laptop → RPi nudge:
  ```python
  send_json(sock, {"type": "SCAN_MOVE", "horizontal": dx, "vertical": dy})
  ```
- Laptop → RPi finish:
  ```python
  send_json(sock, {"type": "POSE_DONE", "pose_id": pid, "status": "SUCCESS"})
  ```
- Laptop → RPi get PWMs:
  ```python
  send_json(sock, {"cmd": "GET_PWMS"})
  # wait for {"type":"PWMS","pwm_btm":..., "pwm_top":...}
  ```
- RPi → Laptop pose:
  ```python
  send_json(conn, {"type": "POSE_READY", "pose_id": pid})
  ```
- RPi end-of-grid:
  ```python
  send_json(conn, {"type": "SEARCH", "active": False})
  ```
