#!/usr/bin/env python3
"""
Simplified Laptop-Style Object Centering (standalone)
- Continuously reads from USB camera
- Runs YOLO model
- Computes centering offsets (px → mm) and mapped movements (x/y) exactly like laptop logic
- Displays centers, lines, movement values, confidence, and FPS in an OpenCV window
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ----------------------------- config wiring -------------------------------- #
try:
	# Prefer shared laptop config if available
	from software.vizzy.shared import config as C  # type: ignore
	CAM_INDEX = int(getattr(C, "CAM_INDEX", 4))
	YOLO_MODEL_PATH = str(getattr(C, "YOLO_MODEL", "software/vizzy/laptop/yolo11m-seg.engine"))
	PIXEL_TO_MM = float(getattr(C, "PIXEL_TO_MM", (1.0 / 2.90)))
	WORKING_DISTANCE_MM = float(getattr(C, "WORKING_DISTANCE_MM", 600.0))
	MOVEMENT_SCALE_FACTOR = float(getattr(C, "MOVEMENT_SCALE_FACTOR", 1.2))
	YOLO_VERBOSE = bool(getattr(C, "YOLO_VERBOSE", False))
except Exception:
	# Fallback defaults if shared config cannot be imported
	CAM_INDEX = 4
	YOLO_MODEL_PATH = "software/vizzy/laptop/yolo11m-seg.engine"
	PIXEL_TO_MM = 1.0 / 2.90
	WORKING_DISTANCE_MM = 600.0
	MOVEMENT_SCALE_FACTOR = 1.2
	YOLO_VERBOSE = False


# ------------------------------- helpers ------------------------------------ #
def _contour_center(mask_u8: np.ndarray) -> Optional[Tuple[int, int]]:
	"""Return (cx, cy) using contour moments of the largest blob in a binary mask."""
	contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return None
	largest = max(contours, key=cv2.contourArea)
	M = cv2.moments(largest)
	if M["m00"] <= 0:
		return None
	cx = int(M["m10"] / M["m00"])
	cy = int(M["m01"] / M["m00"])
	return cx, cy


def instance_center(box_xyxy, mask_tensor, frame_w: int, frame_h: int) -> Tuple[int, int]:
	"""
	Return (cx, cy) for a single detection:
	  - If a mask is provided, compute segmentation center (contour moments).
	  - Else, return box center.
	"""
	if mask_tensor is not None:
		mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
		mask_np = cv2.resize(mask_np, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
		mask_u8 = (mask_np * 255).astype(np.uint8)
		c = _contour_center(mask_u8)
		if c is not None:
			return c

	x1, y1, x2, y2 = map(int, box_xyxy)
	return int((x1 + x2) / 2), int((y1 + y2) / 2)


def calculate_movement_needed(
	obj_offset_x_camera_mm: float,
	obj_offset_y_camera_mm: float,
	working_distance_mm: float,
	scale_factor: float = 1.0,
) -> Tuple[float, float]:
	"""
	Laptop mapping:
	  movement_x_mm = obj_offset_y_camera_mm
	  movement_y_mm = -obj_offset_x_camera_mm
	  then scale both by scale_factor
	"""
	movement_x_mm = obj_offset_y_camera_mm
	movement_y_mm = -obj_offset_x_camera_mm
	movement_x_mm *= scale_factor
	movement_y_mm *= scale_factor
	return movement_x_mm, movement_y_mm


# ------------------------------- main logic --------------------------------- #
def run(model_path: str, cam_index: int, class_id: Optional[int], class_name: Optional[str]) -> None:
	print("=" * 80)
	print("Object Centering - Live Feed (Laptop-style, standalone)")
	print("=" * 80)
	print(f"Model: {model_path}")
	print(f"Camera index: {cam_index}")
	print(f"PIXEL_TO_MM: {PIXEL_TO_MM:.4f}  WORK_DIST_MM: {WORKING_DISTANCE_MM:.1f}  SCALE: {MOVEMENT_SCALE_FACTOR:.2f}")
	if class_id is not None:
		print(f"Class filter (id): {class_id}")
	if class_name is not None:
		print(f"Class filter (name): {class_name}")
	print("Press 'q' to quit")
	print("=" * 80)

	try:
		model = YOLO(model_path)
	except Exception as e:
		print(f"ERROR: Failed to load YOLO model: {e}")
		return

	names = getattr(model, "names", {})
	if isinstance(names, (list, tuple)):
		id_for_name = {str(v).lower(): i for i, v in enumerate(names)}
	elif isinstance(names, dict):
		id_for_name = {str(v).lower(): int(k) for k, v in names.items()}
	else:
		id_for_name = {}

	if class_id is None and class_name:
		class_id = id_for_name.get(class_name.lower(), None)
		if class_id is None:
			print(f"Warning: class name '{class_name}' not found in model names; no class filter will be applied.")

	cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
	if not cap.isOpened():
		print(f"ERROR: Could not open camera index {cam_index}")
		return

	# Configure camera
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(800))
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(600))
	cap.set(cv2.CAP_PROP_FPS, float(30.0))

	time.sleep(1.5)
	for _ in range(5):
		cap.grab()

	ok, test = cap.read()
	if not ok or test is None:
		print("ERROR: Failed to read initial frame from camera")
		cap.release()
		return

	h0, w0 = test.shape[:2]
	center_x = w0 // 2  # image x (columns)
	center_y = h0 // 2  # image y (rows)
	print(f"Frame size: {w0}x{h0} px  center=({center_x},{center_y})")

	frame_count = 0
	fps_t0 = time.time()
	fps_val = 0.0

	while True:
		ok, frame = cap.read()
		if not ok or frame is None:
			print("WARNING: Failed to read frame")
			break

		frame_count += 1
		h, w = frame.shape[:2]
		center_x = w // 2
		center_y = h // 2

		# Run YOLO (optionally with class filter)
		classes_arg = [int(class_id)] if class_id is not None else None
		results = model(frame, classes=classes_arg, verbose=YOLO_VERBOSE)

		# Select best detection (highest conf, tie-break by bbox area)
		best_conf = 0.0
		best_area = 0
		best = None
		for r in results:
			try:
				boxes = r.boxes
				masks = getattr(r, "masks", None)
				mask_list = list(masks.data) if (masks is not None and getattr(masks, "data", None) is not None) else None
				n = 0 if boxes is None else len(boxes)
				for i in range(n):
					xyxy = boxes.xyxy[i].detach().cpu().numpy()
					conf = float(boxes.conf[i].item())
					cls_id_det = int(boxes.cls[i].item())
					x1, y1, x2, y2 = map(int, xyxy)
					area = max(0, x2 - x1) * max(0, y2 - y1)
					mask_tensor = mask_list[i] if mask_list is not None and i < len(mask_list) else None
					cx, cy = instance_center(xyxy, mask_tensor, w, h)
					if (conf > best_conf) or (conf == best_conf and area > best_area):
						best_conf = conf
						best_area = area
						best = {
							"cls_id": cls_id_det,
							"cls_name": str(names[cls_id_det]) if isinstance(names, (list, tuple)) else (str(names.get(cls_id_det, cls_id_det)) if isinstance(names, dict) else str(cls_id_det)),
							"conf": conf,
							"cx": int(cx),
							"cy": int(cy),
						}
			except Exception:
				pass

		# Use YOLO's annotated frame if available
		annotated = frame.copy()
		for r in results:
			try:
				annotated = r.plot()
			except Exception:
				pass

		# Frame center (blue)
		cv2.circle(annotated, (center_x, center_y), 6, (255, 0, 0), -1)
		cv2.circle(annotated, (center_x, center_y), 10, (255, 0, 0), 2)

		# Movement calculation
		movement_x_mm = 0.0
		movement_y_mm = 0.0
		if best is not None:
			bx, by = best["cx"], best["cy"]
			# Object center (red) and line to frame center (green)
			cv2.circle(annotated, (bx, by), 6, (0, 0, 255), -1)
			cv2.circle(annotated, (bx, by), 10, (0, 0, 255), 2)
			cv2.line(annotated, (bx, by), (center_x, center_y), (0, 255, 0), 2)

			# Camera-coordinate offsets (match laptop/object_centering mapping)
			cam_to_obj_x_camera = bx - center_x  # image x (columns)
			cam_to_obj_y_camera = by - center_y  # image y (rows)

			obj_offset_x_camera_mm = cam_to_obj_x_camera * PIXEL_TO_MM
			obj_offset_y_camera_mm = cam_to_obj_y_camera * PIXEL_TO_MM

			movement_x_mm, movement_y_mm = calculate_movement_needed(
				obj_offset_x_camera_mm=obj_offset_x_camera_mm,
				obj_offset_y_camera_mm=obj_offset_y_camera_mm,
				working_distance_mm=WORKING_DISTANCE_MM,
				scale_factor=MOVEMENT_SCALE_FACTOR,
			)

		# FPS update
		if frame_count % 30 == 0:
			now = time.time()
			elapsed = now - fps_t0
			if elapsed > 0:
				fps_val = 30.0 / elapsed
			fps_t0 = now

		# HUD text
		lines = []
		if best is None:
			lines = [
				"No detections",
				f"Model: {model_path}",
			]
		else:
			name_str = best.get("cls_name", str(best.get("cls_id", "?")))
			lines = [
				f"Object: {name_str}  conf={best_conf:.2f}",
				f"move_x {movement_x_mm:+.2f} mm  move_y {movement_y_mm:+.2f} mm",
				f"scale={MOVEMENT_SCALE_FACTOR:.2f}  px2mm={PIXEL_TO_MM:.4f}",
			]

		y0 = 28
		for i, text in enumerate(lines):
			y = y0 + i * 26
			(tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
			cv2.rectangle(annotated, (10, y - th - 6), (10 + tw + 12, y + 6), (0, 0, 0), -1)
			cv2.putText(annotated, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		fps_text = f"FPS: {fps_val:.1f}"
		(fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		cv2.rectangle(annotated, (w - fw - 20, 10), (w - 10, 10 + fh + 10), (0, 0, 0), -1)
		cv2.putText(annotated, fps_text, (w - fw - 15, 10 + fh + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

		cv2.imshow("Object Centering - Live Feed (Press 'q' to quit)", annotated)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	cap.release()
	cv2.destroyAllWindows()
	print("Cleanup complete")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Standalone laptop-style centering viewer")
	p.add_argument("--model", type=str, default=YOLO_MODEL_PATH, help="YOLO model path")
	p.add_argument("--cam-index", type=int, default=CAM_INDEX, help="Camera index (V4L2)")
	p.add_argument("--class-id", type=int, default=None, help="Optional class id filter")
	p.add_argument("--class-name", type=str, default=None, help="Optional class name filter")
	return p.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run(
		model_path=args.model,
		cam_index=args.cam_index,
		class_id=args.class_id,
		class_name=args.class_name,
	)
