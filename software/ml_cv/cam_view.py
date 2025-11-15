#!/usr/bin/env python3
import cv2
import argparse
import sys

def main():
    p = argparse.ArgumentParser(description="Simple USB camera viewer (OpenCV) with scaling and upside-down flip.")
    p.add_argument("--camera_index", "-d", type=int, default=4, help="Camera index (/dev/videoN). Default: 4")
    p.add_argument("--scale", type=float, default=1.0, help="Display scale factor (e.g., 0.5 = half size, 1.5 = 150%)")
    p.add_argument("--flipud", action="store_true", help="Flip the image upside down (180°)")
    args = p.parse_args()

    if args.scale <= 0:
        print("ERROR: --scale must be > 0", file=sys.stderr)
        sys.exit(2)

    # Open the video device (uses V4L2 backend on Linux). Immediately test
    # a frame so we can report failures early.
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    
    # Request MJPG (smaller USB payload than raw YUYV)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(800))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(600))
    cap.set(cv2.CAP_PROP_FPS,          float(30.0))
    
    # Test a frame to report failures early
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {args.camera_index}.", file=sys.stderr)
        sys.exit(1)
    
    ok, test_frame = cap.read()
    if not ok or test_frame is None:
        print(f"ERROR: Could not read initial frame from camera index {args.camera_index}.", file=sys.stderr)
        cap.release()
        sys.exit(1)

    win = "Camera (press 'q' to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)  # resizable window

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("WARN: Failed to read frame.")
            break

        # Flip upside down if requested (same as rotate 180°)
        if args.flipud:
            # flipCode = -1 flips both horizontally & vertically (180°)
            frame = cv2.flip(frame, -1)

        # Scale the displayed image (capture stays original size)
        if args.scale != 1.0:
            new_w = max(1, int(frame.shape[1] * args.scale))
            new_h = max(1, int(frame.shape[0] * args.scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
