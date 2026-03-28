#!/usr/bin/env python3
"""
YOLOv8 Headless Detection Script for Raspberry Pi 3 B
- Runs without display (no cv2.imshow)
- Logs FPS, detection count, inference time
- Uses custom weights from ./model folder
"""

import cv2
import time
import sys
from pathlib import Path
from collections import deque

# Suppress ultralytics logging to keep output clean
import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


class HeadlessDetector:
    def __init__(self, model_path="model/best.pt", source=0, conf=0.25):
        self.model_path = Path(model_path)
        self.source = source
        self.conf = conf

        # Stats
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0
        self.total_detections = 0
        self.start_time = time.time()

        self._validate_setup()
        self._load_model()
        self._init_capture()

    def _validate_setup(self):
        """Check if model exists"""
        if not self.model_path.exists():
            # Try to find any .pt file in model folder
            model_dir = Path("model")
            if model_dir.exists():
                pt_files = list(model_dir.glob("*.pt"))
                if pt_files:
                    self.model_path = pt_files[0]
                    print(f"[INFO] Found model: {self.model_path}")
                else:
                    print(f"[ERROR] No .pt files found in ./model/")
                    sys.exit(1)
            else:
                print(f"[ERROR] Model folder not found: ./model/")
                sys.exit(1)

    def _load_model(self):
        """Load YOLO model"""
        print(f"[INFO] Loading model: {self.model_path}")
        t0 = time.time()

        # Use NCNN for RPi (much faster) or standard YOLO
        self.model = YOLO(str(self.model_path))

        # Warmup
        dummy = cv2.imread(str(self.model_path)) if self.model_path.suffix in ['.jpg', '.png'] else None
        if dummy is None:
            dummy = cv2.imread("/dev/null") if Path("/dev/null").exists() else None

        load_time = time.time() - t0
        print(f"[INFO] Model loaded in {load_time:.2f}s")
        print(f"[INFO] Classes: {self.model.names}")

    def _init_capture(self):
        """Initialize video capture"""
        print(f"[INFO] Opening video source: {self.source}")

        if isinstance(self.source, str) and Path(self.source).exists():
            # Video file
            self.cap = cv2.VideoCapture(self.source)
        else:
            # Camera (V4L2 for RPi better performance)
            self.cap = cv2.VideoCapture(int(self.source))
            # RPi optimizations
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open video source: {self.source}")
            sys.exit(1)

        # Get actual properties
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_target = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Resolution: {self.w}x{self.h}, Target FPS: {self.fps_target:.1f}")

    def _format_detection(self, box, name, conf):
        """Format single detection for logging"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return f"{name}@{conf:.2f} [{cx},{cy}]"

    def run(self):
        """Main detection loop"""
        print("\n" + "=" * 50)
        print("HEADLESS DETECTION STARTED")
        print("Press Ctrl+C to stop")
        print("=" * 50 + "\n")

        # Print CSV header style for easy parsing
        print("timestamp,frame,fps,inf_ms,detections,objects")

        try:
            while True:
                loop_start = time.time()

                # Capture
                ret, frame = self.cap.read()
                if not ret:
                    print("[WARN] Frame capture failed")
                    time.sleep(0.1)
                    continue

                # Inference
                t0 = time.time()
                results = self.model(frame, verbose=False, conf=self.conf)
                inf_time = (time.time() - t0) * 1000  # ms

                # Parse detections
                detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = self.model.names[cls]
                        detections.append(self._format_detection(box, name, conf))

                # Stats
                self.frame_count += 1
                self.total_detections += len(detections)

                # FPS calculation
                loop_time = time.time() - loop_start
                current_fps = 1.0 / loop_time if loop_time > 0 else 0
                self.fps_history.append(current_fps)
                avg_fps = sum(self.fps_history) / len(self.fps_history)

                # Output formats
                timestamp = time.strftime("%H:%M:%S")
                det_count = len(detections)
                det_str = ";".join(detections) if detections else "none"

                # Log line (compact)
                log_line = (
                    f"[{timestamp}] "
                    f"Frame {self.frame_count:4d} | "
                    f"FPS: {avg_fps:5.1f} | "
                    f"Inf: {inf_time:5.1f}ms | "
                    f"Dets: {det_count:2d} | "
                    f"{det_str[:60]}{'...' if len(det_str) > 60 else ''}"
                )
                print(log_line)

                # Optional: CSV format to stderr for piping
                # import sys
                # print(f"{timestamp},{self.frame_count},{avg_fps:.1f},{inf_time:.1f},{det_count},{det_str}", file=sys.stderr)

        except KeyboardInterrupt:
            self._print_summary()

    def _print_summary(self):
        """Print final statistics"""
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 50)
        print("DETECTION SUMMARY")
        print("=" * 50)
        print(f"Total frames:    {self.frame_count}")
        print(f"Total detections: {self.total_detections}")
        print(f"Runtime:         {elapsed:.1f}s")
        print(f"Average FPS:     {avg_fps:.1f}")
        print(f"Frames dropped:  ~{max(0, int(elapsed * self.fps_target) - self.frame_count)}")
        print("=" * 50)

        self.cap.release()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Headless YOLOv8 Detection for RPi")
    parser.add_argument("--model", default="model/best.pt", help="Path to model weights")
    parser.add_argument("--source", default="0", help="Video source (0=camera, or file path)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--ncnn", action="store_true", help="Use NCNN format for RPi (faster)")
    args = parser.parse_args()

    # Convert source to int if it's a number
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    detector = HeadlessDetector(
        model_path=args.model,
        source=source,
        conf=args.conf
    )
    detector.run()


if __name__ == "__main__":
    main()