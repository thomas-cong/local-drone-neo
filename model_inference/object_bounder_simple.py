#!/usr/bin/env python3
"""
Simple Object Detection using YOLOv8
Lightweight and reliable object detection
"""
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


class ObjectBounder:
    """Simple object detection using YOLOv8"""

    def __init__(self, model_size='n', threshold=0.5):
        """
        Initialize YOLOv8 object detection

        Args:
            model_size: Model size ('n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=xlarge)
            threshold: Confidence threshold (default: 0.5)
        """
        print(f"Loading YOLOv8{model_size} model...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.threshold = threshold
        print("Model ready")

    def predict(self, image, threshold=None):
        """
        Detect objects in image

        Args:
            image: numpy array (BGR) or PIL Image
            threshold: Override default threshold

        Returns:
            supervision.Detections object
        """
        if threshold is None:
            threshold = self.threshold

        # Run inference
        results = self.model(image, conf=threshold, verbose=False)[0]

        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(results)
        return detections

    def annotate(self, image, detections, show_labels=True):
        """
        Annotate image with boxes and labels

        Args:
            image: numpy array (BGR)
            detections: supervision.Detections
            show_labels: Show class labels

        Returns:
            Annotated image (BGR numpy array)
        """
        annotated = image.copy()

        # Add boxes
        box_annotator = sv.BoxAnnotator()
        annotated = box_annotator.annotate(annotated, detections)

        # Add labels
        if show_labels:
            labels = [
                f"{self.model.names[class_id]} {confidence:.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]
            label_annotator = sv.LabelAnnotator()
            annotated = label_annotator.annotate(annotated, detections, labels)

        return annotated

    def detect_and_annotate(self, image, threshold=None, show_labels=True):
        """Detect and annotate in one call"""
        detections = self.predict(image, threshold)
        annotated = self.annotate(image, detections, show_labels)
        return annotated, detections

    def process_video(self, input_path, output_path,
                      threshold=None,
                      show_labels=True,
                      skip_frames=1):
        """
        Process video file

        Args:
            input_path: Input video path
            output_path: Output video path
            threshold: Override threshold
            show_labels: Show labels
            skip_frames: Process every Nth frame
        """
        if threshold is None:
            threshold = self.threshold

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # depth estimator style: fix bad fps
        if fps is None or fps <= 0 or np.isnan(fps):
            print("Source FPS invalid, defaulting to 30.0")
            fps = 30.0

        print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total} frames")

        # Use the same codec strategy as depth estimator
        fourcc_options = [
            ('avc1', 'H.264 (hardware)'),
            ('h264', 'H.264 (software)'),
            ('mp4v', 'MPEG-4 (fallback)'),
        ]

        out = None
        for fourcc_str, codec_name in fourcc_options:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"Using codec: {codec_name}")
                break
            out.release()
            out = None

        if out is None:
            raise IOError("Failed to create output with any codec")

        print(f"Processing video (every {skip_frames} frame(s))...")

        frame_idx = 0
        start_time = time.time()
        last_annotated = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % skip_frames == 0:
                    annotated, detections = self.detect_and_annotate(
                        frame,
                        threshold=threshold,
                        show_labels=show_labels
                    )
                    last_annotated = annotated
                else:
                    annotated = last_annotated if last_annotated is not None else frame

                out.write(annotated)
                frame_idx += 1

                if frame_idx % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_idx / elapsed
                    percent = (frame_idx / total) * 100 if total > 0 else 0
                    eta = (total - frame_idx) / fps_actual if fps_actual > 0 and total > 0 else 0
                    print(f"Progress: {frame_idx}/{total} ({percent:.1f}%) | "
                          f"{fps_actual:.1f} FPS | ETA: {eta:.0f}s")

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            cap.release()
            out.release()
            elapsed = time.time() - start_time
            print(f"\nDone! Processed {frame_idx} frames in {elapsed:.1f}s "
                  f"({frame_idx / elapsed:.1f} FPS)")
            print(f"Output: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple Object Detection (YOLOv8)")
    parser.add_argument("--input", required=True, help="Input video/image path")
    parser.add_argument("--output", help="Output path (for video)")
    parser.add_argument("--model", default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help="Model size: n(ano), s(mall), m(edium), l(arge), x(large)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--no-labels", action="store_true", help="Hide labels")
    args = parser.parse_args()

    detector = ObjectBounder(model_size=args.model, threshold=args.threshold)

    ext = os.path.splitext(args.input)[1].lower()

    # Image path
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        image = cv2.imread(args.input)
        annotated, detections = detector.detect_and_annotate(
            image, show_labels=not args.no_labels
        )

        print(f"\nDetected {len(detections)} objects:")
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            print(f"  - {detector.model.names[class_id]}: {confidence:.2f}")

        if args.output:
            cv2.imwrite(args.output, annotated)
            print(f"Saved to: {args.output}")
        else:
            cv2.imshow('Detection', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Video path
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        if not args.output:
            print("Error: --output required for video")
            return

        detector.process_video(
            args.input,
            args.output,
            threshold=args.threshold,
            show_labels=not args.no_labels,
            skip_frames=args.skip_frames
        )
    else:
        print(f"Unsupported file: {ext}")


if __name__ == "__main__":
    main()
