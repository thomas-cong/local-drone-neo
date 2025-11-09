#!/usr/bin/env python3
"""
Quick test script to verify similarity between two headshot images.

This script uses `FacialVerification` (EdgeFace embeddings + YOLO person crops)
to register `anush.png` as the gallery identity and then runs inference on
`anush1.jpg`. The resulting annotated frame—with bounding box and similarity
score—is saved as `anush1_with_match.png` in the same directory.
"""

import argparse
import os
import sys
from typing import Optional

import cv2

# Ensure we can import from the project root when executed from elsewhere.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from facial_verification import FacialVerification  # noqa: E402


def build_verifier(checkpoint_path: Optional[str]) -> FacialVerification:
    """
    Construct the FacialVerification helper with lightweight defaults.
    """
    return FacialVerification(
        model_name="edgeface_base",
        checkpoint_path=checkpoint_path,
        detector_model_size="n",
        detector_threshold=0.25,
        match_threshold=0.45,
        include_embeddings=False,
    )


def run_test(
    gallery_path: str,
    probe_path: str,
    output_path: str,
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Register the gallery image and evaluate similarity on the probe image.
    """
    verifier = build_verifier(checkpoint_path)
    verifier.register_identity("anush", image_path=gallery_path)

    frame = cv2.imread(probe_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load probe image: {probe_path}")

    result = verifier.process_frame(frame, show_labels=True)
    detections = result["face_section"]["detections"]
    if not detections:
        raise RuntimeError("No faces detected in probe image; cannot produce annotation.")

    annotated = result["annotated_frame"]
    cv2.imwrite(output_path, annotated)

    match_info = detections[0]["match"]
    similarity = match_info.get("similarity")
    print(f"Saved annotated probe image with match overlay to: {output_path}")
    print(f"Similarity score vs gallery: {similarity}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headshot match smoke test.")
    parser.add_argument(
        "--gallery",
        default=os.path.join(os.path.dirname(__file__), "anush.png"),
        help="Path to the gallery image (default: anush.png in headshots dir)",
    )
    parser.add_argument(
        "--probe",
        default=os.path.join(os.path.dirname(__file__), "anush1.jpg"),
        help="Path to the probe image (default: anush1.jpg in headshots dir)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "anush1_with_match.png"),
        help="Path to save the annotated probe image",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("EDGEFACE_CHECKPOINT"),
        help="Optional EdgeFace checkpoint path; overrides EDGEFACE_CHECKPOINT env var",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_test(
        gallery_path=args.gallery,
        probe_path=args.probe,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()


