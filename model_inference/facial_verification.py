#!/usr/bin/env python3
"""
EdgeFace-based facial verification and recognition pipeline.

This module combines YOLO person detections with the EdgeFace-Base backbone
to extract face embeddings on edge devices (Jetson). Detected faces can be
matched against a gallery of known embeddings and the results can be merged
into existing JSON payloads used by the broader CV pipeline.
"""
import json
import os
import time
from collections import deque
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import supervision as sv
from object_bounder_simple import ObjectBounder

try:
    from backbones import get_model
except ImportError:  # pragma: no cover - handled at runtime
    get_model = None

try:
    from face_alignment import align as face_align_module  # type: ignore

    FACE_ALIGNMENT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    face_align_module = None
    FACE_ALIGNMENT_AVAILABLE = False


class FacialVerification:
    """
    Facial verification using YOLO person boxes + EdgeFace embeddings.

    The class mirrors the structure of `DepthEstimator` and `EdgeTAMTracker` so it
    can slot into the existing Jetson pipeline with minimal glue code.
    """

    def __init__(
        self,
        model_name: str = "edgeface_base",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        detector_model_size: str = "n",
        detector_threshold: float = 0.25,
        match_threshold: float = 0.45,
        face_margin: float = 0.25,
        input_size: int = 112,
        include_embeddings: bool = True,
        embedding_precision: int = 5,
    ) -> None:
        """
        Args:
            model_name: EdgeFace backbone identifier (default: edgeface_base).
            checkpoint_path: Path to the model checkpoint (.pt). If None the loader
                             will try $EDGEFACE_CHECKPOINT or ./checkpoints/{model_name}.pt.
            device: Torch device string (e.g. 'cuda', 'cuda:0', 'cpu'). Auto-detected if None.
            detector_model_size: YOLOv8 model size used for person detection.
            detector_threshold: Confidence threshold for YOLO detections.
            match_threshold: Cosine similarity threshold for accepting gallery matches.
            face_margin: Fractional padding added around YOLO boxes before cropping faces.
            input_size: EdgeFace input resolution (default 112 x 112).
            include_embeddings: Whether to attach raw embeddings in JSON payloads.
            embedding_precision: Decimal precision for embedding values when serialized.
        """
        if get_model is None:
            raise ImportError(
                "Could not import EdgeFace backbones. "
                "Ensure the EdgeFace repository (providing `backbones.get_model`) is on PYTHONPATH."
            )

        self.model_name = model_name
        self.face_margin = face_margin
        self.input_size = input_size
        self.include_embeddings = include_embeddings
        self.embedding_precision = embedding_precision
        self.match_threshold = match_threshold
        self.person_class_id = 0  # COCO class id for "person"
        self.embedding_dim: Optional[int] = None

        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if isinstance(device, int):
                self.device = f"cuda:{device}"
            else:
                self.device = device

        print("Loading EdgeFace backbone...")
        self.model = self._load_edgeface_model(model_name, checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"EdgeFace ready on {self.device.upper()}")

        # Preprocessing transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Optional face alignment helper
        self.aligner = face_align_module if FACE_ALIGNMENT_AVAILABLE else None
        if self.aligner:
            print("Face alignment module detected; attempting to align crops.")

        # YOLO detector for person boxes
        self.detector = ObjectBounder(model_size=detector_model_size, threshold=detector_threshold)

        # Supervision annotators for visualization
        self.box_annotator = sv.BoxAnnotator(thickness=2, text_scale=0.5, text_thickness=1)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

        # Gallery of known embeddings {name: np.ndarray}
        self.gallery: Dict[str, np.ndarray] = {}

        # Performance tracking
        self.frame_idx = 0
        self.latency_window = deque(maxlen=60)

    # ---------------------------------------------------------------------
    # Model + gallery helpers
    # ---------------------------------------------------------------------
    def _load_edgeface_model(self, model_name: str, checkpoint_path: Optional[str]):
        checkpoint_path = checkpoint_path or os.environ.get(
            "EDGEFACE_CHECKPOINT",
            os.path.join("checkpoints", f"{model_name}.pt"),
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"EdgeFace checkpoint not found: {checkpoint_path}. "
                "Download the .pt file from https://huggingface.co/Idiap/EdgeFace-Base "
                "or update EDGEFACE_CHECKPOINT."
            )

        model = get_model(model_name)
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing keys while loading EdgeFace checkpoint: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys while loading EdgeFace checkpoint: {unexpected}")
        return model

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def register_identity(
        self,
        name: str,
        image_path: Optional[str] = None,
        image_bgr: Optional[np.ndarray] = None,
        embedding: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        """
        Register a known identity for gallery matching.

        Args:
            name: Identity label.
            image_path: Path to an image containing the face.
            image_bgr: Raw BGR image (OpenCV). Used if image_path is None.
            embedding: Optional precomputed embedding iterable.

        Returns:
            The normalized embedding stored in the gallery.
        """
        if embedding is not None:
            embedding_np = np.asarray(list(embedding), dtype=np.float32)
        else:
            if image_bgr is None and image_path:
                image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError("Either image_path, image_bgr, or embedding must be provided.")
            tensor = self._transform_face_image(image_bgr)
            if tensor is None:
                raise ValueError("Failed to preprocess face image for gallery registration.")
            embedding_np = self._compute_embedding(tensor)

        embedding_np = self._normalize_embedding(embedding_np)
        self.gallery[name] = embedding_np.astype(np.float32)
        print(f"Registered identity '{name}' (embedding dim={embedding_np.size})")
        return embedding_np

    def load_gallery_from_json(self, path: str) -> None:
        """
        Load gallery entries from JSON.

        Expected formats:
            {"alice": "path/to/image.jpg", "bob": "path/to/another.jpg"}
        or:
            [{"name": "alice", "image": "path.jpg"},
             {"name": "bob", "embedding": [...]}]
        """
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, dict):
            iterable = [{"name": name, "image": image_path} for name, image_path in data.items()]
        else:
            iterable = data

        for entry in iterable:
            name = entry.get("name")
            if not name:
                continue
            embedding = entry.get("embedding")
            image_path = entry.get("image") or entry.get("path")
            self.register_identity(name, image_path=image_path, embedding=embedding)

    # ---------------------------------------------------------------------
    # Face preprocessing & embedding
    # ---------------------------------------------------------------------
    def _prepare_face_crop(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        if frame is None or bbox is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(np.float32)
        width = x2 - x1
        height = y2 - y1
        if width <= 1 or height <= 1:
            return None

        pad_x = width * self.face_margin
        pad_y = height * self.face_margin
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))

        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _transform_face_image(self, face_bgr: np.ndarray) -> Optional[torch.Tensor]:
        if face_bgr is None or face_bgr.size == 0:
            return None
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        if self.aligner and hasattr(self.aligner, "get_aligned_face"):
            try:
                aligned = self.aligner.get_aligned_face(face_rgb)  # type: ignore
                if isinstance(aligned, np.ndarray) and aligned.size > 0:
                    face_rgb = aligned
            except Exception:
                # Alignment is best-effort; fall back to raw crop on failure.
                pass

        pil_image = Image.fromarray(face_rgb)
        return self.transform(pil_image)

    def _compute_embedding(self, face_tensor: torch.Tensor) -> np.ndarray:
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(face_tensor)

        if isinstance(embedding, dict):
            embedding = next(iter(embedding.values()))
        if isinstance(embedding, (list, tuple)):
            embedding = embedding[0]

        embedding = embedding.squeeze(0)
        embedding = F.normalize(embedding, p=2, dim=0)
        embedding_np = embedding.cpu().numpy().astype(np.float32)
        self.embedding_dim = embedding_np.size
        return embedding_np

    def _match_embedding(self, embedding: np.ndarray) -> Dict[str, Optional[float]]:
        if not self.gallery:
            return {"name": None, "similarity": None, "is_match": False}

        best_name = None
        best_score = -1.0
        for name, ref_embedding in self.gallery.items():
            score = float(np.dot(embedding, ref_embedding))
            if score > best_score:
                best_score = score
                best_name = name

        is_match = best_score >= self.match_threshold
        return {
            "name": best_name,
            "similarity": round(best_score, 4) if best_score >= 0 else None,
            "is_match": bool(is_match),
            "threshold": self.match_threshold,
        }

    # Main processing
    # ---------------------------------------------------------------------
    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: Optional[int] = None,
        timestamp: Optional[float] = None,
        detections: Optional[sv.Detections] = None,
        threshold: Optional[float] = None,
        show_labels: bool = True,
        include_embeddings: Optional[bool] = None,
    ) -> Dict[str, Optional[Dict]]:
        """
        Process a single frame, returning detections, embeddings, and optional payload.
        """
        if threshold is None:
            threshold = self.detector.threshold

        include_embeddings = self.include_embeddings if include_embeddings is None else include_embeddings

        start_time = time.time()

        if detections is None:
            detections = self.detector.predict(frame, threshold=threshold)

        face_entries: List[Dict] = []
        face_boxes = []
        face_confidences = []

        if detections is not None and len(detections) > 0:
            for det_idx, (bbox, class_id, confidence) in enumerate(
                zip(detections.xyxy, detections.class_id, detections.confidence)
            ):
                if int(class_id) != self.person_class_id:
                    continue

                crop = self._prepare_face_crop(frame, bbox)
                if crop is None:
                    continue

                tensor = self._transform_face_image(crop)
                if tensor is None:
                    continue

                embedding = self._compute_embedding(tensor)
                embedding = self._normalize_embedding(embedding)
                match = self._match_embedding(embedding)

                serialized_embedding = None
                if include_embeddings:
                    serialized_embedding = [
                        round(float(val), self.embedding_precision) for val in embedding.tolist()
                    ]

                entry = {
                    "id": f"face_{len(face_entries)}",
                    "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "confidence": float(confidence),
                    "embedding_dim": int(self.embedding_dim or len(embedding)),
                    "match": match,
                }
                if serialized_embedding is not None:
                    entry["embedding"] = serialized_embedding

                face_entries.append(entry)
                face_boxes.append(bbox)
                face_confidences.append(confidence)

        latency = time.time() - start_time
        self.latency_window.append(latency)
        self.frame_idx += 1

        face_section = {
            "model": self.model_name,
            "embedding_dim": self.embedding_dim,
            "gallery_size": len(self.gallery),
            "detections": face_entries,
            "latency_ms": round(latency * 1000, 2),
            "avg_latency_ms": round(float(np.mean(self.latency_window) * 1000), 2)
            if self.latency_window
            else None,
        }

        payload = None
        if frame_idx is not None or timestamp is not None:
            payload = {
                "frame": int(frame_idx) if frame_idx is not None else None,
                "timestamp": round(float(timestamp if timestamp is not None else time.time()), 3),
                "face_recognition": face_section,
            }

        annotated_frame = frame
        if face_boxes:
            detections_for_draw = sv.Detections(
                xyxy=np.asarray(face_boxes),
                confidence=np.asarray(face_confidences),
                class_id=np.zeros(len(face_boxes), dtype=int),
            )
            labels = None
            if show_labels:
                labels = []
                for entry in face_entries:
                    match = entry["match"]
                    if match["name"] and match.get("is_match"):
                        label = f"{match['name']} {match['similarity']:.2f}"
                    elif match["name"]:
                        label = f"{match['name']}? {match['similarity'] or 0:.2f}"
                    else:
                        label = f"person {entry['confidence']:.2f}"
                    labels.append(label)
            annotated_frame = self.box_annotator.annotate(frame.copy(), detections_for_draw)
            if labels:
                annotated_frame = self.label_annotator.annotate(annotated_frame, detections_for_draw, labels)

        return {
            "annotated_frame": annotated_frame,
            "face_section": face_section,
            "payload": payload,
        }

    def update_json_entry(self, entry: Dict, face_section: Dict) -> Dict:
        """
        Merge face recognition data into an existing JSON payload.
        """
        if face_section and face_section.get("detections"):
            entry["face_recognition"] = face_section
        return entry

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        skip_frames: int = 1,
        show_labels: bool = True,
        json_output_path: Optional[str] = None,
    ) -> None:
        """
        Run end-to-end facial recognition on a video file.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps is None or fps <= 0 or np.isnan(fps):
            print("Source FPS invalid, defaulting to 30.0")
            fps = 30.0

        out = None
        if output_path:
            fourcc_options = [
                ("avc1", "H.264 (hardware)"),
                ("h264", "H.264 (software)"),
                ("mp4v", "MPEG-4 (fallback)"),
            ]
            for fourcc_str, codec_name in fourcc_options:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    print(f"Using codec: {codec_name}")
                    break
                out.release()
                out = None
            if out is None:
                raise IOError("Failed to create output video with any codec")

        json_file = None
        json_entries = 0
        if json_output_path:
            json_dir = os.path.dirname(json_output_path)
            if json_dir:
                os.makedirs(json_dir, exist_ok=True)
            json_file = open(json_output_path, "w", encoding="utf-8")
            json_file.write("[\n")

        print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total} frames")
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
                    result = self.process_frame(
                        frame,
                        frame_idx=frame_idx,
                        timestamp=time.time(),
                        show_labels=show_labels,
                    )
                    annotated = result["annotated_frame"]
                    face_section = result["face_section"]
                    payload = result["payload"]
                    last_annotated = annotated

                    if json_file and payload:
                        if json_entries > 0:
                            json_file.write(",\n")
                        json.dump(payload, json_file, indent=2)
                        json_file.flush()
                        json_entries += 1
                else:
                    annotated = last_annotated if last_annotated is not None else frame

                if out is not None:
                    out.write(annotated)

                frame_idx += 1
                if frame_idx % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_idx / elapsed
                    percent = (frame_idx / total) * 100 if total > 0 else 0
                    eta = (total - frame_idx) / fps_actual if fps_actual > 0 and total > 0 else 0
                    print(
                        f"Progress: {frame_idx}/{total} ({percent:.1f}%) | "
                        f"{fps_actual:.1f} FPS | ETA: {eta:.0f}s"
                    )
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            cap.release()
            if out is not None:
                out.release()
            if json_file:
                json_file.write("\n]\n")
                json_file.close()
                print(f"Wrote {json_entries} JSON entries to {json_output_path}")

            elapsed = time.time() - start_time
            if frame_idx > 0 and elapsed > 0:
                print(
                    f"\nDone! Processed {frame_idx} frames in {elapsed:.1f}s "
                    f"({frame_idx / elapsed:.1f} FPS)"
                )
            if out is not None and output_path:
                print(f"Output video: {output_path}")

    def close(self) -> None:
        """Placeholder for interface parity."""
        return


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="EdgeFace Facial Verification")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", help="Annotated output video path")
    parser.add_argument("--checkpoint", help="EdgeFace checkpoint path (.pt)")
    parser.add_argument("--detector-model", default="n", choices=["n", "s", "m", "l", "x"], help="YOLOv8 model size")
    parser.add_argument("--detector-threshold", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--match-threshold", type=float, default=0.45, help="Cosine similarity threshold for matches")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--gallery", help="JSON file with gallery definitions")
    parser.add_argument("--json-output", help="Write per-frame JSON payloads to file")
    parser.add_argument("--no-labels", action="store_true", help="Hide labels/IDs in video output")
    parser.add_argument("--cpu", action="store_true", help="Force inference on CPU")
    args = parser.parse_args()

    verifier = FacialVerification(
        model_name="edgeface_base",
        checkpoint_path=args.checkpoint,
        device="cpu" if args.cpu else None,
        detector_model_size=args.detector_model,
        detector_threshold=args.detector_threshold,
        match_threshold=args.match_threshold,
    )

    if args.gallery:
        verifier.load_gallery_from_json(args.gallery)

    verifier.process_video(
        input_path=args.input,
        output_path=args.output,
        skip_frames=args.skip_frames,
        show_labels=not args.no_labels,
        json_output_path=args.json_output,
    )


if __name__ == "__main__":
    main()
