#!/usr/bin/env python3
"""
EdgeTAM Object Tracker - Class-based implementation
Optimized for speed with multi-object tracking
"""
import os
import sys
import time
import tempfile
import cv2
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()


class EdgeTAMTracker:
    """EdgeTAM-based multi-object video tracker"""
    
    def __init__(self, checkpoint=None, config=None, device="cuda"):
        """
        Initialize EdgeTAM tracker
        
        Args:
            checkpoint: Path to EdgeTAM checkpoint
            config: Path to EdgeTAM config
            device: Device to use (cuda/cpu)
        """
        from sam2.build_sam import build_sam2_video_predictor, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mask_alpha = 0.4
        
        # Model paths
        self.checkpoint = checkpoint or os.environ.get(
            "EDGETAM_CHECKPOINT", 
            "./EdgeTAM/checkpoints/edgetam.pt"
        )
        self.config = config or os.environ.get("EDGETAM_CONFIG", "edgetam.yaml")
        
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        
        # Load models
        print("Loading EdgeTAM models...")
        self.video_predictor = build_sam2_video_predictor(self.config, self.checkpoint)
        self.sam2_model = build_sam2(self.config, self.checkpoint).to(self.device)
        self.image_predictor = SAM2ImagePredictor(self.sam2_model)
        
        if self.device == "cuda":
            print("Using GPU with FP16 autocast for detection")
        else:
            print("Using CPU (FP32)")
        
        print("EdgeTAM ready")
        
        # Color palette for visualizations
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 255, 0),  # Spring Green
            (255, 128, 0),  # Orange
        ]
    
    def find_objects(self, frame, max_objects=5, min_coverage=0.1, max_coverage=0.6):
        """
        Find objects in frame using automatic segmentation
        
        Args:
            frame: Input frame (BGR)
            max_objects: Maximum objects to detect
            min_coverage: Minimum mask coverage (fraction of frame)
            max_coverage: Maximum mask coverage (fraction of frame)
            
        Returns:
            List of prompt dicts with 'points' and 'labels'
        """
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image_predictor.set_image(frame_rgb)
        
        # Sample grid to find distinct objects
        grid_size = 4
        step_x = w // (grid_size + 1)
        step_y = h // (grid_size + 1)
        
        found_objects = []
        used_masks = []
        
        autocast_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=autocast_dtype):
            for i in range(1, grid_size + 1):
                for j in range(1, grid_size + 1):
                    x = i * step_x
                    y = j * step_y
                    
                    pts = np.array([[x, y]], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)
                    
                    try:
                        masks, scores, _ = self.image_predictor.predict(
                            point_coords=pts,
                            point_labels=labels,
                            multimask_output=True,
                        )
                    except:
                        continue
                    
                    if len(masks) == 0:
                        continue
                    
                    # Evaluate each mask
                    for mask, score in zip(masks, scores):
                        if isinstance(mask, torch.Tensor):
                            mask = mask.cpu().numpy()
                        
                        mask_bool = mask > 0.5
                        coverage = np.sum(mask_bool) / (h * w)
                        
                        # Check coverage limits
                        if coverage < min_coverage or coverage > max_coverage:
                            continue
                        
                        # Check if this is a new object (not overlapping existing)
                        is_new = True
                        for used_mask in used_masks:
                            overlap = np.sum(mask_bool & used_mask) / np.sum(mask_bool | used_mask)
                            if overlap > 0.5:
                                is_new = False
                                break
                        
                        if is_new:
                            yy, xx = np.where(mask_bool)
                            if len(xx) > 0:
                                cx = int(np.mean(xx))
                                cy = int(np.mean(yy))
                                
                                found_objects.append({
                                    'points': np.array([[cx, cy]], dtype=np.float32),
                                    'labels': np.array([1], dtype=np.int32),
                                    'coverage': coverage,
                                    'score': float(score)
                                })
                                used_masks.append(mask_bool)
                                print(f"Object {len(found_objects)}: coverage={coverage:.2%}, score={score:.2f}")
                                
                                if len(found_objects) >= max_objects:
                                    break
                    
                    if len(found_objects) >= max_objects:
                        break
                
                if len(found_objects) >= max_objects:
                    break
        
        # Fallback to center if nothing found
        if not found_objects:
            print("No objects found, using center point")
            return [{
                'points': np.array([[w//2, h//2]], dtype=np.float32),
                'labels': np.array([1], dtype=np.int32)
            }]
        
        print(f"Found {len(found_objects)} objects to track")
        return found_objects
    
    def _create_sparse_video(self, video_path, skip_frames):
        """
        Create temporary video with every Nth frame
        
        Note: Must write to file because EdgeTAM's init_state() requires a path
        (uses decord internally which doesn't support VideoCapture objects)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use /tmp for faster I/O if available (RAM disk on most systems)
        temp_dir = '/tmp' if os.path.exists('/tmp') else None
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4', dir=temp_dir)
        os.close(temp_fd)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps / skip_frames, (width, height))
        
        frame_idx = 0
        frame_map = []
        
        print(f"Creating sparse video (every {skip_frames} frames)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % skip_frames == 0:
                out.write(frame)
                frame_map.append(frame_idx)
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        print(f"Sparse video: {len(frame_map)} frames (from {frame_idx} original)")
        return temp_path, frame_map
    
    def track(self, video_path, prompts, skip_frames=5):
        """
        Track objects through video
        
        Args:
            video_path: Path to input video
            prompts: List of prompt dicts from find_objects()
            skip_frames: Track every Nth frame
            
        Returns:
            Dict mapping frame indices to mask dicts
        """
        sparse_video, frame_map = self._create_sparse_video(video_path, skip_frames)
        
        try:
            with torch.inference_mode():
                inference_state = self.video_predictor.init_state(sparse_video)
                
                # Add all prompts
                for obj_id, prompt in enumerate(prompts):
                    points = prompt['points']
                    labels = prompt['labels']
                    
                    if points.ndim == 2:
                        points = points.reshape(1, -1, 2)
                    if labels.ndim == 1:
                        labels = labels.reshape(1, -1)
                    
                    self.video_predictor.add_new_points_or_box(
                        inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        points=points,
                        labels=labels,
                    )
                
                # Collect tracking results
                tracking_results = {}
                for sparse_idx, obj_ids, masks in self.video_predictor.propagate_in_video(inference_state):
                    original_idx = frame_map[sparse_idx]
                    masks_dict = {obj_id: mask for obj_id, mask in zip(obj_ids, masks)}
                    tracking_results[original_idx] = masks_dict
                    
                    if sparse_idx % 10 == 0:
                        print(f"  Tracked {sparse_idx}/{len(frame_map)} frames")
        
        finally:
            if os.path.exists(sparse_video):
                os.remove(sparse_video)
        
        return tracking_results
    
    def draw_masks(self, frame, masks_dict):
        """
        Draw tracking masks on frame
        
        Args:
            frame: Input frame (BGR)
            masks_dict: Dict mapping object IDs to masks
            
        Returns:
            Annotated frame
        """
        if not masks_dict:
            return frame
        
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        for obj_id, mask in masks_dict.items():
            if mask is None:
                continue
            
            # Convert mask
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            if mask.ndim == 3:
                mask = mask.squeeze()
            
            # Resize if needed
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), 
                                interpolation=cv2.INTER_NEAREST)
            
            mask_bool = mask > 0.5
            coverage = np.count_nonzero(mask_bool) / (h * w)
            
            # Filter out invalid masks
            if coverage < 0.001 or coverage > 0.8:
                continue
            
            # Get color
            color = self.colors[obj_id % len(self.colors)]
            
            # Blend mask
            overlay[mask_bool] = (
                overlay[mask_bool] * (1 - self.mask_alpha) + 
                np.array(color, dtype=np.float32) * self.mask_alpha
            ).astype(np.uint8)
        return overlay
    
    def process_video(self, input_path, output_path, max_objects=3, skip_frames=5):
        """
        Full pipeline: detect, track, and render
        
        Args:
            input_path: Input video path
            output_path: Output video path
            max_objects: Maximum objects to track
            skip_frames: Track every Nth frame
        """
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total} frames")
        
        # Find objects in first frame
        ret, first_frame = cap.read()
        if not ret:
            raise IOError("Failed to read first frame")
        
        print("Finding objects in first frame...")
        prompts = self.find_objects(first_frame, max_objects=max_objects)
        
        # Setup output
        fourcc_options = [
            ('avc1', 'H.264 (hardware)'),
            ('h264', 'H.264 (software)'),
            ('mp4v', 'MPEG-4 (fallback)')
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
        
        # Track objects
        print(f"Tracking {len(prompts)} objects (every {skip_frames} frames)...")
        tracking_results = self.track(input_path, prompts, skip_frames)
        
        # Apply masks to video
        print("Applying masks to output video...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_idx = 0
        last_masks = {}
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get masks for this frame
                if frame_idx in tracking_results:
                    last_masks = tracking_results[frame_idx]
                
                # Draw and write
                annotated = self.draw_masks(frame, last_masks)
                out.write(annotated)
                
                # Progress
                frame_idx += 1
                if frame_idx % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_idx / elapsed
                    percent = (frame_idx / total) * 100
                    eta = (total - frame_idx) / fps_actual if fps_actual > 0 else 0
                    print(f"Progress: {frame_idx}/{total} ({percent:.1f}%) | "
                          f"{fps_actual:.1f} FPS | ETA: {eta:.0f}s")
        
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            cap.release()
            out.release()
            
            elapsed = time.time() - start_time
            print(f"\nDone! Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)")
            print(f"Output: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="EdgeTAM Multi-Object Tracking")
    parser.add_argument("--input", required=True, help="Input MP4 video")
    parser.add_argument("--output", required=True, help="Output MP4 video")
    parser.add_argument("--skip-frames", type=int, default=5, 
                       help="Track every Nth frame (default=5)")
    parser.add_argument("--max-objects", type=int, default=3,
                       help="Maximum number of objects to track (default=3)")
    parser.add_argument("--checkpoint", type=str, help="Path to EdgeTAM checkpoint")
    parser.add_argument("--config", type=str, help="Path to EdgeTAM config")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}")
        sys.exit(1)
    
    # Create tracker and process
    tracker = EdgeTAMTracker(
        checkpoint=args.checkpoint,
        config=args.config
    )
    
    tracker.process_video(
        args.input,
        args.output,
        max_objects=args.max_objects,
        skip_frames=args.skip_frames
    )


if __name__ == "__main__":
    main()
