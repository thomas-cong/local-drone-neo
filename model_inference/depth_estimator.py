#!/usr/bin/env python3
"""
Depth Estimation Video Processor
Uses Depth-Anything-V2 for monocular depth estimation
"""
import os
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'

import time
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image


class DepthEstimator:
    """Video depth estimation using Depth-Anything-V2"""
    
    def __init__(self, model="depth-anything/Depth-Anything-V2-small-hf", device=None):
        """
        Initialize depth estimation model
        
        Args:
            model: HuggingFace model checkpoint
            device: Device to use (cuda/cpu), auto-detected if None
        """
        print("Loading Depth-Anything-V2 model...")
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cuda" if device == 0 else "cpu"
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModelForDepthEstimation.from_pretrained(model).to(self.device)
        self.model.eval()
        
        print(f"Depth model ready on {self.device.upper()}")
    
    def estimate_depth(self, frame):
        """
        Estimate depth for a single frame
        
        Args:
            frame: Input frame (BGR numpy array)
            
        Returns:
            Depth map as numpy array (normalized 0-255)
        """
        # Convert BGR to RGB PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Prepare inputs
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get depth prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=pil_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to numpy and normalize to 0-255
        depth_np = prediction.squeeze().cpu().numpy()
        depth_normalized = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)
        
        return depth_uint8
    
    def colorize_depth(self, depth_map, colormap=cv2.COLORMAP_INFERNO):
        """
        Apply colormap to depth map for visualization
        
        Args:
            depth_map: Grayscale depth map (0-255)
            colormap: OpenCV colormap (default: INFERNO)
            
        Returns:
            Colorized depth map (BGR)
        """
        return cv2.applyColorMap(depth_map, colormap)
    
    def process_video(self, input_path, output_path, colormap=cv2.COLORMAP_INFERNO, 
                     skip_frames=1, show_side_by_side=False):
        """
        Process video and generate depth estimation output
        
        Args:
            input_path: Input video path
            output_path: Output video path
            colormap: OpenCV colormap for depth visualization
            skip_frames: Process every Nth frame (1 = all frames)
            show_side_by_side: Show original and depth side-by-side
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
        
        # Setup output
        output_width = width * 2 if show_side_by_side else width
        
        fourcc_options = [
            ('avc1', 'H.264 (hardware)'),
            ('h264', 'H.264 (software)'),
            ('mp4v', 'MPEG-4 (fallback)')
        ]
        
        out = None
        for fourcc_str, codec_name in fourcc_options:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))
            if out.isOpened():
                print(f"Using codec: {codec_name}")
                break
            out.release()
            out = None
        
        if out is None:
            raise IOError("Failed to create output with any codec")
        
        print(f"Processing video (every {skip_frames} frame(s))...")
        
        frame_idx = 0
        last_depth_colored = None
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Estimate depth for every Nth frame
                if frame_idx % skip_frames == 0:
                    depth_map = self.estimate_depth(frame)
                    depth_colored = self.colorize_depth(depth_map, colormap)
                    last_depth_colored = depth_colored
                else:
                    # Reuse last depth map for skipped frames
                    depth_colored = last_depth_colored
                
                # Create output frame
                if show_side_by_side and depth_colored is not None:
                    output_frame = np.hstack([frame, depth_colored])
                elif depth_colored is not None:
                    output_frame = depth_colored
                else:
                    output_frame = frame
                
                out.write(output_frame)
                
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
    
    def process_frame_stream(self, frame_generator):
        """
        Process frames from a generator (useful for camera streams)
        
        Args:
            frame_generator: Generator yielding BGR frames
            
        Yields:
            Tuple of (original_frame, depth_colored)
        """
        for frame in frame_generator:
            depth_map = self.estimate_depth(frame)
            depth_colored = self.colorize_depth(depth_map)
            yield frame, depth_colored


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Depth Estimation Video Processor")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-base-hf",
                       help="HuggingFace model checkpoint")
    parser.add_argument("--colormap", default="inferno", 
                       choices=['inferno', 'viridis', 'magma', 'jet', 'hot', 'cool'],
                       help="Colormap for depth visualization")
    parser.add_argument("--skip-frames", type=int, default=1,
                       help="Process every Nth frame (default=1, all frames)")
    parser.add_argument("--side-by-side", action="store_true",
                       help="Show original and depth side-by-side")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}")
        return
    
    # Map colormap names to OpenCV constants
    colormap_dict = {
        'inferno': cv2.COLORMAP_INFERNO,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'magma': cv2.COLORMAP_MAGMA,
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'cool': cv2.COLORMAP_COOL,
    }
    
    colormap = colormap_dict.get(args.colormap, cv2.COLORMAP_INFERNO)
    
    # Create estimator and process
    estimator = DepthEstimator(model=args.model)
    estimator.process_video(
        args.input,
        args.output,
        colormap=colormap,
        skip_frames=args.skip_frames,
        show_side_by_side=args.side_by_side
    )


if __name__ == "__main__":
    main()
