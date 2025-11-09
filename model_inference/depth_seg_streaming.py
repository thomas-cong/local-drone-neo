#\!/usr/bin/env python3
"""
Real-time Depth + Segmentation Streaming
Optimized for live camera feeds with per-frame processing.
Prioritizes speed and smoothness over tracking quality.
"""
import os
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'

import time
import cv2
import numpy as np
import torch
import json
from collections import deque
from tqdm import tqdm
from depth_estimator import DepthEstimator
from edgetam_tracker import EdgeTAMTracker
from object_bounder_simple import ObjectBounder

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

DEFAULT_MQTT_ENABLED = os.environ.get("MQTT_ENABLE", "1") != "0"
DEFAULT_MQTT_HOST = os.environ.get("MQTT_HOST", "127.0.0.1")
DEFAULT_MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
DEFAULT_MQTT_TOPIC = os.environ.get("MQTT_TOPIC", "depth/seg")
DEFAULT_MQTT_CLIENT_ID = os.environ.get("MQTT_CLIENT_ID", "depth-seg-publisher")
DEFAULT_MQTT_QOS = int(os.environ.get("MQTT_QOS", "0"))


class StreamingDepthSegmentation:
    """
    Real-time streaming processor with per-frame segmentation
    
    Fast single-frame processing without temporal tracking overhead.
    Ideal for live camera streams where speed > tracking quality.
    """
    
    def __init__(self, depth_model="depth-anything/Depth-Anything-V2-Small-hf",
                 edgetam_checkpoint=None, edgetam_config=None,
                 process_resolution=(320, 180), process_every_n=3,
                 max_objects=2, reinit_every=100, json_output=None, json_interval=5,
                 matrix_resolution=(160, 90), min_coverage=0.01, max_coverage=0.4,
                 save_frames_on_exit=0, save_frames_dir="saved_frames",
                 mqtt_enabled=None, mqtt_host=None, mqtt_port=None,
                 mqtt_topic=None, mqtt_client_id=None, mqtt_qos=None,
                 enable_object_detection=True, object_detect_every_n=5,
                 object_model_size='n', object_threshold=0.5, object_overlap_threshold=0.6,
                 save_json_file=None):
        """
        Initialize streaming processor
        
        Args:
            depth_model: Depth model (use Small for speed)
            process_resolution: Processing resolution (lower = faster)
            process_every_n: Process depth/seg every N frames (higher = faster)
            max_objects: Number of objects to track
            reinit_every: Re-detect objects every N frames (0 = never)
            json_output: Deprecated; retained for compatibility, acts as alias for mqtt_topic
            json_interval: Publish JSON every N frames (default: 5)
            matrix_resolution: Resolution for segmentation matrix (width, height)
            min_coverage: Minimum object coverage (fraction, e.g., 0.01 = 1%)
            max_coverage: Maximum object coverage (fraction, e.g., 0.3 = 30%)
            save_frames_on_exit: Number of recent frames to save on exit (default: 0, disabled)
            save_frames_dir: Directory to save frames on exit (when enabled)
            mqtt_enabled: Force-enable or disable MQTT publishing (None = use env default)
            mqtt_host: MQTT broker host (default from MQTT_HOST env or 127.0.0.1)
            mqtt_port: MQTT broker port (default from MQTT_PORT env or 1883)
            mqtt_topic: MQTT topic for depth payloads (default from MQTT_TOPIC env or 'depth/seg')
            mqtt_client_id: MQTT client ID (default from MQTT_CLIENT_ID env)
            mqtt_qos: MQTT QoS level (default from MQTT_QOS env or 0)
            enable_object_detection: Enable YOLO object detection filtering (default: True)
            object_detect_every_n: Run object detection every N frames (default: 5)
            object_model_size: YOLO model size ('n'=nano, 's'=small, etc.) (default: 'n')
            object_threshold: YOLO confidence threshold (default: 0.5)
            object_overlap_threshold: Minimum overlap ratio to include segment (default: 0.6)
            save_json_file: Path to save JSON payloads to file (default: None, disabled)
        """
        self.process_resolution = process_resolution
        self.process_every_n = process_every_n
        self.max_objects = max_objects
        self.reinit_every = reinit_every
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        
        print("Loading models for real-time streaming...")
        
        # Load depth estimator
        self.depth_estimator = DepthEstimator(model=depth_model, device=0)
        
        # Load EdgeTAM for segmentation
        self.tracker = EdgeTAMTracker(
            checkpoint=edgetam_checkpoint,
            config=edgetam_config,
            device="cuda"
        )
        
        # Load object detector (optional)
        self.enable_object_detection = enable_object_detection
        self.object_detect_every_n = object_detect_every_n
        self.object_overlap_threshold = object_overlap_threshold
        self.object_detector = None
        self.last_detections = None
        self.last_raw_frame = None
        
        # YOLO COCO class IDs for target objects: person, car, laptop
        # Note: 'building' is not in standard COCO, so we use: person(0), car(2), laptop(63)
        self.target_class_ids = {0, 2, 63}  # person, car, laptop
        self.target_class_names = {0: 'person', 2: 'car', 63: 'laptop'}
        
        if self.enable_object_detection:
            print(f"Loading YOLO object detector (model size: {object_model_size})...")
            self.object_detector = ObjectBounder(
                model_size=object_model_size,
                threshold=object_threshold
            )
        
        # Streaming state
        self.frame_idx = 0
        self.last_depth = None
        self.last_depth_map = None  # Raw depth values
        self.last_masks = None
        self.last_filtered_masks = None  # Masks filtered by object detection
        self.last_object_mappings = None  # Maps obj_id to detected bbox
        self.prompts = None
        self.initialized = False
        
        # JSON output configuration
        self.json_interval = json_interval
        self.matrix_resolution = matrix_resolution
        self.mqtt_enabled = DEFAULT_MQTT_ENABLED if mqtt_enabled is None else mqtt_enabled
        if json_output is not None and mqtt_topic is None:
            print("Warning: 'json_output' is deprecated and will be removed; using its value as the MQTT topic.")
            mqtt_topic = json_output
        self.mqtt_host = mqtt_host or DEFAULT_MQTT_HOST
        self.mqtt_port = mqtt_port or DEFAULT_MQTT_PORT
        self.mqtt_topic = mqtt_topic or DEFAULT_MQTT_TOPIC
        self.mqtt_client_id = mqtt_client_id or DEFAULT_MQTT_CLIENT_ID
        self.mqtt_qos = mqtt_qos if mqtt_qos is not None else DEFAULT_MQTT_QOS
        self.mqtt_client = None
        self.mqtt_active = False
        
        # JSON file output
        self.save_json_file = save_json_file
        self.json_file_handle = None
        self.json_entries_written = 0
        if self.save_json_file:
            # Create directory if needed
            json_dir = os.path.dirname(self.save_json_file)
            if json_dir:
                os.makedirs(json_dir, exist_ok=True)
            # Open file and write array start
            self.json_file_handle = open(self.save_json_file, 'w')
            self.json_file_handle.write('[\n')
            print(f"Writing JSON data to file: {self.save_json_file}")
        
        # Frame buffer for saving on exit
        self.save_frames_on_exit = save_frames_on_exit
        self.save_frames_dir = save_frames_dir
        self.frame_buffer = deque(maxlen=save_frames_on_exit) if save_frames_on_exit > 0 else None
        
        # Performance tracking
        self.fps_tracker = deque(maxlen=30)
        self.last_time = time.time()
        
        self._setup_mqtt()

        print("✓ Streaming processor ready\n")
    
    def _calculate_mask_bbox_overlap(self, mask, bbox):
        """
        Calculate overlap ratio between mask and bounding box
        
        Args:
            mask: Binary mask (numpy array or torch tensor)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Overlap ratio (0-1)
        """
        # Convert mask to numpy if needed
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Convert to binary mask
        mask_bool = mask > 0.5
        
        # Get bbox coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Create bbox mask
        bbox_mask = np.zeros_like(mask_bool, dtype=bool)
        bbox_mask[y1:y2, x1:x2] = True
        
        # Calculate overlap
        intersection = np.logical_and(mask_bool, bbox_mask).sum()
        mask_area = mask_bool.sum()
        
        if mask_area == 0:
            return 0.0
        
        overlap_ratio = intersection / mask_area
        return overlap_ratio
    
    def _filter_masks_by_objects(self, masks, detections, frame_shape):
        """
        Filter masks to only include those with sufficient overlap with target objects
        
        Args:
            masks: Dict of masks {obj_id: mask}
            detections: supervision.Detections object from YOLO
            frame_shape: Shape of frame for resizing masks
            
        Returns:
            Tuple of (filtered_masks, object_mappings)
            filtered_masks: Dict of filtered masks
            object_mappings: Dict mapping obj_id to bbox info
        """
        if not masks or detections is None or len(detections) == 0:
            return masks, {}
        
        filtered_masks = {}
        object_mappings = {}
        
        h, w = frame_shape[:2]
        
        for obj_id, mask in masks.items():
            if mask is None:
                continue
            
            # Resize mask to frame size if needed
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask
            
            if mask_np.shape[:2] != (h, w):
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                mask_resized = mask_np
            
            # Check overlap with each detection
            best_overlap = 0.0
            best_detection = None
            best_bbox = None
            
            for i, (bbox, class_id, confidence) in enumerate(zip(
                detections.xyxy, detections.class_id, detections.confidence
            )):
                # Only consider target classes
                if class_id not in self.target_class_ids:
                    continue
                
                # Calculate overlap
                overlap = self._calculate_mask_bbox_overlap(mask_resized, bbox)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_detection = {
                        'class_id': int(class_id),
                        'class_name': self.target_class_names.get(int(class_id), 'unknown'),
                        'confidence': float(confidence),
                        'bbox': bbox.tolist()
                    }
                    best_bbox = bbox
            
            # Include mask if overlap exceeds threshold
            if best_overlap >= self.object_overlap_threshold:
                filtered_masks[obj_id] = mask
                object_mappings[obj_id] = {
                    'overlap': float(best_overlap),
                    'detection': best_detection
                }
        
        return filtered_masks, object_mappings
    
    def _convert_detections_to_prompts(self, detections, frame_shape, target_shape):
        """
        Convert YOLO bounding boxes to EdgeTAM prompts
        
        Args:
            detections: supervision.Detections from YOLO
            frame_shape: Original frame shape (H, W)
            target_shape: Target depth frame shape (H, W)
            
        Returns:
            List of prompt dicts with 'points' and 'labels'
        """
        if detections is None or len(detections) == 0:
            return []
        
        prompts = []
        orig_h, orig_w = frame_shape[:2]
        target_h, target_w = target_shape[:2]
        
        # Scale factors
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        for bbox, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            # Only use target classes
            if class_id not in self.target_class_ids:
                continue
            
            # Get bbox center in original coordinates
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Scale to target resolution
            scaled_x = center_x * scale_x
            scaled_y = center_y * scale_y
            
            # Create prompt with center point
            prompt = {
                'points': np.array([[scaled_x, scaled_y]], dtype=np.float32),
                'labels': np.array([1], dtype=np.int32),  # 1 = foreground
                'class_id': int(class_id),
                'class_name': self.target_class_names.get(int(class_id), 'unknown'),
                'confidence': float(confidence),
                'bbox': bbox.tolist()
            }
            prompts.append(prompt)
        
        return prompts
    
    def _process_depth_and_segment(self, frame, detections=None, frame_shape=None):
        """Process depth estimation and segmentation on a frame"""
        # Estimate depth
        depth_map = self.depth_estimator.estimate_depth(frame)
        depth_colored = self.depth_estimator.colorize_depth(depth_map)
        
        # Store raw depth map for distance calculations
        self.last_depth_map = depth_map
        
        # Create prompts from YOLO detections if object detection is enabled
        if self.enable_object_detection and detections is not None and frame_shape is not None:
            self.prompts = self._convert_detections_to_prompts(
                detections, frame_shape, frame.shape
            )
            if not self.initialized and self.prompts:
                print(f"Using {len(self.prompts)} YOLO detections as segmentation prompts")
                self.initialized = True
        else:
            # Fallback to auto-detection if object detection is disabled
            if not self.initialized or (self.reinit_every > 0 and 
                                        self.frame_idx % (self.reinit_every * self.process_every_n) == 0):
                if not self.initialized:
                    print("Initializing object detection...")
                self.prompts = self.tracker.find_objects(depth_colored, 
                                                                  max_objects=self.max_objects,
                                                                  min_coverage=self.min_coverage,
                                                                  max_coverage=self.max_coverage)
                self.initialized = True
        
        # Segment using single-frame prediction
        masks = None
        if self.prompts:
            masks = self._segment_frame(depth_colored)
        
        return depth_colored, masks
    
    def _segment_frame(self, frame):
        """Fast single-frame segmentation"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.tracker.image_predictor.set_image(frame_rgb)
        
        masks_dict = {}
        autocast_dtype = torch.float16
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
            for obj_id, prompt in enumerate(self.prompts):
                try:
                    masks, _, _ = self.tracker.image_predictor.predict(
                        point_coords=prompt['points'],
                        point_labels=prompt['labels'],
                        multimask_output=False,
                    )
                    if len(masks) > 0:
                        masks_dict[obj_id] = masks[0]
                except:
                    continue
        
        return masks_dict if masks_dict else None
    
    def _calculate_segment_distances(self, masks):
        """Calculate mean depth for each segment"""
        if not masks or self.last_depth_map is None:
            return {}
        
        distances = {}
        for obj_id, mask in masks.items():
            if mask is None:
                distances[obj_id] = None
                continue
            
            # Convert mask to bool array
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask_bool = mask > 0.5
            
            # Get depth values within mask
            masked_depth = self.last_depth_map[mask_bool]
            
            if len(masked_depth) > 0:
                # Use median depth as representative distance
                distances[obj_id] = float(np.median(masked_depth))
            else:
                distances[obj_id] = None
        
        return distances
    
    def _create_depth_matrix(self):
        """Create depth matrix at target resolution"""
        if self.last_depth_map is None:
            return None
        
        matrix_w, matrix_h = self.matrix_resolution
        
        # Resize depth map to matrix resolution
        depth_resized = cv2.resize(self.last_depth_map, (matrix_w, matrix_h), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Convert to list of lists (round to 1 decimal for smaller JSON)
        return depth_resized.astype(np.int32).tolist()
    
    def _create_segmentation_matrix(self, masks, original_shape):
        """Create segmentation matrix from masks"""
        if not masks:
            return None
        
        # Initialize matrix with -1 (no object)
        matrix_w, matrix_h = self.matrix_resolution
        seg_matrix = np.full((matrix_h, matrix_w), -1, dtype=np.int16)
        
        # Get original dimensions
        orig_h, orig_w = original_shape[:2]
        
        # Process each mask in order (later masks can overwrite earlier ones)
        for obj_id in sorted(masks.keys()):
            mask = masks[obj_id]
            if mask is None:
                continue
            
            # Convert mask to numpy if needed
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Resize mask to original frame size if needed
            if mask.shape[:2] != (orig_h, orig_w):
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # Resize mask to matrix resolution
            mask_resized = cv2.resize(mask, (matrix_w, matrix_h), interpolation=cv2.INTER_LINEAR)
            mask_bool = mask_resized > 0.5
            
            # Set object ID where mask is true
            seg_matrix[mask_bool] = obj_id
        
        return seg_matrix.tolist()
    
    def _setup_mqtt(self):
        """Initialize MQTT client if available and enabled."""
        if not self.mqtt_enabled:
            return
        if self.mqtt_active and self.mqtt_client:
            return
        if not MQTT_AVAILABLE:
            print("paho-mqtt not installed; MQTT publishing disabled.")
            self.mqtt_enabled = False
            return
        try:
            self.mqtt_client = mqtt.Client(client_id=self.mqtt_client_id)
            self.mqtt_client.connect(self.mqtt_host, self.mqtt_port, keepalive=60)
            self.mqtt_client.loop_start()
            self.mqtt_active = True
            print(f"MQTT publishing depth data to {self.mqtt_host}:{self.mqtt_port} topic '{self.mqtt_topic}' (QoS {self.mqtt_qos})")
        except Exception as exc:
            self.mqtt_client = None
            self.mqtt_active = False
            self.mqtt_enabled = False
            print(f"Failed to connect to MQTT broker: {exc}")
    
    def _shutdown_mqtt(self):
        """Stop MQTT loop and disconnect."""
        if not self.mqtt_client:
            return
        try:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        except Exception:
            pass
        finally:
            self.mqtt_client = None
            self.mqtt_active = False
    
    def _publish_json(self, entry):
        """Publish a single JSON payload over MQTT and/or write to file."""
        # Write to file if enabled
        if self.json_file_handle:
            try:
                # Add comma before entry if not the first one
                if self.json_entries_written > 0:
                    self.json_file_handle.write(',\n')
                # Write JSON entry with indentation
                json.dump(entry, self.json_file_handle, indent=2)
                self.json_file_handle.flush()  # Ensure it's written
                self.json_entries_written += 1
            except Exception as exc:
                print(f"JSON file write failed: {exc}")
        
        # Publish to MQTT if enabled
        if self.mqtt_active and self.mqtt_client:
            try:
                payload = json.dumps(entry)
                self.mqtt_client.publish(self.mqtt_topic, payload, qos=self.mqtt_qos)
            except Exception as exc:
                print(f"MQTT publish failed: {exc}")
    
    def _add_json_entry(self, frame_num, timestamp, distances, masks, original_shape):
        """Publish depth distances entry via MQTT and/or write to file."""
        if not self.mqtt_active and not self.json_file_handle:
            return
        
        entry = {
            'frame': frame_num,
            'timestamp': round(timestamp, 3),
            'matrix_resolution': {
                'width': self.matrix_resolution[0], 
                'height': self.matrix_resolution[1]
            }
        }
        
        # Add depth matrix
        depth_matrix = self._create_depth_matrix()
        if depth_matrix is not None:
            entry['depth_matrix'] = depth_matrix
        
        # Filter masks to only include those with object detections
        # This ensures segmentation matrix only has segments that map to detected objects
        masks_to_send = masks
        if self.enable_object_detection and self.last_object_mappings:
            # Only include masks that have object mappings (detected objects)
            masks_to_send = {obj_id: mask for obj_id, mask in masks.items() 
                           if obj_id in self.last_object_mappings}
        
        # Add segmentation matrix (only for important/detected objects)
        seg_matrix = self._create_segmentation_matrix(masks_to_send, original_shape)
        if seg_matrix is not None:
            entry['segmentation_matrix'] = seg_matrix
        
        # Add object-to-bbox mappings (includes class, confidence, bbox, overlap)
        if self.enable_object_detection and self.last_object_mappings:
            entry['object_detections'] = {}
            for obj_id, mapping in self.last_object_mappings.items():
                entry['object_detections'][f'segment_{obj_id}'] = {
                    'overlap_ratio': mapping['overlap'],
                    'detected_object': {
                        'class_name': mapping['detection']['class_name'],
                        'class_id': mapping['detection']['class_id'],
                        'confidence': mapping['detection']['confidence'],
                        'bbox': mapping['detection']['bbox']  # [x1, y1, x2, y2]
                    }
                }
        
        self._publish_json(entry)
    
    def _close_json_file(self):
        """Close JSON file and write closing bracket."""
        if self.json_file_handle:
            try:
                self.json_file_handle.write('\n]\n')
                self.json_file_handle.close()
                print(f"\nWrote {self.json_entries_written} JSON entries to: {self.save_json_file}")
            except Exception as exc:
                print(f"Error closing JSON file: {exc}")
            finally:
                self.json_file_handle = None
    
    def _save_buffered_frames(self):
        """Save buffered frames to disk"""
        if self.frame_buffer is None or len(self.frame_buffer) == 0:
            return
        
        # Create output directory
        os.makedirs(self.save_frames_dir, exist_ok=True)
        
        # Save each frame
        saved_count = 0
        for frame_idx, frame in self.frame_buffer:
            output_path = os.path.join(self.save_frames_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        print(f"\nSaved {saved_count} frames to: {self.save_frames_dir}/")
    
    def process_frame(self, frame):
        """
        Process single frame with adaptive quality
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Processed frame with depth + segmentation overlay
        """
        original_shape = frame.shape
        
        # Store raw frame for object detection
        self.last_raw_frame = frame.copy()
        
        # Run object detection every N frames (on raw frame before resizing)
        if self.enable_object_detection and self.object_detector is not None:
            if self.frame_idx % self.object_detect_every_n == 0:
                self.last_detections = self.object_detector.predict(frame)
        
        # Resize for processing
        frame_small = cv2.resize(frame, self.process_resolution)
        
        # Process depth + segmentation every N frames
        if self.frame_idx % self.process_every_n == 0:
            # Pass detections to use as prompts
            self.last_depth, self.last_masks = self._process_depth_and_segment(
                frame_small, 
                detections=self.last_detections if self.enable_object_detection else None,
                frame_shape=original_shape
            )
            
            # When using YOLO prompts, masks directly correspond to detections
            # So we don't need to filter - just build the mapping
            if self.enable_object_detection and self.last_detections is not None and self.last_masks:
                self.last_filtered_masks = self.last_masks
                # Build object mappings from prompts
                self.last_object_mappings = {}
                for obj_id, prompt in enumerate(self.prompts):
                    if obj_id in self.last_masks:
                        self.last_object_mappings[obj_id] = {
                            'overlap': 1.0,  # 100% by design since we used bbox as prompt
                            'detection': {
                                'class_name': prompt['class_name'],
                                'class_id': prompt['class_id'],
                                'confidence': prompt['confidence'],
                                'bbox': prompt['bbox']
                            }
                        }
            else:
                self.last_filtered_masks = self.last_masks
                self.last_object_mappings = {}
        
        # Use cached results for skipped frames
        depth_colored = self.last_depth if self.last_depth is not None else frame_small
        masks = self.last_filtered_masks if self.last_filtered_masks is not None else self.last_masks
        
        # Apply masks (only filtered masks are drawn)
        if masks:
            output = self.tracker.draw_masks(depth_colored, masks)
        else:
            output = depth_colored
        
        # Upscale to original resolution
        h, w = original_shape[:2]
        output_full = cv2.resize(output, (w, h))
        
        # Update FPS tracking
        current_time = time.time()
        self.fps_tracker.append(1.0 / (current_time - self.last_time))
        self.last_time = current_time
        
        # Add to JSON at specified interval (use filtered masks)
        if (self.mqtt_active or self.json_file_handle) and masks and self.frame_idx % self.json_interval == 0:
            distances = self._calculate_segment_distances(masks)
            self._add_json_entry(self.frame_idx, current_time, distances, masks, original_shape)
        
        # Save to frame buffer if enabled
        if self.frame_buffer is not None:
            self.frame_buffer.append((self.frame_idx, output_full.copy()))
        
        # No overlay - just return the masked depth frame
        self.frame_idx += 1
        return output_full
    
    def _open_capture(self, source, api_preference=cv2.CAP_ANY):
        """
        Attempt to open a cv2.VideoCapture for the given source, ensuring it's ready.
        Returns the capture object or raises an IOError.
        """
        cap = cv2.VideoCapture(source, api_preference)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        return cap
    
    def stream_from_camera(self, camera_id=0, display=True, use_csi=False, width=1280, height=720, fps=30):
        """
        Stream from camera with real-time processing
        
        Args:
            camera_id: Camera device ID (ignored if use_csi=True)
            display: Show output window
            use_csi: Use CSI camera via GStreamer (nvarguscamerasrc)
            width: Camera width (only for CSI)
            height: Camera height (only for CSI)
            fps: Camera FPS (only for CSI)
        """
        if use_csi:
            # GStreamer pipeline for CSI camera (Jetson)
            cam_pipe = (
                f"nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
                f"nvvidconv ! video/x-raw,format=BGRx ! "
                f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
            )
            cap = cv2.VideoCapture(cam_pipe, cv2.CAP_GSTREAMER)
        else:
            cap = self._open_capture(camera_id)
        
        # Get camera info
        if not use_csi:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera: {width}x{height} @ {fps:.1f} FPS")
        print("Press 'q' to quit\n")
        
        self._setup_mqtt()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                output = self.process_frame(frame)
                
                # Display
                if display:
                    cv2.imshow('Depth + Segmentation Stream', output)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            # Save buffered frames
            self._save_buffered_frames()
            
            # Close JSON file
            self._close_json_file()
            
            self._shutdown_mqtt()
            
            avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
            print(f"\n✓ Stream ended. Average FPS: {avg_fps:.1f}")
    
    def stream_from_rtsp(self, rtsp_url, display=True, reconnect=True, reconnect_interval=3, max_retries=None):
        """
        Stream from RTSP source with optional reconnect logic
        
        Args:
            rtsp_url: RTSP URL string
            display: Show output window
            reconnect: Attempt to reconnect if stream drops
            reconnect_interval: Seconds between reconnect attempts
            max_retries: Maximum reconnect attempts (None = infinite)
        """
        print(f"Connecting to RTSP stream: {rtsp_url}")
        retries = 0
        
        self._setup_mqtt()
        
        while True:
            try:
                cap = self._open_capture(rtsp_url, cv2.CAP_FFMPEG)
            except IOError as exc:
                print(exc)
                if not reconnect:
                    raise
                retries += 1
                if max_retries is not None and retries > max_retries:
                    raise IOError(f"Failed to connect to RTSP stream after {max_retries} attempts")
                print(f"Retrying in {reconnect_interval}s...")
                time.sleep(reconnect_interval)
                continue
            
            print("RTSP stream connected")
            retries = 0
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("RTSP stream read failed")
                        break
                    
                    output = self.process_frame(frame)
                    
                    if display:
                        cv2.imshow('Depth + Segmentation Stream', output)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt
            except KeyboardInterrupt:
                print("\nStopped by user")
                break
            finally:
                cap.release()
                if display:
                    cv2.destroyAllWindows()
            
            if not reconnect:
                break
            
            retries += 1
            if max_retries is not None and retries > max_retries:
                print(f"Exceeded maximum reconnect attempts ({max_retries}).")
                break
            print(f"Reconnecting in {reconnect_interval}s...")
            time.sleep(reconnect_interval)
        
        # Save buffered frames
        self._save_buffered_frames()
        
        # Close JSON file
        self._close_json_file()
        
        self._shutdown_mqtt()
        
        avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
        print(f"\n✓ RTSP stream ended. Average FPS: {avg_fps:.1f}")
    
    def stream_from_video(self, video_path, output_path=None, display=True):
        """
        Stream from video file
        
        Args:
            video_path: Input video path
            output_path: Optional output path
            display: Show output window
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_known = total_frames > 0
        
        # Setup output writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if total_known:
            print(f"Processing video: {width}x{height} @ {fps:.1f} FPS ({total_frames} frames)")
        else:
            print(f"Processing video stream: {width}x{height} @ {fps:.1f} FPS (unknown length)")
        if display:
            print("Press 'q' to quit\n")
        
        self._setup_mqtt()
        
        try:
            pbar = tqdm(total=total_frames if total_known else None,
                       desc="Processing", unit="frame",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                output = self.process_frame(frame)
                
                # Save
                if out:
                    out.write(output)
                
                # Display
                if display:
                    cv2.imshow('Depth + Segmentation Stream', output)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Update progress
                if pbar:
                    pbar.update(1)
                    avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
                    pbar.set_postfix({'FPS': f'{avg_fps:.1f}'})
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            if pbar:
                pbar.close()
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()

            # Save buffered frames
            self._save_buffered_frames()
            
            # Close JSON file
            self._close_json_file()
            
            self._shutdown_mqtt()
            
            avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
            print(f"\n✓ Done! Average FPS: {avg_fps:.1f}")
            if output_path:
                print(f"Output: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Depth + Segmentation Streaming')
    parser.add_argument('--source', type=str, default='0',
                       help='Camera ID, video file path, or RTSP URL (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--resolution', type=str, default='320x180',
                       help='Processing resolution (default: 320x180)')
    parser.add_argument('--process-every', type=int, default=3,
                       help='Process every N frames (default: 3)')
    parser.add_argument('--max-objects', type=int, default=2,
                       help='Maximum objects to detect (default: 2)')
    parser.add_argument('--reinit-every', type=int, default=0,
                       help='Re-detect objects every N frames, 0=never (default: 0)')
    parser.add_argument('--depth-model', type=str, 
                       default='depth-anything/Depth-Anything-V2-Small-hf',
                       help='Depth model to use')
    parser.add_argument('--json-interval', type=int, default=5,
                       help='Publish JSON every N frames (default: 5)')
    parser.add_argument('--matrix-resolution', type=str, default='160x90',
                       help='Segmentation matrix resolution (default: 160x90)')
    parser.add_argument('--min-coverage', type=float, default=0.01,
                       help='Minimum object coverage fraction (default: 0.01 = 1%%)')
    parser.add_argument('--max-coverage', type=float, default=0.4,
                       help='Maximum object coverage fraction (default: 0.4 = 40%%)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window')
    parser.add_argument('--save-frames', type=int, default=0,
                       help='Number of recent frames to save on exit (default: 0, disabled)')
    parser.add_argument('--save-frames-dir', type=str, default='saved_frames',
                       help='Directory to save frames on exit (default: saved_frames)')
    parser.add_argument('--use-csi', action='store_true',
                       help='Use CSI camera via GStreamer (Jetson only)')
    parser.add_argument('--cam-width', type=int, default=1280,
                       help='Camera width for CSI (default: 1280)')
    parser.add_argument('--cam-height', type=int, default=720,
                       help='Camera height for CSI (default: 720)')
    parser.add_argument('--cam-fps', type=int, default=30,
                       help='Camera FPS for CSI (default: 30)')
    parser.add_argument('--mqtt-topic', dest='mqtt_topic', type=str, default=None,
                       help="MQTT topic for depth streaming payloads (default: env MQTT_TOPIC or 'depth/seg')")
    parser.add_argument('--json-output', dest='mqtt_topic', type=str,
                       help=argparse.SUPPRESS)
    parser.add_argument('--mqtt-host', type=str, default=None,
                       help='MQTT broker host (default: env MQTT_HOST or 127.0.0.1)')
    parser.add_argument('--mqtt-port', type=int, default=None,
                       help='MQTT broker port (default: env MQTT_PORT or 1883)')
    parser.add_argument('--mqtt-client-id', type=str, default=None,
                       help='MQTT client ID (default: env MQTT_CLIENT_ID or depth-seg-publisher)')
    parser.add_argument('--mqtt-qos', type=int, default=None,
                       help='MQTT QoS level (default: env MQTT_QOS or 0)')
    parser.add_argument('--mqtt-enable', dest='mqtt_enable', action='store_true',
                       help='Force-enable MQTT publishing (overrides MQTT_ENABLE env)')
    parser.add_argument('--mqtt-disable', dest='mqtt_enable', action='store_false',
                       help='Disable MQTT publishing')
    parser.add_argument('--enable-object-detection', action='store_true', default=True,
                       help='Enable YOLO object detection filtering (default: enabled)')
    parser.add_argument('--disable-object-detection', dest='enable_object_detection', action='store_false',
                       help='Disable object detection filtering')
    parser.add_argument('--object-detect-every', type=int, default=5,
                       help='Run object detection every N frames (default: 5)')
    parser.add_argument('--object-model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size: n(ano), s(mall), m(edium), l(arge), x(large) (default: n)')
    parser.add_argument('--object-threshold', type=float, default=0.5,
                       help='YOLO confidence threshold (default: 0.5)')
    parser.add_argument('--object-overlap-threshold', type=float, default=0.6,
                       help='Minimum overlap ratio to include segment (default: 0.6 = 60%%)')
    parser.add_argument('--save-json-file', type=str, default=None,
                       help='Save JSON payloads to file (default: None, disabled)')
    parser.set_defaults(mqtt_enable=None)
    
    args = parser.parse_args()
    
    # Parse resolution
    res_parts = args.resolution.lower().split('x')
    target_size = (int(res_parts[0]), int(res_parts[1]))
    
    # Parse matrix resolution
    matrix_parts = args.matrix_resolution.lower().split('x')
    matrix_size = (int(matrix_parts[0]), int(matrix_parts[1]))
    
    # Create processor
    processor = StreamingDepthSegmentation(
        depth_model=args.depth_model,
        process_resolution=target_size,
        process_every_n=args.process_every,
        max_objects=args.max_objects,
        reinit_every=args.reinit_every,
        json_interval=args.json_interval,
        matrix_resolution=matrix_size,
        min_coverage=args.min_coverage,
        max_coverage=args.max_coverage,
        save_frames_on_exit=args.save_frames,
        save_frames_dir=args.save_frames_dir,
        mqtt_enabled=args.mqtt_enable,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        mqtt_topic=args.mqtt_topic,
        mqtt_client_id=args.mqtt_client_id,
        mqtt_qos=args.mqtt_qos,
        enable_object_detection=args.enable_object_detection,
        object_detect_every_n=args.object_detect_every,
        object_model_size=args.object_model_size,
        object_threshold=args.object_threshold,
        object_overlap_threshold=args.object_overlap_threshold,
        save_json_file=args.save_json_file
    )
    
    # Determine source type
    source = args.source.strip()
    
    if source.lower().startswith("rtsp://"):
        processor.stream_from_rtsp(source, display=not args.no_display)
        return
    
    try:
        camera_id = int(source)
    except ValueError:
        processor.stream_from_video(source, args.output, display=not args.no_display)
        return
    
    processor.stream_from_camera(
        camera_id,
        display=not args.no_display,
        use_csi=args.use_csi,
        width=args.cam_width,
        height=args.cam_height,
        fps=args.cam_fps
    )


if __name__ == "__main__":
    main()
