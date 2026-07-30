"""YOLO + ByteTrack integration (Engineer A)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from ultralytics import YOLO

from common.logging import get_logger

logger = get_logger(component="yolo_detector")


class YoloDetector:
    """Wrapper around Ultralytics YOLOv8 for detection and ByteTrack tracking."""

    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> None:
        """Initialize the YOLO model."""
        logger.info("Loading YOLO model", model_path=model_path)
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, Dict[str, int], Dict[int, tuple[float, float]]]:
        """
        Runs detection and tracking on a single frame.
        Returns:
            - Annotated frame (np.ndarray)
            - Class counts (dict: class_name -> count)
            - Centroids (dict: track_id -> (x_center, y_center))
        """
        # Run YOLO with integrated ByteTrack
        # persist=True keeps tracking IDs consistent across frames
        # tracker="bytetrack.yaml" uses the built-in ByteTrack
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        class_counts: Dict[str, int] = {}
        centroids: Dict[int, tuple[float, float]] = {}
        annotated_frame = frame.copy()

        if results and len(results) > 0:
            result = results[0]
            annotated_frame = result.plot()  # Draw bounding boxes and IDs
            
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # type: ignore
                track_ids = result.boxes.id.int().cpu().tolist()  # type: ignore
                class_ids = result.boxes.cls.int().cpu().tolist()  # type: ignore
                
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    class_name = self.model.names[class_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Compute centroid
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    centroids[track_id] = (float(cx), float(cy))

        return annotated_frame, class_counts, centroids
