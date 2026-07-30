"""Region of Interest (ROI) and lane counting module."""

from typing import Dict, List, Tuple
import cv2
import numpy as np

class ZoneCounter:
    def __init__(self, rois: Dict[str, List[Tuple[float, float]]]):
        """
        rois: Dict mapping lane name to a list of (x,y) relative coordinates forming a polygon [0.0 - 1.0].
        """
        self.relative_rois = rois
        self.absolute_rois = None
        self.frame_shape = None

    def _ensure_absolute_rois(self, frame_shape: Tuple[int, ...]) -> None:
        """Lazily compute integer pixel polygons based on frame size."""
        if self.absolute_rois is not None and self.frame_shape == frame_shape:
            return
        self.frame_shape = frame_shape
        height, width = frame_shape[:2]
        self.absolute_rois = {}
        for name, points in self.relative_rois.items():
            abs_points = [(int(x * width), int(y * height)) for x, y in points]
            self.absolute_rois[name] = np.array(abs_points, np.int32).reshape((-1, 1, 2))

    def update(self, centroids: Dict[int, Tuple[float, float]], frame_shape: Tuple[int, ...]) -> Dict[str, int]:
        """
        Update counts based on centroids and the current frame shape.
        Returns the CURRENT count of vehicles inside the ROI zones.
        """
        self._ensure_absolute_rois(frame_shape)
        current_counts = {name: 0 for name in self.absolute_rois.keys()}
        
        for track_id, (x, y) in centroids.items():
            for name, poly in self.absolute_rois.items():
                # cv2.pointPolygonTest returns >0 if inside, 0 if on boundary, <0 if outside
                dist = cv2.pointPolygonTest(poly, (float(x), float(y)), measureDist=False)
                if dist >= 0:
                    current_counts[name] += 1
                    
        return current_counts

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw the configured ROIs on the frame for debugging."""
        self._ensure_absolute_rois(frame.shape)
        for name, poly in self.absolute_rois.items():
            # Draw polygon
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            # Draw label
            pt = tuple(poly[0][0])
            cv2.putText(frame, name, (int(pt[0]), int(pt[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

