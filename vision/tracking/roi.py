"""Region of Interest (ROI) and lane counting module."""

from typing import Dict, List, Tuple
import cv2
import numpy as np

class ZoneCounter:
    def __init__(self, rois: Dict[str, List[Tuple[int, int]]]):
        """
        rois: Dict mapping lane name to a list of (x,y) coordinates forming a polygon.
        """
        self.rois = {}
        for name, points in rois.items():
            # Convert to numpy array for cv2.pointPolygonTest
            self.rois[name] = np.array(points, np.int32).reshape((-1, 1, 2))

    def update(self, centroids: Dict[int, Tuple[float, float]]) -> Dict[str, int]:
        """
        Update counts based on centroids.
        Returns the CURRENT count of vehicles inside the ROI zones.
        For traffic lights, we usually care about the current queue/count in the lane.
        """
        current_counts = {name: 0 for name in self.rois.keys()}
        
        for track_id, (x, y) in centroids.items():
            for name, poly in self.rois.items():
                # cv2.pointPolygonTest returns >0 if inside, 0 if on boundary, <0 if outside
                dist = cv2.pointPolygonTest(poly, (float(x), float(y)), measureDist=False)
                if dist >= 0:
                    current_counts[name] += 1
                    
        return current_counts

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw the configured ROIs on the frame for debugging."""
        for name, poly in self.rois.items():
            # Draw polygon
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            # Draw label
            pt = tuple(poly[0][0])
            cv2.putText(frame, name, (int(pt[0]), int(pt[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
