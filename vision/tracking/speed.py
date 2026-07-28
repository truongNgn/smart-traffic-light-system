"""Speed estimation module based on centroid tracking."""

import collections
import math
from typing import Dict, Tuple

class SpeedEstimator:
    def __init__(self, history_size: int = 5):
        # track_id -> deque of (timestamp, (x, y))
        self.history: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=history_size)
        )

    def update(self, centroids: Dict[int, Tuple[float, float]], timestamp_s: float) -> float:
        """
        Update tracking history and calculate mean speed of all active tracks.
        Returns:
            mean_speed_mps (float)
        """
        current_speeds = []
        
        for track_id, (x, y) in centroids.items():
            self.history[track_id].append((timestamp_s, (x, y)))
            
            history = self.history[track_id]
            if len(history) >= 2:
                # Calculate speed between oldest and newest in history
                old_time, (old_x, old_y) = history[0]
                new_time, (new_x, new_y) = history[-1]
                
                dt = new_time - old_time
                if dt > 0:
                    dist = math.hypot(new_x - old_x, new_y - old_y)
                    speed_px_s = dist / dt
                    # Extremely rough conversion: assuming 1 pixel = 0.05 meters
                    speed_m_s = speed_px_s * 0.05
                    current_speeds.append(speed_m_s)
                    
        # Cleanup lost tracks
        active_ids = set(centroids.keys())
        for track_id in list(self.history.keys()):
            if track_id not in active_ids:
                del self.history[track_id]
                
        if not current_speeds:
            return 0.0
            
        return sum(current_speeds) / len(current_speeds)
