"""Video Stream Simulator (Engineer A)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

from common.logging import get_logger

logger = get_logger(component="video_simulator")


class VideoStreamSimulator:
    """Simulates an RTSP stream by reading and looping a local .mp4 file."""

    def __init__(self, video_path: str | Path, camera_id: str, target_fps: float = 30.0) -> None:
        self.video_path = str(video_path)
        self.camera_id = camera_id
        self.target_fps = target_fps
        self._cap = cv2.VideoCapture(self.video_path)
        
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {self.video_path}")
            
        # Optional: read actual FPS
        self.actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info("Initialized video simulator", video_path=self.video_path, fps=self.actual_fps)

    def stream(self) -> Iterator[Tuple[float, np.ndarray]]:
        """
        Yields (timestamp_s, frame) endlessly. Loops the video when it ends.
        Throttles output to match target_fps.
        """
        frame_time = 1.0 / self.target_fps
        
        while True:
            start_time = time.time()
            
            ret, frame = self._cap.read()
            if not ret:
                # Loop back to beginning
                logger.debug("Restarting video loop", camera_id=self.camera_id)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    break  # Unrecoverable
            
            yield time.time(), frame
            
            elapsed = time.time() - start_time
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def close(self) -> None:
        if self._cap:
            self._cap.release()
