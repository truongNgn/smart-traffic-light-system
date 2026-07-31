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

    def __init__(self, video_path: str | Path, camera_id: str, target_fps: float = 30.0, loop: bool = True) -> None:
        self.video_path = str(video_path)
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.loop = loop
        self.is_rtsp = self.video_path.startswith("rtsp://") or self.video_path.startswith("http://")
        
        max_retries = 15 if self.is_rtsp else 1
        for attempt in range(max_retries):
            self._cap = cv2.VideoCapture(self.video_path)
            if self._cap.isOpened():
                break
            if self.is_rtsp:
                logger.warning("Failed to open stream, retrying...", attempt=attempt+1, max_retries=max_retries, path=self.video_path)
                time.sleep(2)
                
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video/stream: {self.video_path}")
            
        # Optional: read actual FPS
        self.actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info("Initialized video simulator", video_path=self.video_path, fps=self.actual_fps, is_rtsp=self.is_rtsp)

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
                if self.is_rtsp:
                    logger.warning("RTSP stream disconnected", camera_id=self.camera_id)
                    # For RTSP, try to reconnect
                    time.sleep(1)
                    self._cap = cv2.VideoCapture(self.video_path)
                    continue

                if not self.loop:
                    logger.info("End of video stream reached", camera_id=self.camera_id)
                    break
                # Loop back to beginning
                logger.debug("Restarting video loop", camera_id=self.camera_id)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    break  # Unrecoverable
            
            yield time.time(), frame
            
            # Throttle only for local files; live streams inherently block on read()
            if not self.is_rtsp:
                elapsed = time.time() - start_time
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def close(self) -> None:
        if self._cap:
            self._cap.release()
