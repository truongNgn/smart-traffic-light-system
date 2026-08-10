"""Smoke test for Engineer A Stage 1 (Vision + Streaming)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common.config import redis_settings, vision_settings
from common.logging import configure_logging, get_logger
from common.schemas.vision import VehicleCountEvent
from streaming.bus.redis_client import RedisStreamBus
from vision.detection.yolo import YoloDetector
from vision.ingestion.simulator import VideoStreamSimulator
from vision.tracking.roi import ZoneCounter

logger = get_logger(component="smoke_test")


def create_dummy_video(path: str, fps: int = 30, duration_s: int = 5) -> None:
    """Download a real traffic sample video so YOLO actually detects cars."""
    import urllib.request
    
    url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
    logger.info("Downloading real sample traffic video...", url=url, path=path)
    try:
        urllib.request.urlretrieve(url, path)
        logger.info("Download complete.")
    except Exception as e:
        logger.error("Failed to download sample video.", error=str(e))
        # Fallback to the old dummy video if download fails
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for i in range(fps * duration_s):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = (i * 5) % width
            y = height // 2
            cv2.rectangle(frame, (x, y - 20), (x + 80, y + 20), (255, 255, 255), -1)
            out.write(frame)
        out.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for Vision Pipeline")
    parser.add_argument("--model", type=str, default=None, help="Path to the YOLO model (e.g., yolov8n.engine)")
    parser.add_argument("--full", action="store_true", help="Run inference on the entire video without looping or 5s limit")
    args = parser.parse_args()

    configure_logging()
    logger.info("Starting Vision Stage 1 Smoke Test")
    
    # 1. Initialize components
    model_path = args.model if args.model else vision_settings.yolo_model_path
    loop_video = not args.full
    
    if vision_settings.stream_url:
        logger.info("Using RTSP stream from config", stream_url=vision_settings.stream_url)
        video_source = vision_settings.stream_url
    else:
        test_video = "test.mp4"
        if not Path(test_video).exists():
            create_dummy_video(test_video)
        video_source = test_video

    simulator = VideoStreamSimulator(
        video_path=video_source, 
        camera_id=vision_settings.camera_id, 
        target_fps=vision_settings.camera_fps, 
        loop=loop_video
    )
    detector = YoloDetector(
        model_path=model_path,
        conf_threshold=vision_settings.conf_threshold,
        iou_threshold=vision_settings.iou_threshold,
    )
    zone_counter = ZoneCounter(rois=vision_settings.rois)
    
    # We allow the bus connection to fail gracefully if Redis is not running locally outside Docker
    bus = None
    try:
        bus = RedisStreamBus(host=redis_settings.host, port=redis_settings.port)
    except Exception as e:
        logger.warning("Redis is not available. Continuing without publishing.", error=str(e))
        
    # Output video for visual verification
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    out_video = cv2.VideoWriter(f"smoke_test_output_{vision_settings.camera_id}.mp4", fourcc, vision_settings.camera_fps, (640, 480))

    try:
        frame_count = 0
        for timestamp_s, frame in simulator.stream():
            # 2. Process frame
            annotated_frame, class_counts, centroids = detector.process_frame(frame)
            
            # 2.5 ROI Tracking & Drawing
            lane_counts = zone_counter.update(centroids, frame_shape=frame.shape)
            annotated_frame = zone_counter.draw_zones(annotated_frame)
            
            y_offset = 30
            for lane_name, count in lane_counts.items():
                cv2.putText(
                    annotated_frame,
                    f"{lane_name}: {count} vehicles",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),  # Màu vàng
                    2,
                )
                y_offset += 30
            
            # 3. Create Event
            event = VehicleCountEvent(
                camera_id=simulator.camera_id,
                class_counts=class_counts,
                lane_counts=lane_counts,
                centroids=centroids,
            )
            
            # 4. Publish
            if bus:
                bus.publish(redis_settings.stream_name, event)
            else:
                logger.info("Mock Publish", event_data=event.model_dump())
                
            # 5. Save output frame
            # Ensure frame is 640x480 for the video writer, or resize it
            resized = cv2.resize(annotated_frame, (640, 480))
            out_video.write(resized)
            
            frame_count += 1
            if not args.full and frame_count >= vision_settings.camera_fps * 5:  # Run for 5 seconds by default
                logger.info("Smoke test completed 5 seconds of video processing.")
                break
                
    finally:
        simulator.close()
        out_video.release()
        logger.info("Smoke test finished.")


if __name__ == "__main__":
    main()
