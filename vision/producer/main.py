"""Production vision pipeline. Processes video streams and emits events to Redis."""

import argparse
import time

import cv2
import structlog

from common.config import redis_settings, vision_settings
from common.schemas.vision import VehicleCountEvent
from streaming.bus.redis_client import RedisBus
from vision.detection.yolo import YoloDetector
from vision.ingestion.simulator import VideoSimulator
from vision.tracking.roi import ZoneCounter
from vision.tracking.speed import SpeedEstimator

logger = structlog.get_logger("vision_producer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Stream Producer")
    parser.add_argument(
        "--video", type=str, default="test.mp4", help="Path to input video"
    )
    parser.add_argument(
        "--camera-id", type=str, default="N", help="Camera ID (e.g., N, S, E, W)"
    )
    parser.add_argument(
        "--model", type=str, default=vision_settings.yolo_model_path, help="YOLO model path"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without cv2.imshow GUI"
    )
    args = parser.parse_args()

    logger.info("Starting Vision Stream Producer", camera_id=args.camera_id, video=args.video)

    # Initialize components
    bus = RedisBus(
        host=redis_settings.host,
        port=redis_settings.port,
        stream_name=redis_settings.stream_name,
    )
    
    # Wait for Redis connection (simple retry loop for compose)
    while True:
        try:
            bus.connect()
            break
        except Exception as e:
            logger.warning("Waiting for Redis...", error=str(e))
            time.sleep(2.0)

    simulator = VideoSimulator(args.video, target_fps=vision_settings.camera_fps, loop=True)
    detector = YoloDetector(
        model_path=args.model,
        conf_threshold=vision_settings.conf_threshold,
        iou_threshold=vision_settings.iou_threshold,
    )
    
    zone_counter = ZoneCounter(rois=vision_settings.rois)
    speed_estimator = SpeedEstimator()

    try:
        for frame in simulator.stream():
            current_time = time.time()
            
            # 1. Detection and Tracking
            annotated_frame, class_counts, centroids = detector.process_frame(frame)
            
            # 2. Advanced Tracking Metrics
            lane_counts = zone_counter.update(centroids)
            mean_speed_mps = speed_estimator.update(centroids, current_time)

            # 3. Create Event
            event = VehicleCountEvent(
                camera_id=args.camera_id,
                timestamp_s=current_time,
                class_counts=class_counts,
                lane_counts=lane_counts,
                mean_speed_mps=mean_speed_mps,
                centroids=centroids,
            )

            # 4. Publish to Bus
            bus.publish_vision_event(event)
            
            # 5. Visualization (optional)
            if not args.headless:
                # Draw ROIs
                annotated_frame = zone_counter.draw_zones(annotated_frame)
                
                # Draw speed overlay
                cv2.putText(
                    annotated_frame,
                    f"Speed: {mean_speed_mps:.1f} m/s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )
                
                cv2.imshow(f"Feed {args.camera_id}", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit requested by user.")
                    break

    except KeyboardInterrupt:
        logger.info("Producer shutting down...")
    finally:
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
