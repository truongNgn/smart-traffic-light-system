"""Production vision pipeline. Processes video streams and emits events to Redis."""

import argparse
import threading
import time
from dataclasses import dataclass

import cv2
import structlog

from common.config import redis_settings, vision_settings
from common.schemas.vision import VehicleCountEvent
from streaming.bus.redis_client import RedisStreamBus
from vision.detection.yolo import YoloDetector
from vision.ingestion.simulator import VideoStreamSimulator
from vision.tracking.roi import ZoneCounter
from vision.tracking.speed import SpeedEstimator

logger = structlog.get_logger("vision_producer")


@dataclass(frozen=True)
class CameraStreamConfig:
    camera_id: str
    video: str


def parse_streams(value: str | None) -> list[CameraStreamConfig]:
    if not value:
        return []

    streams: list[CameraStreamConfig] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue

        if "=" not in item:
            raise ValueError(
                "Invalid stream entry. Expected CAMERA_ID=URL, "
                f"got {item!r}."
            )

        camera_id, video = item.split("=", 1)
        camera_id = camera_id.strip()
        video = video.strip()
        if not camera_id or not video:
            raise ValueError(
                "Invalid stream entry. CAMERA_ID and URL must both be non-empty, "
                f"got {item!r}."
            )
        streams.append(CameraStreamConfig(camera_id=camera_id, video=video))

    return streams


def connect_bus() -> RedisStreamBus:
    while True:
        try:
            return RedisStreamBus(
                host=redis_settings.host,
                port=redis_settings.port,
            )
        except Exception as e:
            logger.warning("Waiting for Redis...", error=str(e))
            time.sleep(2.0)


def run_camera_stream(
    *,
    stream: CameraStreamConfig,
    model_path: str,
    headless: bool,
) -> None:
    logger.info("Starting camera stream", camera_id=stream.camera_id, video=stream.video)

    bus = connect_bus()

    simulator = VideoStreamSimulator(
        video_path=stream.video,
        camera_id=stream.camera_id,
        target_fps=vision_settings.camera_fps,
        loop=True,
    )
    detector = YoloDetector(
        model_path=model_path,
        conf_threshold=vision_settings.conf_threshold,
        iou_threshold=vision_settings.iou_threshold,
    )

    zone_counter = ZoneCounter(rois=vision_settings.rois)
    speed_estimator = SpeedEstimator()

    try:
        for current_time, frame in simulator.stream():
            # 1. Detection and Tracking
            annotated_frame, class_counts, centroids = detector.process_frame(frame)

            # 2. Advanced Tracking Metrics
            lane_counts = zone_counter.update(centroids, frame_shape=frame.shape)
            mean_speed_mps = speed_estimator.update(centroids, current_time)

            # 3. Create Event
            event = VehicleCountEvent(
                camera_id=stream.camera_id,
                class_counts=class_counts,
                lane_counts=lane_counts,
                mean_speed_mps=mean_speed_mps,
                centroids=centroids,
            )

            # 4. Publish to Bus
            bus.publish(redis_settings.stream_name, event)

            # 5. Visualization (optional)
            if not headless:
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

                # Draw lane counts
                y_offset = 70
                for lane_name, count in lane_counts.items():
                    cv2.putText(
                        annotated_frame,
                        f"{lane_name}: {count} vehicles",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    y_offset += 30

                cv2.imshow(f"Feed {stream.camera_id}", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit requested by user.")
                    break

    except KeyboardInterrupt:
        logger.info("Camera stream shutting down...", camera_id=stream.camera_id)
    finally:
        simulator.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Stream Producer")
    parser.add_argument(
        "--video", type=str, default=vision_settings.stream_url or "test.mp4", help="Path to input video (RTSP or local)"
    )
    parser.add_argument(
        "--camera-id", type=str, default=vision_settings.camera_id, help="Camera ID (e.g., N, S, E, W)"
    )
    parser.add_argument(
        "--model", type=str, default=vision_settings.yolo_model_path, help="YOLO model path"
    )
    parser.add_argument(
        "--streams",
        type=str,
        default=vision_settings.streams,
        help="Comma-separated CAMERA_ID=URL map for multi-camera mode",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without cv2.imshow GUI"
    )
    args = parser.parse_args()

    streams = parse_streams(args.streams)
    if not streams:
        streams = [CameraStreamConfig(camera_id=args.camera_id, video=args.video)]

    try:
        if len(streams) == 1:
            run_camera_stream(stream=streams[0], model_path=args.model, headless=args.headless)
            return

        logger.info("Starting multi-camera producer", camera_count=len(streams))
        workers = [
            threading.Thread(
                target=run_camera_stream,
                kwargs={
                    "stream": stream,
                    "model_path": args.model,
                    "headless": args.headless,
                },
                name=f"vision-{stream.camera_id}",
                daemon=False,
            )
            for stream in streams
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        logger.info("Producer shutting down...")
    finally:
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
