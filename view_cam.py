import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import cv2
import numpy as np
import threading
import time

class RTSPStream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret = ret
                    self.frame = frame

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

urls = [
    "rtsp://localhost:8554/cam1",
    "rtsp://localhost:8554/cam2",
    "rtsp://localhost:8554/cam3",
    "rtsp://localhost:8554/cam4"
]

print("Đang khởi tạo kết nối (Đa luồng/Multi-threading)...")
streams = [RTSPStream(url) for url in urls]

# Chờ 1 chút để các camera bắt kịp frame đầu tiên
time.sleep(2)
print("✅ Cửa sổ đang mở! (Nhấn phím 'q' trên bàn phím để tắt)")

while True:
    frames = []
    for i, stream in enumerate(streams):
        ret, frame = stream.read()
        if ret and frame is not None:
            frame = cv2.resize(frame, (640, 360))
            cv2.putText(frame, f"Cam {i+1}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Cam {i+1} Loading...", (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        frames.append(frame)
    
    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], frames[3]))
    grid = np.vstack((top_row, bottom_row))
    
    cv2.imshow("Smart Traffic System - Live CCTV", grid)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

for stream in streams:
    stream.release()
cv2.destroyAllWindows()
