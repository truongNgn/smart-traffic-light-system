# Hướng dẫn Khởi chạy Hệ Thống Smart Traffic Light

Hệ thống của chúng ta được thiết kế theo kiến trúc Microservices (các dịch vụ hoạt động độc lập) và giao tiếp với nhau thông qua Redis Streams. 
Dưới đây là hướng dẫn chi tiết cách khởi động toàn bộ hệ thống để chiêm ngưỡng toàn bộ luồng dữ liệu (từ bộ xử lý đến Web Dashboard).

## Yêu cầu Hệ thống
- Đã cài đặt **Python 3.11+** và **uv**.
- Đã chạy lệnh cài đặt môi trường: `uv sync`
- Đã cài đặt **Docker** (để khởi chạy máy chủ Redis).

---

## Bước 0: Khởi động Trạm Trung Chuyển (Redis)
Tất cả các dịch vụ đều phải gửi/nhận dữ liệu qua Redis. Hãy chắc chắn Redis đang chạy ngầm bằng lệnh sau:
```bash
docker compose up -d redis
```

---

Để hệ thống hoạt động hoàn chỉnh, bạn cần mở **4 cửa sổ Terminal riêng biệt** (Powershell hoặc CMD). Đảm bảo tất cả các Terminal đều đang đứng ở thư mục gốc của dự án (`smart-traffic-light-system`).

### Terminal 1: Khởi động Cổng API (API Gateway)
Trạm API này có nhiệm vụ móc nối vào Redis, nhặt dữ liệu và "bơm" thẳng ra cổng WebSocket (Pub/Sub) cho trang Web Dashboard sử dụng.
```bash
uv run python -m uvicorn api.main:app --reload
```

### Terminal 2: Khởi động Trạm Kiểm Soát An Toàn (Control Service)
Bộ phận đầu não đảm bảo an toàn tuyệt đối cho ngã tư. Nó lắng nghe quyết định từ AI, áp dụng luật an toàn (bắt buộc chèn đèn Vàng và Đỏ 2 giây) và cung cấp tính năng **Watchdog** (tự động khóa ngã tư về chế độ Đỏ toàn bộ nếu AI mất tín hiệu quá 10 giây).
```bash
uv run python -m control.service
```

### Terminal 3: Khởi động Web Dashboard (Streamlit)
Bảng điều khiển trực quan hóa dữ liệu theo thời gian thực (Real-time). Lệnh này sẽ tự động mở tab mới trong trình duyệt của bạn (hoặc bạn có thể truy cập `http://localhost:8501`).
*(Lưu ý: Ở lần chạy đầu tiên, nếu màn hình Terminal có hỏi Email, bạn chỉ cần nhấn `Enter` để bỏ qua).*
```bash
uv run streamlit run dashboard/app.py
```

### Terminal 4: Khởi động AI hoặc Tác nhân giả lập (Mock Agent)
Cuối cùng, chạy file kịch bản đóng vai trò là một con AI. Nó sẽ bắn lệnh xin đổi đèn giao thông (East -> chờ 5s -> North) để bạn quan sát sự thay đổi màu sắc và hệ thống phòng vệ hoạt động trực tiếp trên Web Dashboard!
```bash
uv run python -m tests.mock_agent
```

> **Lưu ý:** Hiện tại chúng ta dùng `mock_agent.py` để mô phỏng. Ở các Stage sau, lệnh này sẽ được thay thế bằng lệnh khởi chạy mô hình Trí tuệ nhân tạo (Reinforcement Learning) thực thụ.
