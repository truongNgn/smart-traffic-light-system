import threading
import json
import time
import websocket
from collections import deque
import structlog
from typing import Dict, Any, List

logger = structlog.get_logger("dashboard_state")

# Global State Stores
MAX_HISTORY = 60  # Store last 60 events per lane

# vehicle_counts[direction] = deque([{"time": ts, "count": int}, ...])
vehicle_counts: Dict[str, deque] = {
    "NORTH": deque(maxlen=MAX_HISTORY),
    "SOUTH": deque(maxlen=MAX_HISTORY),
    "EAST": deque(maxlen=MAX_HISTORY),
    "WEST": deque(maxlen=MAX_HISTORY),
}

latest_phase_state: Dict[str, Any] = {
    "active_direction": None,
    "is_yellow": False,
    "is_all_red": True,
    "timestamp_s": 0.0
}

latest_reasoning: List[Dict[str, Any]] = []

def on_message(ws, message):
    try:
        envelope = json.loads(message)
        topic = envelope.get("topic")
        payload = envelope.get("payload", {})

        if topic == "vehicle_counts":
            ts = payload.get("timestamp_s", time.time())
            counts = payload.get("counts", {})
            for direction, count in counts.items():
                if direction in vehicle_counts:
                    vehicle_counts[direction].append({"time": ts, "count": count})
                    
        elif topic == "phase_states":
            global latest_phase_state
            # Map enum index back to string if needed, or rely on active_direction name
            # The payload will have active_direction as integer if it's an enum, 
            # wait, Pydantic serializes Enum as integer if not configured, let's check it.
            latest_phase_state = payload
            
        elif topic == "reasoning_logs":
            latest_reasoning.insert(0, payload)
            if len(latest_reasoning) > 20:
                latest_reasoning.pop()
                
    except Exception as e:
        logger.error("Error parsing websocket message", error=str(e))

def on_error(ws, error):
    pass  # Suppress error logs to keep terminal clean

def on_close(ws, close_status_code, close_msg):
    logger.warning("WebSocket closed. Reconnecting...")

def on_open(ws):
    logger.info("WebSocket connected to telemetry stream.")

def run_websocket():
    # Run forever with automatic reconnect
    while True:
        ws = websocket.WebSocketApp(
            "ws://localhost:8000/ws/telemetry",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever()
        time.sleep(2)

# Singleton thread starter
_thread_started = False
def start_background_thread():
    global _thread_started
    if not _thread_started:
        t = threading.Thread(target=run_websocket, daemon=True)
        t.start()
        _thread_started = True
