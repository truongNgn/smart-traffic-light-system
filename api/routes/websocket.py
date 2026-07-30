import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.dependencies import broadcaster

logger = structlog.get_logger("websocket_route")
router = APIRouter()

@router.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await broadcaster.connect(websocket)
    try:
        while True:
            # Keep connection open and handle client messages if any
            data = await websocket.receive_text()
            logger.debug("Received message from client", data=data)
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)
