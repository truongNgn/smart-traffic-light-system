import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from api.routes import health, websocket
from api.dependencies import broadcaster, redis_consumer
from common.config import api_settings

logger = structlog.get_logger("api_main")

async def consume_events():
    """Background task to read from Redis and broadcast."""
    try:
        await redis_consumer.connect()
        async for event in redis_consumer.listen(last_id="$"):
            await broadcaster.broadcast(event)
    except Exception as e:
        logger.error("Error in consume_events background task", error=str(e))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI application")
    broadcaster.start()
    consumer_task = asyncio.create_task(consume_events())
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application")
    await broadcaster.stop()
    consumer_task.cancel()
    await redis_consumer.close()

app = FastAPI(title="Smart Traffic System API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(websocket.router, tags=["WebSocket"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=api_settings.host, port=api_settings.port, reload=True)
