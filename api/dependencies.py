from streaming.pubsub.broadcaster import Broadcaster
from streaming.bus.redis_consumer import RedisConsumer
from common.config import redis_settings

# Global singletons for FastAPI app
broadcaster = Broadcaster()
redis_consumer = RedisConsumer(
    host=redis_settings.host,
    port=redis_settings.port,
    stream_names=[redis_settings.stream_name, "traffic.phase_state.v1", "traffic.reasoning.v1"],
)
