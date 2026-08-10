"""Factory for message bus implementations."""

from streaming.bus.interface import MessageBus
from streaming.bus.redis_client import RedisMessageBus

def get_message_bus() -> MessageBus:
    """Returns the configured MessageBus implementation.
    Currently hardcoded to Redis for Phase 0.
    """
    return RedisMessageBus()
