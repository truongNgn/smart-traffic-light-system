import asyncio
import time
from common.constants import Direction
from common.schemas.control import PhaseAction
import redis.asyncio as redis

async def run_mock():
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    
    # Send East Green
    action = PhaseAction(target_direction=Direction.EAST, timestamp_s=time.time())
    await r.xadd("agent_commands", {"data": action.model_dump_json()})
    print(f"Sent: {action}")
    
    await asyncio.sleep(5)
    
    # Send North Green (Should trigger Yellow -> All Red -> Green)
    action = PhaseAction(target_direction=Direction.NORTH, timestamp_s=time.time())
    await r.xadd("agent_commands", {"data": action.model_dump_json()})
    print(f"Sent: {action}")

    await asyncio.sleep(15)
    # The watchdog in the service should trigger ALL_RED after 10s of silence

    await r.aclose()

if __name__ == "__main__":
    asyncio.run(run_mock())
