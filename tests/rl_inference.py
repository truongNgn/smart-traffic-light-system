"""Real-time RL Inference Agent that bridges SUMO simulation with the Microservices architecture.

This script replaces the mock_agent by loading the trained DQN model. It:
1. Runs SUMO to generate the traffic state.
2. Uses the DQN model to predict the optimal green phase.
3. Publishes the reasoning and phase command to Redis for the Dashboard and Control Service.
4. Listens to the Control Service's phase_states stream to apply the safe (yellow/red-buffered)
   phase transitions back to the SUMO simulation.
"""

import asyncio
import time
import json
from pathlib import Path

import numpy as np
import torch
import redis.asyncio as redis
import structlog

from common.constants import PhaseAction as CPhaseAction, Direction
from common.schemas.control import PhaseAction, PhaseState, ReasoningLog
from rl.agent.dqn_agent import DQNAgent
from rl.train.checkpoint import load_checkpoint
from simulation.traci_wrapper.session import TraciSession
from simulation.state.grid_encoder import GridEncoder
from simulation.traci_wrapper.tls_controller import TlsController

logger = structlog.get_logger("rl_inference")

# Map DQN output (0, 1) to a primary target Direction (used by control service)
ACTION_TO_DIRECTION = {
    int(CPhaseAction.EAST_WEST): Direction.EAST,
    int(CPhaseAction.NORTH_SOUTH): Direction.NORTH
}

DIRECTION_TO_ACTION = {
    Direction.EAST: CPhaseAction.EAST_WEST,
    Direction.WEST: CPhaseAction.EAST_WEST,
    Direction.NORTH: CPhaseAction.NORTH_SOUTH,
    Direction.SOUTH: CPhaseAction.NORTH_SOUTH,
}

async def run_inference():
    # 1. Setup Redis
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    logger.info("Connected to Redis")
    
    # 2. Setup Agent
    agent = DQNAgent()
    checkpoint_path = Path("checkpoints/dqn_eval_best.pt")
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return
        
    load_checkpoint(checkpoint_path, agent)
    agent.policy_net.eval()
    logger.info("Loaded agent from checkpoint", path=str(checkpoint_path))

    # 3. Setup SUMO
    sumocfg = Path("simulation/net/intersection.sumocfg")
    sim = TraciSession(
        sumocfg_path=sumocfg,
        use_gui=False, # Chạy ngầm (headless) vì Dashboard đã hiển thị giao diện rồi
        step_length_s=1.0,
        backend="traci"
    ).start()
    logger.info("Started SUMO simulation")
    
    encoder = GridEncoder()
    tls = TlsController(sim.traci, "C")
    
    last_id = "$"
    current_active_direction = None
    
    try:
        while True:
            # Step the simulation (this simulates time passing and cars moving)
            sim.step(1)
            
            # Read state from SUMO
            state = encoder.encode(sim.traci)
            obs = np.asarray(state.grid, dtype=np.float32)
            
            # Agent decides action
            action_idx = agent.act(obs, epsilon=0.0) # greedy
            
            # Extract Q-values for logging
            state_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.policy_net(state_t).squeeze(0).tolist()
                
            q_dict = {
                "EAST_WEST": float(q_values[0]),
                "NORTH_SOUTH": float(q_values[1])
            }
            target_direction = ACTION_TO_DIRECTION[action_idx]
            
            reasoning = ReasoningLog(
                timestamp_s=time.time(),
                q_values=q_dict,
                chosen_action=target_direction,
                exploration=False
            )
            
            # Publish Reasoning Log for the dashboard
            await r.xadd("agent_reasoning", {"data": reasoning.model_dump_json()})
            
            # Publish PhaseAction for the Control Service
            # Control Service ignores redundant "GREEN" commands for the same direction.
            phase_action = PhaseAction(target_direction=target_direction, timestamp_s=time.time())
            await r.xadd("agent_commands", {"data": phase_action.model_dump_json()})
            
            # Check for new phase states from Control Service
            streams = await r.xread({"phase_states": last_id}, count=10, block=1)
            if streams:
                for stream_name, messages in streams:
                    for message_id, data in messages:
                        last_id = message_id
                        if "data" in data:
                            payload = json.loads(data["data"])
                            phase_state = PhaseState(**payload)
                            
                            # Apply the phase state to SUMO
                            if phase_state.is_all_red:
                                sim.traci.trafficlight.setRedYellowGreenState("C", tls.all_red_state())
                            elif phase_state.is_yellow and phase_state.active_direction is not None:
                                phase = DIRECTION_TO_ACTION[phase_state.active_direction]
                                sim.traci.trafficlight.setRedYellowGreenState("C", tls.yellow_state(phase))
                            elif phase_state.active_direction is not None:
                                phase = DIRECTION_TO_ACTION[phase_state.active_direction]
                                sim.traci.trafficlight.setRedYellowGreenState("C", tls.green_state(phase))
                            
                            current_active_direction = phase_state.active_direction

            # Sleep a bit to sync roughly with real-time (since step_length_s=1.0)
            await asyncio.sleep(1.0)
            
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("Error in inference loop", error=str(e))
    finally:
        sim.close()
        await r.aclose()
        logger.info("Shutdown complete")

if __name__ == "__main__":
    asyncio.run(run_inference())
