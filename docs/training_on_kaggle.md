# Training on Kaggle

The local dev machine has no GPU, so real training runs (hundreds of
episodes) belong on a Kaggle notebook with a free T4/P100 GPU instead.
`rl/agent/dqn_agent.py` already auto-selects CUDA when available - no code
changes needed between local smoke-testing and a real Kaggle run.

## 1. Notebook settings

- Accelerator: **GPU T4 x2** (or any available GPU)
- Internet: **On** (needed to `apt-get install sumo` and `git clone`)

## 2. Setup cell

```bash
!apt-get update -qq && apt-get install -y -qq sumo sumo-tools sumo-doc
!git clone https://github.com/truongNgn/smart-traffic-light-system.git
%cd smart-traffic-light-system
!pip install -q pydantic pydantic-settings structlog traci sumolib gymnasium numpy
# torch is already preinstalled on Kaggle's GPU image - don't reinstall it
```

```python
import os
os.environ["SUMO_HOME"] = "/usr/share/sumo"
```

Verify SUMO is visible before doing anything else:

```bash
!sumo --version
```

## 3. Build the network and generate demand

```bash
!python -m simulation.net.build_net
!python -m simulation.net.generate_routes --duration 3600 --seed 42
```

## 4. Train

```bash
!python -m rl.train.train \
    --episodes 500 \
    --episode-duration 3600 \
    --checkpoint-dir /kaggle/working/checkpoints
```

Or from a notebook cell for more control (custom epsilon schedule, etc.):

```python
from rl.train.config import TrainingConfig
from rl.train.train import train

cfg = TrainingConfig(
    num_episodes=500,
    episode_duration_s=3600,
    checkpoint_dir="/kaggle/working/checkpoints",
    batch_size=128,          # GPU can afford a bigger batch than the CPU default
)
agent = train(cfg)
```

Progress is logged as structured JSON (one line per episode: reward,
epsilon, avg_loss, final_waiting_time_s) - watch it in the cell output to
confirm the reward trend is moving in the right direction before committing
to a long run.

## 5. Resuming across Kaggle's session time limit

Kaggle sessions get killed after ~9-12 hours. Checkpoint files
(`dqn_episode_N.pt`, `dqn_best.pt`, `dqn_final.pt`) already land in
`/kaggle/working/`, which Kaggle preserves as notebook output between
sessions. To continue:

```bash
!python -m rl.train.train --episodes 1000 --resume /kaggle/working/checkpoints/dqn_final.pt
```

`--resume` restores the policy net, target net, and optimizer state, and
picks up numbering from the checkpoint's saved episode count.

## 6. Bringing the model back

Download `dqn_best.pt` (or `dqn_final.pt`) from the notebook's Output tab
and drop it into `checkpoints/` in the local repo - that directory is
gitignored, so it won't get committed, and Stage 5's benchmark/evaluation
scripts will pick it up from there.

```python
from rl.agent.dqn_agent import DQNAgent
from rl.train.checkpoint import load_checkpoint

agent = DQNAgent()
load_checkpoint("checkpoints/dqn_best.pt", agent)
```
