# Smart Traffic Light DQN Training Brain

This document is the working memory for the SUMO smart traffic light training effort. It records what has been tried, what failed, what is currently safest, and what the next training direction should be.

## Current Objective

Build a traffic-police-style controller for a Vietnamese four-way intersection:

- Action `0`: `EAST_WEST` green, both East and West approaches move together.
- Action `1`: `NORTH_SOUTH` green, both North and South approaches move together.
- The controller should improve over fixed-time in normal traffic, heavy balanced traffic, EW peak, NS peak, and mixed peak traffic.
- The controller must not starve one phase while optimizing the other.

## Current Best Safe Checkpoint

Use:

```text
checkpoints/dqn_eval_best.pt
```

The file currently copied as:

```text
checkpoints/dqn_ew_repair_best.pt
```

has the same SHA256 hash as `dqn_eval_best.pt`, so the hard guardrail selection kept the safe base model instead of promoting a risky repair checkpoint.

Current safe checkpoint metadata:

- episode: `2200`
- output size: `2`
- action space: `EAST_WEST`, `NORTH_SOUTH`

## Latest Benchmark

Checkpoint tested:

```text
checkpoints/dqn_ew_repair_best.pt
```

Because this file is hash-identical to `dqn_eval_best.pt`, these are effectively the current safe base results.

| Scenario | Mean Wait Improvement | Final Wait Improvement | Queue Improvement | Throughput Improvement | Verdict |
|---|---:|---:|---:|---:|---|
| Normal, 1200s, 10 seeds | `+23.3%` | `0.0%` | `+27.0%` | `0.0%` | Good |
| Heavy 2x, 1800s, 3 seeds | `+19.9%` | `+50.6%` | `+0.6%` | `-5.5%` | Acceptable but throughput loss |
| EW 3x, 1800s, 3 seeds | `-16.8%` | `+52.3%` | `-28.3%` | `+11.7%` | Main unresolved weakness |
| NS 3x, 1800s, 3 seeds | `+31.6%` | `-293.7%` | `-4.8%` | `+11.4%` | Good mean wait and throughput, final wait tradeoff |
| Mixed peak, 2400s, 3 seeds | `+90.9%` | `+98.4%` | `+4.2%` | `+5.2%` | Very strong |

Overall grade for the safe base: `B+`, about `78/100`.

## What We Learned

### 1. Four-direction action space was not appropriate

Earlier models using single-direction actions did not match real intersection behavior. A real green phase normally lets opposite directions on the same road move together. The project now uses two phase actions:

```text
EAST_WEST
NORTH_SOUTH
```

This is the correct action abstraction for the current SUMO intersection.

### 2. Simple waiting-time reward is not enough

The original reward was:

```text
previous_total_wait - current_total_wait
```

This improves total waiting, but it can hide starvation. A model can make total waiting look better by over-serving the dominant road while letting the other phase wait too long.

Balanced reward variants added penalties for:

- total waiting level
- max phase waiting
- phase wait imbalance
- total queue
- max phase queue
- max vehicle waiting

This improved normal, heavy, NS, and mixed cases, but EW3x remains difficult.

### 3. Sequential fine-tuning is unstable

Sequential training like:

```text
normal -> heavy -> EW -> NS -> mixed
```

or repair training like:

```text
base -> EW repair -> mixed refresh
```

can overfit to the most recent scenario. The episode `2675` repair checkpoint proved this:

- EW3x improved from `-16.8%` to `+9.9%`
- Heavy improved from `+19.9%` to `+30.0%`
- But NS3x catastrophically failed:
  - mean wait improvement: `-8228.4%`
  - throughput: `-16.6%`

That checkpoint must not be used.

### 4. Hard guardrails are necessary

Average metrics are not enough. A checkpoint can look acceptable on aggregate while one seed completely starves a phase.

Hard guardrail selection should reject a checkpoint when:

- any NS3x seed is worse than fixed-time on mean waiting
- any scenario has a mean-wait spike
- any scenario has a final-wait spike
- heavy throughput drops too much
- mixed traffic regresses too much

If no repair checkpoint passes, keep the base checkpoint.

This behavior is intentional and correct.

## Current Notebook Strategy

Main notebook for the next attempt:

```text
notebooks/smart-traffic-light-ew-repair-finetune.ipynb
```

It now uses interleaved fine-tuning:

1. Load `dqn_eval_best.pt` from Hugging Face:

```text
johnnynnt/smart-traffic-light-dqn-sumo
```

2. Build a scenario bank:

```text
normal
heavy_1_5x
heavy_2x
heavy_2_5x
imbalanced_ew2x
imbalanced_ew3x
imbalanced_ew3_5x
imbalanced_ns2x
imbalanced_ns3x
imbalanced_ns3_5x
mixed_balanced_peak
mixed_extreme_peak
```

3. Train with scenario sampling, not sequential stages:

```text
Normal: 20%
Heavy: 25%
EW: 20%
NS: 20%
Mixed: 15%
```

4. Use one shared replay buffer across all sampled scenarios.

5. Evaluate recent checkpoints with hard guardrails.

6. Promote a checkpoint to:

```text
dqn_ew_repair_best.pt
```

only if it passes guardrails. Otherwise, copy the safe base checkpoint.

## Why Interleaved Training

Interleaved training is preferred because the policy sees a mixed traffic distribution throughout training. This reduces catastrophic forgetting:

```text
episode 1: normal
episode 2: EW3x
episode 3: heavy2x
episode 4: NS3x
episode 5: mixed
...
```

This is closer to real deployment, where traffic does not arrive in neat training stages.

## Remaining Problem

EW3x is the main unresolved weakness:

```text
mean wait: -16.8%
queue: -28.3%
throughput: +11.7%
```

Interpretation:

The model is moving more vehicles through EW peak, but it creates worse average wait and queue. It is acting too throughput-oriented in that scenario.

## Why Fine-Tuning Has Not Improved EW Yet

The latest interleaved fine-tuning run did not produce a new promoted model. The downloaded file:

```text
checkpoints/dqn_ew_repair_best.pt
```

had the same SHA256 hash as:

```text
checkpoints/dqn_eval_best.pt
```

and both reported:

```text
episode = 2200
```

Interpretation:

```text
fine-tuning likely ran
-> candidate checkpoints were created
-> evaluation rejected all unsafe repair candidates
-> notebook copied dqn_eval_best.pt as dqn_ew_repair_best.pt
```

This is not a notebook failure. It means the hard guardrail worked: no fine-tuned checkpoint improved EW while also preserving normal, heavy, NS, and mixed safety.

### Technical Causes

#### 1. EW and NS compete directly

With the current two-action setup, the model can only choose:

```text
EAST_WEST
NORTH_SOUTH
```

Improving EW often means allocating more green time to EW. But this can directly reduce service for NS. The failed episode `2675` repair checkpoint demonstrated this clearly:

- EW3x improved to `+9.9%`
- Heavy improved to `+30.0%`
- NS3x collapsed to `-8228.4%`

So the system is currently stuck between:

```text
keep base safe -> EW remains weak
push EW harder -> NS can starve
```

#### 2. The action space is coarse

The current action space does not include duration choices. Each action means selecting one of two fixed-duration phase decisions. The agent cannot express decisions such as:

```text
keep EW green slightly longer
switch early because NS wait is rising
serve NS briefly then return to EW
```

Those behaviors are handled indirectly by repeating actions across control steps, which makes learning harder.

#### 3. The state is missing important traffic-officer features

The current neural network mainly receives grid occupancy. It does not directly receive the most important fairness/safety signals a human traffic officer would watch:

- current active phase
- red time by phase
- queue length by phase
- waiting time by phase
- max vehicle wait by phase
- time since last switch
- whether the last action was forced by the guard

The environment and guardrail can compute some of these values, but the DQN policy network does not directly learn from them as observation features.

This makes EW vs NS balancing partly invisible to the model.

#### 4. Reward is a scalar approximation of a multi-objective problem

Traffic control has multiple objectives:

- reduce average waiting
- reduce final waiting
- reduce queue
- improve throughput
- avoid phase starvation
- avoid extreme per-vehicle wait

DQN receives one scalar reward. If the scalar reward gives too much weight to throughput, EW can move many vehicles but increase mean wait and queue. If it gives too much weight to fairness, EW repair can weaken.

This is why fine-tuning often behaves like pulling a short blanket:

```text
improve EW -> hurt NS
protect NS -> EW remains weak
```

#### 5. Fine-tuning may be trapped by the base policy

The base policy is already stable in normal, heavy, NS, and mixed traffic. Fine-tuning lightly may not move far enough to repair EW. Fine-tuning aggressively can move far enough, but may destroy NS behavior.

That is why the hard guardrail repeatedly falls back to the safe base.

## Implication

Continuing the same kind of fine-tune is unlikely to solve EW cleanly. The next meaningful improvement probably requires at least one of:

1. richer observation state,
2. runtime fairness/safety override,
3. training from scratch with interleaved scenario sampling,
4. Dueling DQN or another RL architecture,
5. adaptive phase duration rather than only fixed 10-second phase choices.

## Next Recommended Steps

### Step 1: Run the interleaved notebook again

Use the current interleaved notebook and inspect the evaluation cell. If `dqn_ew_repair_best.pt` is still episode `2200`, that means no checkpoint passed hard guardrails.

### Step 2: Save evaluation output from Kaggle

Before downloading the model, save or copy the evaluation printout. The important lines are:

```text
pass=True/False
failures:
mean_wait_imp
queue_imp
arrived_imp
worst_seed_wait
```

Without these logs, local testing can tell whether the final file is safe, but not why other checkpoints were rejected.

### Step 3: If interleaved fine-tune still fails, train from scratch

Fine-tuning may not escape the base model's learned bias. If the interleaved approach does not produce a passing repair checkpoint, create a new full training notebook:

```text
random DQN weights
-> interleaved scenario sampler from episode 1
-> hard guardrail selection
```

This should be called full training or training from scratch, not fine-tuning.

### Step 4: Consider Dueling DQN

The next architecture upgrade should be Dueling DQN, not a larger plain DQN. It can help separate:

- state value: how bad the intersection currently is
- action advantage: which phase is better now

This may help in balanced-vs-imbalanced decisions.

### Step 5: Add runtime safety override

The production controller should not rely on DQN alone. Add a runtime fairness layer:

```text
DQN proposes action
Safety layer checks red time, queue, and max wait
Safety layer overrides only if needed
```

This mirrors a human traffic officer: prioritize the busy road, but do not let the other road die.

Example policy:

```text
if other_phase_red_time > 90 and other_phase_queue > 4:
    force other phase

if other_phase_max_vehicle_wait > 180:
    force other phase

if other_phase_queue > current_phase_queue * 2 and other_phase_red_time > 60:
    force other phase
```

## Deployment Recommendation

Do not deploy a checkpoint only because it improves EW3x.

Deploy only if it passes all:

- normal remains better than fixed-time
- heavy remains better on mean wait
- EW3x is not worse on mean wait
- NS3x remains better on mean wait and throughput
- mixed remains very strong
- no seed-level starvation spike

Until then, keep using:

```text
dqn_eval_best.pt
```

as the safest demo/production candidate.
