---
title: Warehouse RL
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# warehouse.rl

# Warehouse Robot Dispatch — OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://huggingface.co/openenv)

An RL environment simulating **Autonomous Mobile Robot (AMR) navigation** in a
fulfillment center warehouse — the exact task performed by robots at Amazon
Robotics, Flipkart, and Myntra every day.

---

## Real-World Motivation

Modern fulfillment centers use AMRs to navigate warehouse floors, locate
inventory items at shelf locations, and pick them for order dispatch. Human
supervisors dispatch these robots and monitor their efficiency in real time.

This environment models that operational task:
- The **robot** navigates a warehouse floor plan (grid)
- **Inventory items** sit at shelf locations and must be picked
- **Blocked aisles** and **active worker zones** must be avoided
- **Shift time** is limited — the agent must plan efficient routes

---

## Environment Description

```
Warehouse floor plan (ASCII):
  R = Robot (agent-controlled AMR)
  I = Inventory item at shelf location (pick target)
  X = Blocked aisle / active worker zone (collision = penalty)
  . = Empty aisle (passable)

Example 5x5 warehouse:
  . . X . .
  . R . . I
  . . . X .
  . I . . .
  . . X . .
```

---

## Action Space

| Action | Direction | Description |
|--------|-----------|-------------|
| `move_up` | North | Robot moves one cell up |
| `move_down` | South | Robot moves one cell down |
| `move_left` | West | Robot moves one cell left |
| `move_right` | East | Robot moves one cell right |

Moving onto an inventory item (`I`) automatically picks it.
Moving into a blocked zone (`X`) or wall returns a collision penalty.

---

## Observation Space

Each step returns a JSON object with:

```json
{
  "warehouse": "ASCII grid string",
  "observation": [float32 array],
  "info": {
    "robot_pos": [row, col],
    "items_remaining": 2,
    "shift_time": 87,
    "picks": 0
  }
}
```

**Observation vector** (normalized to [0, 1]):
- Per inventory item: `[rel_row, rel_col, manhattan_dist, is_active]`
- Robot state: `[row%, col%, shift_time%, items_remaining%]`
- Proximity sensors: `[blocked_up, blocked_down, blocked_left, blocked_right]`

---

## Tasks

| Task ID | Grid | Items | Blocked | Max Steps | Difficulty |
|---------|------|-------|---------|-----------|------------|
| `single_pick` | 5×5 | 1 | 3 | 50 | Easy |
| `multi_pick` | 5×5 | 2 | 3 | 100 | Medium |
| `congested_floor` | 7×7 | 3 | 8 | 150 | Hard |

### Task 1: Single Item Pick (Easy)
Pick 1 inventory item in a small 5×5 warehouse with 3 blocked aisles.
Models a simple single-item order fulfillment dispatch.

### Task 2: Multi-Item Order Fulfillment (Medium)
Pick 2 inventory items in a standard 5×5 warehouse.
Requires planning an efficient route across the floor.

### Task 3: Peak-Hour Congested Warehouse (Hard)
Pick 3 inventory items in a 7×7 warehouse with 8 blocked zones.
Models peak-hour operations where many aisles are occupied by workers.

---

## Reward Function

| Event | Reward | Rationale |
|-------|--------|-----------|
| Each step taken | -0.1 | Efficiency cost — faster routes score higher |
| Collision with wall/blocked zone | -0.5 | Safety penalty |
| Inventory item picked | +10.0 | Core task completion |
| All items picked (completion) | +25.0 | Full shift bonus |
| Shift time exhausted | -5.0 | Incomplete order penalty |

**Score formula** (normalized to [0.0, 1.0]):
```
score = clamp((total_reward + 5) / (max_reward + 5), 0.0, 1.0)
max_reward = num_items × 10 + 25
```

Partial credit is always awarded — picking 1 of 2 items still scores ~0.35.

---

## Setup & Usage

### Local (Python)
```bash
pip install -r requirements.txt
python server.py
# Server runs at http://localhost:8000
```

### Docker
```bash
docker build -t warehouse-robot-dispatch .
docker run -p 8000:8000 warehouse-robot-dispatch
```

### API Usage
```bash
# Start episode
curl -X POST http://localhost:8000/reset \
     -H "Content-Type: application/json" \
     -d '{"task": "multi_pick"}'

# Take action
curl -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action": "move_right"}'

# Get current state
curl http://localhost:8000/state
```

### Run Inference (LLM agent)
```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export WAREHOUSE_TASK=multi_pick
export ENV_BASE_URL=http://localhost:8000

python inference.py
```

---

## Baseline Scores

Measured with Qwen2.5-72B-Instruct (greedy, temperature=0.2):

| Task | Avg Score | Avg Steps | Notes |
|------|-----------|-----------|-------|
| `single_pick` | ~0.65 | ~18 | Usually finds item |
| `multi_pick` | ~0.45 | ~45 | Often picks 1 of 2 |
| `congested_floor` | ~0.25 | ~80 | Struggles with congestion |

---

## File Structure

```
warehouse-robot-dispatch/
├── src/
│   └── env/
│       ├── warehouse_env.py   # Core Gymnasium environment
│       ├── entities.py        # Robot, Item, BlockedZone dataclasses
│       ├── grid.py            # Grid engine (NumPy)
│       └── utils.py           # Utilities
├── server.py                  # FastAPI OpenEnv HTTP server
├── inference.py               # Baseline LLM agent script
├── openenv.yaml               # OpenEnv spec
├── Dockerfile                 # Container definition
├── requirements.txt
└── README.md
```

---

## OpenEnv Compliance

- `POST /reset` → returns initial observation
- `POST /step` → returns observation, reward, done, score
- `GET /state` → returns current state
- `GET /health` → liveness check
- Score always in `[0.0, 1.0]`
- Reward signal at every step (not binary end-of-episode)
- 3 tasks: easy → medium → hard with deterministic graders
# warehouse.rl


