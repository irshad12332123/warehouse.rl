# Reward Logic Bug Analysis

## Problem Summary
The score is always showing 1.0 because the **reward normalization formula is broken**.

## Root Cause

### Broken Normalization in `server.py`

```python
def _max_reward(cfg: dict) -> float:
    return cfg["num_items"] * 10.0 + 25.0

def _compute_score(total_reward: float, task: str) -> float:
    max_r = _max_reward(TASK_CONFIGS[task])
    raw   = (total_reward + 5.0) / (max_r + 5.0)  # ❌ WRONG
    return float(max(0.0, min(1.0, raw)))
```

### What's Wrong?

1. **Incomplete max_reward calculation**
   - Only counts: `num_items * 10.0 + 25.0`
   - Ignores: step costs (-0.1 per step), potential blocked cells, reward shaping

2. **Arbitrary +5.0 offset**
   - No justification for this offset
   - Causes HIGH total_rewards to exceed max_reward
   - Results in `score = 1.0` (clamped)

3. **Example Breakdown for "multi_pick"**:
   - `max_r = 2 * 10.0 + 25.0 = 45.0`
   - Denominator: `45 + 5 = 50`
   - A task getting total_reward ≥ 45 → `score = 1.0`
   - But you're likely getting 41-50 reward through combination of:
     - 2 picks: +20
     - Completion: +25
     - ~40 steps: -4
     - Shaping rewards: +2 to +5
     - **Total ≈ 43-46 (exceeds max_r!) → score = 1.0**

## The Fix

Replace the normalization function to properly account for actual rewards:

```python
def _compute_score(total_reward: float, task: str) -> float:
    """
    Normalize task reward to [0, 1] range.

    Maximum achievable reward accounting for:
    - Picking all items (num_items * R_PICK)
    - Completion bonus (R_COMPLETION)
    - Minimum step cost (rough estimate)
    """
    cfg = TASK_CONFIGS[task]

    # Components of max reward
    pick_reward = cfg["num_items"] * 10.0  # R_PICK
    completion_reward = 25.0               # R_COMPLETION

    # Rough estimate of minimum steps needed (Manhattan dist + buffer)
    estimated_min_steps = (cfg["width"] + cfg["height"]) * cfg["num_items"]
    step_cost = -0.1 * estimated_min_steps  # R_STEP cost

    # Account for shaping (conservative estimate, ~5% of item rewards)
    shaping_estimate = (pick_reward * 0.05) if cfg.get("use_shaping", True) else 0

    # Calculate realistic max
    realistic_max = pick_reward + completion_reward + step_cost + shaping_estimate

    # Normalize without arbitrary offsets
    if realistic_max <= 0:
        realistic_max = 1.0  # Avoid division by zero

    score = total_reward / realistic_max
    return float(max(0.0, min(1.0, score)))
```

## Tasks Affected

- **single_pick**: max_r = 35.0 (should be ~25-28)
- **multi_pick**: max_r = 45.0 (should be ~35-40)
- **congested_floor**: max_r = 55.0 (should be ~45-50)

All tasks getting inflated scores because actual rewards don't approach theoretical max.
