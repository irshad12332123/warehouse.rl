"""
server.py — FastAPI server for the Warehouse Robot Dispatch OpenEnv environment.

Exposes the standard OpenEnv HTTP API:
    POST /reset   → start new episode, returns initial warehouse state
    POST /step    → execute one robot action, returns obs/reward/done/score
    GET  /state   → current warehouse state
    GET  /health  → liveness check

Action space (text strings the LLM sends):
    "move_up"    → robot moves north  (-1, 0)
    "move_down"  → robot moves south  (+1, 0)
    "move_left"  → robot moves west   (0, -1)
    "move_right" → robot moves east   (0, +1)

Tasks:
    "single_pick"     → 1 item, small warehouse 5x5, 3 blocked zones (easy)
    "multi_pick"      → 2 items, standard warehouse 5x5, 3 blocked zones (medium)
    "congested_floor" → 3 items, large warehouse 7x7, 8 blocked zones (hard)
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.env.warehouse_env import WarehouseEnv

TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "single_pick": {
        "width": 5, "height": 5,
        "num_items": 1, "num_blocked": 3,
        "max_steps": 50, "shift_time": 50,
        "description": (
            "Small warehouse (5x5). Robot must pick 1 inventory item "
            "while avoiding 3 blocked aisles. Simulates a simple "
            "pick-and-dispatch task in a fulfillment center."
        ),
        "difficulty": "easy",
    },
    "multi_pick": {
        "width": 5, "height": 5,
        "num_items": 2, "num_blocked": 3,
        "max_steps": 100, "shift_time": 100,
        "description": (
            "Standard warehouse (5x5). Robot must pick 2 inventory items "
            "within shift time. Requires planning an efficient route. "
            "Simulates multi-item order fulfillment."
        ),
        "difficulty": "medium",
    },
    "congested_floor": {
        "width": 7, "height": 7,
        "num_items": 3, "num_blocked": 8,
        "max_steps": 150, "shift_time": 150,
        "description": (
            "Large congested warehouse (7x7) with 8 blocked zones. "
            "Robot must pick 3 items while navigating dense obstacles. "
            "Simulates peak-hour warehouse operations with worker zones."
        ),
        "difficulty": "hard",
    },
}

DEFAULT_TASK = "multi_pick"

ACTION_MAP: dict[str, int] = {
    "move_up": 0, "move_down": 1, "move_left": 2, "move_right": 3,
    "up": 0, "down": 1, "left": 2, "right": 3,
    "north": 0, "south": 1, "west": 2, "east": 3,
    "0": 0, "1": 1, "2": 2, "3": 3,
}


def _max_reward(cfg: dict) -> float:
    return cfg["num_items"] * 10.0 + 25.0


def _compute_score(total_reward: float, task: str) -> float:
    max_r = _max_reward(TASK_CONFIGS[task])
    raw   = (total_reward + 5.0) / (max_r + 5.0)
    return float(max(0.0, min(1.0, raw)))


_env:          WarehouseEnv | None = None
_task_name:    str   = DEFAULT_TASK
_total_reward: float = 0.0
_steps_taken:  int   = 0
_done:         bool  = False

app = FastAPI(title="Warehouse Robot Dispatch — OpenEnv", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class ResetRequest(BaseModel):
    task: str = DEFAULT_TASK
    seed: int | None = None


class StepRequest(BaseModel):
    action: str


@app.get("/health")
def health():
    return {"status": "ok", "env": "warehouse-robot-dispatch"}


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    global _env, _task_name, _total_reward, _steps_taken, _done

    task = body.task if body.task in TASK_CONFIGS else DEFAULT_TASK
    cfg  = TASK_CONFIGS[task]
    _task_name    = task
    _total_reward = 0.0
    _steps_taken  = 0
    _done         = False

    _env = WarehouseEnv(
        width       = cfg["width"],
        height      = cfg["height"],
        num_items   = cfg["num_items"],
        num_blocked = cfg["num_blocked"],
        max_steps   = cfg["max_steps"],
        shift_time  = cfg["shift_time"],
        use_shaping = False,
        seed        = body.seed,
    )

    obs, _ = _env.reset(seed=body.seed)

    return {
        "task":        task,
        "difficulty":  cfg["difficulty"],
        "description": cfg["description"],
        "warehouse":   _env.render(),
        "observation": obs.tolist(),
        "info": {
            "robot_pos":       [_env.robot.x, _env.robot.y],
            "items_remaining": _env._items_remaining,
            "shift_time":      _env.robot.shift_time,
            "max_steps":       cfg["max_steps"],
            "action_space":    ["move_up", "move_down", "move_left", "move_right"],
            "grid_legend":     {"R": "Robot", "I": "Inventory item", "X": "Blocked aisle", ".": "Empty aisle"},
        },
        "done":  False,
        "score": 0.0,
    }


@app.post("/step")
def step(body: StepRequest):
    global _env, _total_reward, _steps_taken, _done

    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    if _done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset.")

    action_str = body.action.strip().lower()
    if action_str not in ACTION_MAP:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown action '{body.action}'. Valid: move_up, move_down, move_left, move_right",
        )

    action_int = ACTION_MAP[action_str]
    obs, reward, terminated, truncated, info = _env.step(action_int)

    _total_reward += reward
    _steps_taken  += 1
    _done          = terminated or truncated
    score          = _compute_score(_total_reward, _task_name)

    return {
        "action":      action_str,
        "warehouse":   _env.render(),
        "observation": obs.tolist(),
        "reward":      round(float(reward), 4),
        "done":        _done,
        "score":       round(score, 4),
        "info": {
            "step":            info["step"],
            "items_remaining": info["items_remaining"],
            "shift_time":      info["shift_time"],
            "picks":           info["picks"],
            "total_reward":    round(_total_reward, 4),
            "terminated":      terminated,
            "truncated":       truncated,
        },
    }


@app.get("/state")
def state():
    if _env is None:
        return {"initialized": False}

    obs   = _env._obs()
    score = _compute_score(_total_reward, _task_name)

    return {
        "initialized":  True,
        "task":         _task_name,
        "warehouse":    _env.render(),
        "observation":  obs.tolist(),
        "done":         _done,
        "score":        round(score, 4),
        "total_reward": round(_total_reward, 4),
        "steps_taken":  _steps_taken,
        "info": {
            "robot_pos":       [_env.robot.x, _env.robot.y],
            "items_remaining": _env._items_remaining,
            "shift_time":      _env.robot.shift_time,
            "picks":           _env.robot.picks,
        },
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
