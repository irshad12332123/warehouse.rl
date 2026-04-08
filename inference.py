from __future__ import annotations

import json
import os
import random
import textwrap
import urllib.request
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ================= CONFIG =================
API_KEY: Optional[str] = os.getenv("HF_TOKEN")
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

TASK_NAME: str = os.getenv("WAREHOUSE_TASK", "multi_pick")
BENCHMARK: str = "warehouse_robot_dispatch"

MAX_STEPS: int = 60
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 30

ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8000")

VALID_ACTIONS: List[str] = ["move_up", "move_down", "move_left", "move_right"]
SUCCESS_SCORE_THRESHOLD: float = 0.1

# ================= PROMPT =================
SYSTEM_PROMPT: str = """
You control a warehouse robot.

Goal: collect all items efficiently.

Avoid obstacles. Do not repeat failed moves.

Return ONLY:
move_up / move_down / move_left / move_right
"""

# ================= LOGGING =================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str: str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ================= HTTP =================
def http_post(path: str, body: Dict) -> Dict:
    data: bytes = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{ENV_BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


# ================= GRID =================
def parse_grid(grid_str: str) -> List[List[str]]:
    return [row.split() for row in grid_str.strip().split("\n")]


def find_positions(grid: List[List[str]], target: str) -> List[Tuple[int, int]]:
    return [
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if grid[r][c] == target
    ]


def neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int, str]]:
    r, c = pos
    return [
        (r - 1, c, "move_up"),
        (r + 1, c, "move_down"),
        (r, c - 1, "move_left"),
        (r, c + 1, "move_right"),
    ]


# ================= BFS PATHFINDING =================
def bfs_next_move(grid: List[List[str]]) -> Optional[str]:
    robot_positions = find_positions(grid, "R")
    item_positions = find_positions(grid, "I")

    if not robot_positions or not item_positions:
        return None

    start = robot_positions[0]
    targets = set(item_positions)

    queue: Deque[Tuple[int, int]] = deque([start])
    visited: set = {start}
    parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}

    while queue:
        current = queue.popleft()

        if current in targets:
            # backtrack
            while current != start:
                prev, action = parent[current]
                if prev == start:
                    return action
                current = prev

        for nr, nc, action in neighbors(current):
            if (
                0 <= nr < len(grid)
                and 0 <= nc < len(grid[0])
                and (nr, nc) not in visited
                and grid[nr][nc] != "X"
            ):
                visited.add((nr, nc))
                parent[(nr, nc)] = (current, action)
                queue.append((nr, nc))

    return None


# ================= LLM =================
def llm_action(client: OpenAI, warehouse: str, history: List[str]) -> str:
    prompt = f"""
Warehouse:
{warehouse}

Recent:
{history[-4:] if history else "None"}

Next move?
"""

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        raw = (res.choices[0].message.content or "").strip().lower()
        if raw in VALID_ACTIONS:
            return raw

    except Exception:
        pass

    return random.choice(VALID_ACTIONS)


# ================= MAIN =================
def run_episode(client: OpenAI, task_name: str) -> None:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False
    last_bad_action: Optional[str] = None

    log_start(task_name, BENCHMARK, MODEL_NAME)

    try:
        result = http_post("/reset", {"task": task_name})
        warehouse: str = result["warehouse"]
        done: bool = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            grid = parse_grid(warehouse)
            action = bfs_next_move(grid)

            if not action:
                action = llm_action(client, warehouse, history)

            if last_bad_action == action:
                action = random.choice([a for a in VALID_ACTIONS if a != action])

            error_msg: Optional[str] = None

            try:
                result = http_post("/step", {"action": action})
                reward: float = float(result.get("reward", 0.0))
                done = result.get("done", False)
                warehouse = result.get("warehouse", warehouse)
                score = float(result.get("score", score))

                if reward < 0:
                    last_bad_action = action
                else:
                    last_bad_action = None

            except Exception as e:
                reward = 0.0
                error_msg = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, error_msg)
            history.append(f"{action}:{reward:.2f}")

        score = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(success, steps_taken, score, rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks: List[str] = [
        "single_pick",
        "multi_pick",
        "congested_floor",
    ]

    for task in tasks:
        run_episode(client, task)

if __name__ == "__main__":
    main()
