from __future__ import annotations
import urllib.request
import json
import os
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ================= CONFIG =================
API_KEY: Optional[str] = os.getenv("HF_TOKEN")
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

TASKS: List[str] = ["single_pick", "multi_pick", "congested_floor"]
BENCHMARK: str = "warehouse_robot_dispatch"

MAX_STEPS: int = 60

ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8000")

VALID_ACTIONS: List[str] = ["move_up", "move_down", "move_left", "move_right"]
SUCCESS_SCORE_THRESHOLD: float = 0.1

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
    """
    Parse the ASCII warehouse string into a 2D list.
    Only lines that look like grid rows (containing '.', 'R', 'I', 'X') are kept;
    the status line (Step N/M | ...) is discarded.
    """
    rows: List[List[str]] = []
    for line in grid_str.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip status lines (contain '|' or 'Step')
        if "|" in line or line.startswith("Step"):
            continue
        cells = line.split()
        # Only keep rows that look like grid cells
        if cells and all(c in {".", "R", "I", "X"} for c in cells):
            rows.append(cells)
    return rows


def find_positions(grid: List[List[str]], target: str) -> List[Tuple[int, int]]:
    positions: List[Tuple[int, int]] = []
    for r in range(len(grid)):
        if not grid[r]:
            continue
        for c in range(len(grid[r])):
            if grid[r][c] == target:
                positions.append((r, c))
    return positions


def neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int, str]]:
    r, c = pos
    return [
        (r - 1, c, "move_up"),
        (r + 1, c, "move_down"),
        (r, c - 1, "move_left"),
        (r, c + 1, "move_right"),
    ]

# ================= BFS =================
def bfs_next_move(grid: List[List[str]]) -> Optional[str]:
    """
    BFS from the robot position to the nearest inventory item.
    Returns the first action to take on the shortest path, or None if
    no path exists or the grid has no robot / items.
    """
    if not grid or not grid[0]:
        return None

    robot_positions: List[Tuple[int, int]] = find_positions(grid, "R")
    item_positions: List[Tuple[int, int]] = find_positions(grid, "I")

    if not robot_positions or not item_positions:
        return None

    start: Tuple[int, int] = robot_positions[0]
    targets = set(item_positions)

    queue: Deque[Tuple[int, int]] = deque([start])
    visited: set = {start}
    parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}

    while queue:
        current = queue.popleft()

        if current in targets:
            # Backtrack to find the first step from start
            path_node = current
            while path_node != start:
                prev, action = parent[path_node]
                if prev == start:
                    return action
                path_node = prev
            # start == target (shouldn't happen, but safe)
            return None

        for nr, nc, action in neighbors(current):
            if (
                0 <= nr < len(grid)
                and 0 <= nc < len(grid[nr])
                and (nr, nc) not in visited
                and grid[nr][nc] != "X"
            ):
                visited.add((nr, nc))
                parent[(nr, nc)] = (current, action)
                queue.append((nr, nc))

    return None

# ================= LLM =================
def llm_action(client: OpenAI, warehouse: str, history: List[str]) -> str:
    """
    Ask the LLM for a move when BFS cannot find a path.
    Falls back to a random valid action if the model returns garbage.
    """
    recent = history[-4:] if history else ["None"]
    prompt: str = (
        f"Warehouse grid (R=robot, I=item, X=blocked, .=empty):\n"
        f"{warehouse}\n\n"
        f"Recent moves: {recent}\n\n"
        f"Choose the single best next move to reach an item (I).\n"
        f"Reply with exactly one of: move_up, move_down, move_left, move_right"
    )

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You control a warehouse robot. "
                        "Respond with ONLY one of: move_up, move_down, move_left, move_right. "
                        "No explanation, no punctuation, nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,   # deterministic — we want the model's best guess
            max_tokens=10,
        )

        raw: str = (res.choices[0].message.content or "").strip().lower()

        # Strip any accidental punctuation the model might add
        raw = raw.strip(".,!?\"' ")

        if raw in VALID_ACTIONS:
            return raw

    except Exception:
        pass

    return random.choice(VALID_ACTIONS)

# ================= EPISODE =================
def run_episode(client: OpenAI, task_name: str) -> None:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False
    consecutive_bad: int = 0        # counts consecutive collision/bad moves
    last_actions: List[str] = []    # short window to detect oscillation

    log_start(task_name, BENCHMARK, MODEL_NAME)

    try:
        result: Dict = http_post("/reset", {"task": task_name})

        warehouse: str = result.get("warehouse", "")
        done: bool = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            grid: List[List[str]] = parse_grid(warehouse)

            # Primary: BFS optimal path
            action: Optional[str] = bfs_next_move(grid)

            # Fallback: LLM (only when BFS cannot find a path, e.g. all items picked
            # or grid parse failed)
            if not action:
                action = llm_action(client, warehouse, history)

            if action not in VALID_ACTIONS:
                action = random.choice(VALID_ACTIONS)

            # Oscillation breaker: if the last 4 actions are just two alternating
            # moves (e.g. up/down/up/down), inject a perpendicular action.
            if len(last_actions) >= 4:
                window = last_actions[-4:]
                unique = set(window)
                opposites = {
                    ("move_up", "move_down"), ("move_down", "move_up"),
                    ("move_left", "move_right"), ("move_right", "move_left"),
                }
                if len(unique) == 2 and tuple(sorted(unique)) in {
                    tuple(sorted(p)) for p in opposites
                }:
                    perpendicular = [
                        a for a in VALID_ACTIONS
                        if a not in unique
                    ]
                    if perpendicular:
                        action = random.choice(perpendicular)

            error_msg: Optional[str] = None

            try:
                result = http_post("/step", {"action": action})

                reward: float = float(result.get("reward", 0.0))
                done = result.get("done", False)
                warehouse = result.get("warehouse", warehouse)
                score = float(result.get("score", score))

                if reward <= -0.5:   # collision penalty threshold
                    consecutive_bad += 1
                else:
                    consecutive_bad = 0

            except Exception as e:
                reward = 0.0
                error_msg = str(e)
                consecutive_bad += 1

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, error_msg)
            history.append(f"{action}:{reward:.2f}")
            last_actions.append(action)
            if len(last_actions) > 8:
                last_actions.pop(0)

        score = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(success, steps_taken, score, rewards)

# ================= MAIN =================
def main() -> None:
    if not API_KEY:
        raise RuntimeError("HF_TOKEN is not set")

    client: OpenAI = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in TASKS:
        run_episode(client, task)


if __name__ == "__main__":
    main()
