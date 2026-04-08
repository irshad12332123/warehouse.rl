"""
inference.py — Warehouse Robot Dispatch OpenEnv inference script.

Runs an LLM agent against the warehouse environment and emits:
    [START] task=<task> env=warehouse_robot_dispatch model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Tasks:
    single_pick      (easy)
    multi_pick       (medium, default)
    congested_floor  (hard)
"""

from __future__ import annotations

import json
import os
import textwrap
import urllib.request
from typing import List, Optional

from openai import OpenAI

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.getenv("WAREHOUSE_TASK", "multi_pick")
BENCHMARK    = "warehouse_robot_dispatch"
MAX_STEPS    = 60
TEMPERATURE  = 0.2
MAX_TOKENS   = 20
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

VALID_ACTIONS = ["move_up", "move_down", "move_left", "move_right"]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI dispatcher controlling an autonomous robot in a warehouse fulfillment center.

    Your job: navigate the robot to pick all inventory items as efficiently as possible.

    Warehouse grid legend:
        R = Robot (you control this)
        I = Inventory item (navigate here to pick it)
        X = Blocked aisle or active worker zone (avoid — collision penalized)
        . = Empty aisle (safe to move through)

    Rules:
        - Moving into X or a wall wastes shift time and gets penalized (-0.5)
        - Picking an item (moving onto I) gives +10 reward
        - Picking ALL items gives +25 completion bonus
        - Shift time is limited — plan an efficient route

    Respond with EXACTLY ONE of:
        move_up
        move_down
        move_left
        move_right
    Nothing else. No explanation.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def get_action(client: OpenAI, warehouse: str, info: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"

    user_prompt = textwrap.dedent(f"""
        Current warehouse layout:
        {warehouse}

        Status:
            Robot position  : {info.get('robot_pos', '?')}
            Items remaining : {info.get('items_remaining', '?')}
            Shift time left : {info.get('shift_time', '?')}
            Items picked    : {info.get('picks', 0)}

        Recent moves:
        {history_block}

        What is your next move?
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip().lower()

        for a in VALID_ACTIONS:
            if a in raw:
                return a

        word = raw.split()[0] if raw else ""
        fallback = {"up": "move_up", "down": "move_down",
                    "left": "move_left", "right": "move_right"}
        return fallback.get(word, "move_up")

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "move_up"


def http_post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{ENV_BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def http_get(path: str) -> dict:
    with urllib.request.urlopen(f"{ENV_BASE_URL}{path}", timeout=10) as r:
        return json.loads(r.read())


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result    = http_post("/reset", {"task": TASK_NAME})
        warehouse = result["warehouse"]
        info      = result["info"]
        done      = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action    = get_action(client, warehouse, info, history)
            error_msg = None

            try:
                result    = http_post("/step", {"action": action})
                reward    = float(result.get("reward", 0.0))
                done      = result.get("done", False)
                score     = float(result.get("score", 0.0))
                warehouse = result.get("warehouse", warehouse)
                info      = result.get("info", info)
            except Exception as e:
                reward    = 0.0
                error_msg = str(e)[:80]
                done      = False

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward,
                     done=done, error=error_msg)

            history.append(
                f"Step {step}: {action} → reward {reward:+.2f} | "
                f"shift={info.get('shift_time','?')} items={info.get('items_remaining','?')}"
            )

        # Get final score from state
        if not done:
            try:
                st    = http_get("/state")
                score = float(st.get("score", score))
            except Exception:
                pass

        success = score >= 0.1

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
