"""
warehouse_env.py — Gymnasium-compatible Warehouse Robot Dispatch environment.

Real-world motivation:
    Modern fulfillment centers (Amazon, Flipkart, Myntra) use autonomous mobile
    robots (AMRs) to navigate warehouse floors, pick inventory items, and return
    to dispatch stations. Human supervisors monitor and dispatch these robots.
    This environment simulates that exact task:
      - Robot navigates a warehouse grid (floor plan)
      - Must pick all assigned inventory items
      - Must avoid blocked aisles and worker zones (obstacles)
      - Shift time (battery) is limited — efficiency matters

Observation (normalized to [0, 1]):
    For each item slot:
        rel_row, rel_col, dist, active
    Robot state:
        robot_row, robot_col, shift_remaining%, items_left%
    Proximity sensors (1-step look-ahead):
        blocked_up, blocked_down, blocked_left, blocked_right

Reward design:
    R_STEP=-0.1        (time cost — efficiency matters in warehouses)
    R_BLOCKED=-0.5     (collision with wall/obstacle — safety penalty)
    R_PICK=+10.0       (item successfully picked)
    R_COMPLETION=+25.0 (all items picked — shift bonus)
    R_TIMEOUT=-5.0     (shift ended without completion)
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.grid import Grid
from src.env.entities import Robot, Item


# ================= CONSTANTS =================
ACTIONS: Dict[int, Tuple[int, int]] = {
    0: (-1,  0),   # move_up
    1: ( 1,  0),   # move_down
    2: ( 0, -1),   # move_left
    3: ( 0,  1),   # move_right
}

R_STEP: float       = -0.1
R_BLOCKED: float    = -0.5
R_PICK: float       = 10.0
R_COMPLETION: float = 25.0
R_TIMEOUT: float    = -5.0

SHAPING_WEIGHT: float = 0.3


# ================= ENV =================
class WarehouseEnv(gym.Env):

    metadata: Dict[str, List[str]] = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        difficulty: str = "medium",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.difficulty: str = difficulty

        # ===== Difficulty Configuration =====
        self.width: int
        self.height: int
        self.num_items: int
        self.num_blocked: int
        self.shift_time: int
        self.max_steps: int
        self.use_shaping: bool

        if difficulty == "easy":
            self.width = 5
            self.height = 5
            self.num_items = 2
            self.num_blocked = 2
            self.shift_time = 100
            self.max_steps = 100
            self.use_shaping = True

        elif difficulty == "medium":
            self.width = 7
            self.height = 7
            self.num_items = 4
            self.num_blocked = 6
            self.shift_time = 120
            self.max_steps = 120
            self.use_shaping = True

        elif difficulty == "hard":
            self.width = 10
            self.height = 10
            self.num_items = 6
            self.num_blocked = 15
            self.shift_time = 100
            self.max_steps = 100
            self.use_shaping = False

        else:
            raise ValueError(f"Invalid difficulty: {difficulty}")

        self._diag: float = float(self.height + self.width)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        obs_dim: int = self.num_items * 4 + 8

        self.observation_space: spaces.Box = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space: spaces.Discrete = spaces.Discrete(4)

        # ===== Runtime State =====
        self.grid: Optional[Grid] = None
        self.robot: Optional[Robot] = None
        self.items: List[Item] = []

        self.steps: int = 0
        self._prev_shaping: float = 0.0

    # ================= RESET =================
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.grid = Grid(self.width, self.height)
        self.steps = 0
        self.items = []

        # Place obstacles
        for _ in range(self.num_blocked):
            pos: Optional[Tuple[int, int]] = self.grid.random_empty()
            if pos:
                self.grid.place(*pos, Grid.OBSTACLE)

        # Place robot
        pos = self.grid.random_empty()
        if pos is None:
            raise RuntimeError("Grid full — cannot place robot")

        self.robot = Robot(x=pos[0], y=pos[1], shift_time=self.shift_time)
        self.grid.force_place(pos[0], pos[1], Grid.DRONE)

        # Place items
        for _ in range(self.num_items):
            pos = self.grid.random_empty()
            if pos:
                item = Item(x=pos[0], y=pos[1])
                self.items.append(item)
                self.grid.place(pos[0], pos[1], Grid.VICTIM)

        self._prev_shaping = self._potential()

        return self._obs(), {}

    # ================= STEP =================
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        assert self.grid is not None
        assert self.robot is not None

        dx, dy = ACTIONS[action]
        cx, cy = self.robot.x, self.robot.y
        nx, ny = cx + dx, cy + dy

        terminated: bool = False
        truncated: bool = False
        reward: float = R_STEP

        cell = self.grid.get(nx, ny)

        # Blocked or wall
        if cell is None or cell == Grid.OBSTACLE:
            reward = R_BLOCKED
            self.steps += 1
            self.robot.consume()

            if self.robot.shift_time <= 0:
                reward += R_TIMEOUT
                terminated = True

            truncated = (not terminated) and (self.steps >= self.max_steps)

            return self._obs(), reward, terminated, truncated, self._info()

        # Move robot
        self.grid.remove(cx, cy)

        if cell == Grid.VICTIM:
            for item in self.items:
                if not item.is_picked and item.x == nx and item.y == ny:
                    item.pick()
                    self.robot.picks += 1
                    reward += R_PICK
                    break

        self.robot.x, self.robot.y = nx, ny
        self.grid.force_place(nx, ny, Grid.DRONE)

        self.robot.consume()
        self.steps += 1

        # Reward shaping
        if self.use_shaping:
            curr: float = self._potential()
            reward += SHAPING_WEIGHT * (0.99 * curr - self._prev_shaping)
            self._prev_shaping = curr

        # Termination
        if all(item.is_picked for item in self.items):
            reward += R_COMPLETION
            terminated = True

        elif self.robot.shift_time <= 0:
            reward += R_TIMEOUT
            terminated = True

        truncated = (not terminated) and (self.steps >= self.max_steps)

        return self._obs(), reward, terminated, truncated, self._info()

    # ================= OBS =================
    def _obs(self) -> np.ndarray:
        assert self.robot is not None
        assert self.grid is not None

        H: int = self.height
        W: int = self.width

        rx, ry = self.robot.x, self.robot.y

        parts: List[float] = []

        for item in self.items:
            if not item.is_picked:
                rel_x = (item.x - rx + H) / (2 * H)
                rel_y = (item.y - ry + W) / (2 * W)
                dist  = (abs(item.x - rx) + abs(item.y - ry)) / self._diag
                active = 1.0
            else:
                rel_x = rel_y = dist = 0.0
                active = 0.0

            parts.extend([rel_x, rel_y, dist, active])

        parts.append(rx / max(H - 1, 1))
        parts.append(ry / max(W - 1, 1))
        parts.append(self.robot.shift_time / self.shift_time)
        parts.append(self._items_remaining / max(self.num_items, 1))

        for dx, dy in ACTIONS.values():
            nx, ny = rx + dx, ry + dy
            cell = self.grid.get(nx, ny)
            blocked = 1.0 if (cell is None or cell == Grid.OBSTACLE) else 0.0
            parts.append(blocked)

        return np.array(parts, dtype=np.float32)

    # ================= HELPERS =================
    @property
    def _items_remaining(self) -> int:
        return sum(1 for item in self.items if not item.is_picked)

    def _potential(self) -> float:
        active = [i for i in self.items if not i.is_picked]
        if not active:
            return 0.0

        min_dist = min(
            abs(self.robot.x - i.x) + abs(self.robot.y - i.y)
            for i in active
        )
        return -(min_dist / self._diag)

    def _info(self) -> Dict[str, Any]:
        assert self.robot is not None

        return {
            "step": self.steps,
            "items_remaining": self._items_remaining,
            "shift_time": self.robot.shift_time,
            "picks": self.robot.picks,
        }

    # ================= RENDER =================
    def render(self) -> str:
        if self.grid is None:
            return "Environment not initialized."

        grid_str: str = self.grid.render()
        grid_str = grid_str.replace("D", "R").replace("V", "I").replace("#", "X")

        return (
            f"{grid_str}\n"
            f"Step {self.steps}/{self.max_steps} | "
            f"Shift: {self.robot.shift_time} | "
            f"Picked: {self.robot.picks}/{self.num_items}"
        )

    def close(self) -> None:
        pass
