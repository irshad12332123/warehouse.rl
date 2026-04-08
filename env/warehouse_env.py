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
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.grid import Grid
from src.env.entities import Robot, Item


ACTIONS: dict[int, tuple[int, int]] = {
    0: (-1,  0),   # move_up    (north)
    1: ( 1,  0),   # move_down  (south)
    2: ( 0, -1),   # move_left  (west)
    3: ( 0,  1),   # move_right (east)
}

R_STEP       = -0.1
R_BLOCKED    = -0.5
R_PICK       =  10.0
R_COMPLETION =  25.0
R_TIMEOUT    = -5.0
SHAPING_WEIGHT = 0.3


class WarehouseEnv(gym.Env):
    """
    Grid-based Warehouse Robot Dispatch environment.

    The robot must navigate the warehouse floor plan, pick all inventory
    items, and avoid blocked aisles/worker zones within shift time.

    Grid symbols:
        . = empty aisle
        R = robot
        I = inventory item
        X = blocked aisle / worker zone (obstacle)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        width:         int  = 5,
        height:        int  = 5,
        num_items:     int  = 2,
        num_blocked:   int  = 3,
        max_steps:     int  = 100,
        shift_time:    int  = 100,
        use_shaping:   bool = True,
        seed:          int | None = None,
    ):
        super().__init__()

        self.width       = width
        self.height      = height
        self.num_items   = num_items
        self.num_blocked = num_blocked
        self.max_steps   = max_steps
        self.shift_time  = shift_time
        self.use_shaping = use_shaping
        self._diag       = float(height + width)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        obs_dim = num_items * 4 + 8
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.grid:  Grid | None  = None
        self.robot: Robot | None = None
        self.items: list[Item]   = []
        self.steps = 0
        self._prev_shaping: float = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.grid  = Grid(self.width, self.height)
        self.steps = 0
        self.items = []

        # Place blocked aisles / worker zones
        for _ in range(self.num_blocked):
            pos = self.grid.random_empty()
            if pos:
                self.grid.place(*pos, Grid.OBSTACLE)

        # Place robot
        pos = self.grid.random_empty()
        if pos is None:
            raise RuntimeError("Warehouse too congested — reduce num_blocked.")
        self.robot = Robot(x=pos[0], y=pos[1], shift_time=self.shift_time)
        self.grid.force_place(pos[0], pos[1], Grid.DRONE)

        # Place inventory items
        for _ in range(self.num_items):
            pos = self.grid.random_empty()
            if pos:
                item = Item(x=pos[0], y=pos[1])
                self.items.append(item)
                self.grid.place(pos[0], pos[1], Grid.VICTIM)

        self._prev_shaping = self._potential()
        return self._obs(), {}

    def step(self, action: int):
        assert self.grid is not None, "Call reset() before step()."

        dx, dy = ACTIONS[action]
        cx, cy = self.robot.x, self.robot.y
        nx, ny = cx + dx, cy + dy

        terminated = False
        truncated  = False
        reward     = R_STEP

        # Blocked aisle / wall collision
        cell = self.grid.get(nx, ny)
        if cell is None or cell == Grid.OBSTACLE:
            reward = R_BLOCKED
            self.steps += 1
            self.robot.consume()
            if self.robot.shift_time <= 0:
                reward    += R_TIMEOUT
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

        # Distance shaping
        if self.use_shaping:
            curr              = self._potential()
            reward           += SHAPING_WEIGHT * (0.99 * curr - self._prev_shaping)
            self._prev_shaping = curr

        # Termination
        if all(item.is_picked for item in self.items):
            reward    += R_COMPLETION
            terminated = True
        elif self.robot.shift_time <= 0:
            reward    += R_TIMEOUT
            terminated = True

        truncated = (not terminated) and (self.steps >= self.max_steps)
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self) -> str:
        if self.grid is None:
            return "Not initialized."
        # Override symbols for warehouse theme
        grid_str = self.grid.render()
        grid_str = grid_str.replace("D", "R").replace("V", "I").replace("#", "X")
        return (
            f"{grid_str}\n"
            f"Step {self.steps}/{self.max_steps} | "
            f"Shift time left: {self.robot.shift_time} | "
            f"Items picked: {self.robot.picks}/{self.num_items}"
        )

    def close(self): pass

    def _obs(self) -> np.ndarray:
        H, W  = self.height, self.width
        rx, ry = self.robot.x, self.robot.y
        parts  = []

        for item in self.items:
            if not item.is_picked:
                rel_x  = (item.x - rx + H) / (2 * H)
                rel_y  = (item.y - ry + W) / (2 * W)
                dist   = (abs(item.x - rx) + abs(item.y - ry)) / self._diag
                active = 1.0
            else:
                rel_x = rel_y = dist = 0.0
                active = 0.0
            parts.extend([rel_x, rel_y, dist, active])

        parts.append(rx / max(H - 1, 1))
        parts.append(ry / max(W - 1, 1))
        parts.append(self.robot.shift_time / self.shift_time)
        parts.append(self._items_remaining / max(self.num_items, 1))

        for ddx, ddy in ACTIONS.values():
            nx, ny = rx + ddx, ry + ddy
            cell   = self.grid.get(nx, ny)
            blocked = 1.0 if (cell is None or cell == Grid.OBSTACLE) else 0.0
            parts.append(blocked)

        return np.array(parts, dtype=np.float32)

    @property
    def _items_remaining(self) -> int:
        return sum(1 for item in self.items if not item.is_picked)

    def _potential(self) -> float:
        active = [item for item in self.items if not item.is_picked]
        if not active:
            return 0.0
        min_d = min(abs(self.robot.x - i.x) + abs(self.robot.y - i.y) for i in active)
        return -(min_d / self._diag)

    def _info(self) -> dict[str, Any]:
        return {
            "step":           self.steps,
            "items_remaining": self._items_remaining,
            "shift_time":     self.robot.shift_time,
            "picks":          self.robot.picks,
        }
