"""
grid.py — Simple 2D grid for the Warehouse Robot environment.

Cell values:
    0 = EMPTY
    1 = OBSTACLE  (blocked aisle / worker zone)
    2 = VICTIM    (inventory item) — kept as VICTIM for Grid compatibility
    3 = DRONE     (robot) — kept as DRONE for Grid compatibility
"""
from __future__ import annotations
import random
from typing import Optional


class Grid:
    EMPTY    = 0
    OBSTACLE = 1
    VICTIM   = 2   # used for inventory items in warehouse context
    DRONE    = 3   # used for robot in warehouse context

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height
        self.cells  = [[self.EMPTY] * width for _ in range(height)]

    def get(self, x: int, y: int) -> Optional[int]:
        """Return cell value, or None if out of bounds."""
        if 0 <= x < self.height and 0 <= y < self.width:
            return self.cells[x][y]
        return None

    def place(self, x: int, y: int, value: int) -> bool:
        """Place value only if cell is EMPTY. Returns True on success."""
        if self.get(x, y) == self.EMPTY:
            self.cells[x][y] = value
            return True
        return False

    def force_place(self, x: int, y: int, value: int) -> None:
        """Place value regardless of current cell content."""
        if 0 <= x < self.height and 0 <= y < self.width:
            self.cells[x][y] = value

    def remove(self, x: int, y: int) -> None:
        """Set cell to EMPTY."""
        if 0 <= x < self.height and 0 <= y < self.width:
            self.cells[x][y] = self.EMPTY

    def random_empty(self) -> Optional[tuple[int, int]]:
        """Return a random empty cell, or None if grid is full."""
        empty = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if self.cells[r][c] == self.EMPTY
        ]
        return random.choice(empty) if empty else None

    def render(self) -> str:
        """Render grid as ASCII string."""
        symbols = {
            self.EMPTY:    ".",
            self.OBSTACLE: "#",
            self.VICTIM:   "V",
            self.DRONE:    "D",
        }
        rows = []
        for row in self.cells:
            rows.append(" ".join(symbols.get(c, "?") for c in row))
        return "\n".join(rows)
