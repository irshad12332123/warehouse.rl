"""
entities.py — Entity classes for the Warehouse Robot environment.

Robot  → the autonomous mobile robot navigating the warehouse
Item   → an inventory item on the warehouse floor to be picked
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Robot:
    """Autonomous mobile robot navigating the warehouse floor."""
    x:          int
    y:          int
    shift_time: int        # remaining shift time (acts like battery)
    picks:      int = 0    # number of items successfully picked

    def consume(self) -> None:
        """Consume one unit of shift time per step."""
        self.shift_time = max(0, self.shift_time - 1)

    @property
    def is_out_of_time(self) -> bool:
        return self.shift_time <= 0


@dataclass
class Item:
    """An inventory item placed on the warehouse floor."""
    x:         int
    y:         int
    is_picked: bool = False

    def pick(self) -> None:
        """Mark item as picked."""
        self.is_picked = True


# ── Backward compatibility aliases (drone_env used Drone/Victim) ──────
@dataclass
class Drone:
    """Alias kept for backward compatibility with drone_env.py."""
    x:        int
    y:        int
    battery:  int
    rescues:  int = 0

    def consume(self) -> None:
        self.battery = max(0, self.battery - 1)

    @property
    def is_battery_dead(self) -> bool:
        return self.battery <= 0


@dataclass
class Victim:
    """Alias kept for backward compatibility with drone_env.py."""
    x:          int
    y:          int
    is_rescued: bool = False

    def rescue(self) -> None:
        self.is_rescued = True
