"""Sutton & Barto Exercise 5.12 racetrack and on-policy MC control."""

from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ACTION_DELTAS = tuple((ax, ay) for ax in (1, 0, -1) for ay in (1, 0, -1))
State = tuple[int, int, int, int]  # x, y, vx, vy; y increases upward


@dataclass(frozen=True)
class Motion:
    kind: str  # move, crash, or finish
    x: int
    y: int


class Track:
    def __init__(self, name: str):
        data = json.loads((ROOT / "tracks.json").read_text(encoding="utf-8"))[name]
        rows = data["rows_top_down"]
        self.name = name
        self.height = len(rows)
        self.width = len(rows[0])
        assert all(len(row) == self.width for row in rows)
        self.cells = frozenset(
            (x, self.height - 1 - top)
            for top, row in enumerate(rows) for x, char in enumerate(row) if char == "#"
        )
        self.starts = tuple(tuple(cell) for cell in data["start"])
        self.finishes = frozenset(tuple(cell) for cell in data["finish"])
        assert set(self.starts) <= self.cells and self.finishes <= self.cells
        self.start_set = frozenset(self.starts)
        self.legal = {
            (vx, vy, at_start): tuple(
                a for a, (ax, ay) in enumerate(ACTION_DELTAS)
                if 0 <= vx + ax <= 4 and 0 <= vy + ay <= 4
                and ((vx + ax, vy + ay) != (0, 0) or at_start)
            )
            for vx in range(5) for vy in range(5) for at_start in (False, True)
        }
        # The outcome of a proposed velocity is deterministic; only action failure
        # and the random reset position need sampling during training.
        self.motion = {
            (x, y, vx, vy): self._trace(x, y, vx, vy)
            for x, y in self.cells for vx in range(5) for vy in range(5)
        }

    def _check_cells(self, cells: tuple[tuple[int, int], ...]) -> str | None:
        # At a grid corner, an off-track side cell counts as a collision.
        if any(cell not in self.cells for cell in cells):
            return "crash"
        if any(cell in self.finishes for cell in cells):
            return "finish"
        return None

    def _trace(self, x: int, y: int, vx: int, vy: int) -> Motion:
        if (vx, vy) == (0, 0):
            return Motion("move", x, y)
        end_x, end_y = x + vx, y + vy
        tx = 0.5 / vx if vx else float("inf")
        ty = 0.5 / vy if vy else float("inf")
        dx = 1 / vx if vx else float("inf")
        dy = 1 / vy if vy else float("inf")
        while (x, y) != (end_x, end_y):
            if tx < ty - 1e-12:
                x += 1
                touched = ((x, y),)
                tx += dx
            elif ty < tx - 1e-12:
                y += 1
                touched = ((x, y),)
                ty += dy
            else:
                touched = ((x + 1, y), (x, y + 1), (x + 1, y + 1))
                x += 1
                y += 1
                tx += dx
                ty += dy
            outcome = self._check_cells(touched)
            if outcome:
                return Motion(outcome, x, y)
        return Motion("move", x, y)

    def step(self, state: State, action: int, rng: random.Random,
             failure_probability: float = 0.1) -> tuple[State, bool, bool]:
        x, y, vx, vy = state
        if action not in self.legal[vx, vy, (x, y) in self.start_set]:
            raise ValueError(f"Illegal action {ACTION_DELTAS[action]} from {state}")
        ax, ay = ACTION_DELTAS[action]
        if rng.random() < failure_probability:
            ax = ay = 0
        new_vx, new_vy = vx + ax, vy + ay
        result = self.motion[x, y, new_vx, new_vy]
        if result.kind == "finish":
            return (result.x, result.y, new_vx, new_vy), True, False
        if result.kind == "crash":
            sx, sy = rng.choice(self.starts)
            return (sx, sy, 0, 0), False, True
        return (result.x, result.y, new_vx, new_vy), False, False


def new_q(track: Track) -> np.ndarray:
    # Optimistic zero values encourage exploration with step costs of -1.
    q = np.full((track.height, track.width, 5, 5, 9), -np.inf, dtype=np.float32)
    for x, y in track.cells:
        at_start = (x, y) in track.start_set
        for vx in range(5):
            for vy in range(5):
                q[y, x, vx, vy, track.legal[vx, vy, at_start]] = 0.0
    return q


def route_initialized_q(track: Track) -> np.ndarray:
    """Initialize from shortest *noise-free* routes, ignoring crash transitions.

    This supplies complete episodes to MC on a sparse-reward task. Training still
    uses the required random start, stochastic action failure, and -1 step reward.
    """
    q = new_q(track)
    predecessors: dict[State, list[State]] = defaultdict(list)
    distance: dict[State, int] = {}
    queue: deque[State] = deque()
    states = [
        (x, y, vx, vy)
        for x, y in track.cells - track.finishes
        for vx in range(5) for vy in range(5)
        if (vx, vy) != (0, 0) or (x, y) in track.start_set
    ]
    for state in states:
        x, y, vx, vy = state
        for action in track.legal[vx, vy, (x, y) in track.start_set]:
            ax, ay = ACTION_DELTAS[action]
            next_vx, next_vy = vx + ax, vy + ay
            result = track.motion[x, y, next_vx, next_vy]
            if result.kind == "finish" and state not in distance:
                distance[state] = 1
                queue.append(state)
            elif result.kind == "move":
                successors = (result.x, result.y, next_vx, next_vy)
                predecessors[successors].append(state)
    while queue:
        successor = queue.popleft()
        for predecessor in predecessors[successor]:
            if predecessor not in distance:
                distance[predecessor] = distance[successor] + 1
                queue.append(predecessor)
    for state in states:
        x, y, vx, vy = state
        for action in track.legal[vx, vy, (x, y) in track.start_set]:
            ax, ay = ACTION_DELTAS[action]
            result = track.motion[x, y, vx + ax, vy + ay]
            if result.kind == "finish":
                value = -1
            elif result.kind == "move":
                value = -1 - distance.get((result.x, result.y, vx + ax, vy + ay), 999)
            else:
                value = -1000
            q[y, x, vx, vy, action] = value
    return q


def greedy_action(q: np.ndarray, state: State) -> int:
    x, y, vx, vy = state
    return int(np.argmax(q[y, x, vx, vy]))


def episode(track: Track, q: np.ndarray, rng: random.Random, epsilon: float,
            max_steps: int = 1000, failure_probability: float = 0.1,
            start: tuple[int, int] | None = None,
            record: bool = False):
    x, y = start if start is not None else rng.choice(track.starts)
    state: State = (x, y, 0, 0)
    visits: list[tuple[State, int]] = []
    positions = [(x, y)] if record else None
    crashes = 0
    for step_number in range(1, max_steps + 1):
        x, y, vx, vy = state
        legal = track.legal[vx, vy, (x, y) in track.start_set]
        action = rng.choice(legal) if rng.random() < epsilon else greedy_action(q, state)
        visits.append((state, action))
        next_state, done, crashed = track.step(state, action, rng, failure_probability)
        if crashed:
            crashes += 1
        if positions is not None:
            positions.append(next_state[:2])
        if done:
            return visits, step_number, crashes, True, positions
        state = next_state
    return visits, max_steps, crashes, False, positions


def mc_update(q: np.ndarray, visits: list[tuple[State, int]], alpha: float):
    # Every-visit, undiscounted on-policy Monte Carlo control (gamma = 1).
    # A truncated episode must never be treated as a zero-value terminal state.
    return_value = 0
    for (x, y, vx, vy), action in reversed(visits):
        return_value -= 1
        old = q[y, x, vx, vy, action]
        q[y, x, vx, vy, action] = old + alpha * (return_value - old)

