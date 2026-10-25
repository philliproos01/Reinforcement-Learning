"""Independent model-based benchmark for the learned Monte Carlo policies.

This is analysis only; no dynamic-programming values enter MC training.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np

from racetrack import ACTION_DELTAS, Track, greedy_action, route_initialized_q


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def build_model(track):
    states = sorted(
        (x, y, vx, vy)
        for x, y in track.cells - track.finishes
        for vx in range(5) for vy in range(5)
        if (vx, vy) != (0, 0) or (x, y) in track.start_set
    )
    lookup = {state: i for i, state in enumerate(states)}
    n = len(states)
    # -1 = finish (zero continuation); -2 = crash (uniform random start).
    intended = np.full((n, 9), -1, dtype=np.int32)
    failure = np.full(n, -1, dtype=np.int32)
    legal = np.zeros((n, 9), dtype=bool)

    def encode(x, y, vx, vy):
        motion = track.motion[x, y, vx, vy]
        if motion.kind == "finish":
            return -1
        if motion.kind == "crash":
            return -2
        return lookup[(motion.x, motion.y, vx, vy)]

    for i, (x, y, vx, vy) in enumerate(states):
        failure[i] = encode(x, y, vx, vy)
        for action in track.legal[vx, vy, (x, y) in track.start_set]:
            ax, ay = ACTION_DELTAS[action]
            intended[i, action] = encode(x, y, vx + ax, vy + ay)
            legal[i, action] = True
    start_indices = np.asarray([lookup[(x, y, 0, 0)] for x, y in track.starts])
    return states, intended, failure, legal, start_indices


def continuation(indices, values, restart_value):
    result = np.zeros(indices.shape, dtype=float)
    moving = indices >= 0
    result[moving] = values[indices[moving]]
    result[indices == -2] = restart_value
    return result


def solve(track, q_learned):
    states, intended, failure, legal, starts = build_model(track)
    values = np.zeros(len(states), dtype=float)
    iterations = 0
    for iterations in range(1, 10_001):
        restart = float(np.mean(values[starts]))
        q_values = (1 + 0.9 * continuation(intended, values, restart)
                    + 0.1 * continuation(failure, values, restart)[:, None])
        q_values[~legal] = np.inf
        updated = np.min(q_values, axis=1)
        delta = float(np.max(np.abs(updated - values)))
        values = updated
        if delta < 1e-10:
            break
    else:
        raise RuntimeError("Optimal value iteration did not converge")
    optimal_steps = float(np.mean(values[starts]))
    optimal_actions = np.argmin(q_values, axis=1)

    # Evaluate the fixed greedy MC policy in the same model.
    chosen = np.asarray([greedy_action(q_learned, state) for state in states])
    chosen_intended = intended[np.arange(len(states)), chosen]
    learned_values = np.zeros(len(states), dtype=float)
    for policy_iterations in range(1, 100_001):
        restart = float(np.mean(learned_values[starts]))
        updated = (1 + 0.9 * continuation(chosen_intended, learned_values, restart)
                   + 0.1 * continuation(failure, learned_values, restart))
        delta = float(np.max(np.abs(updated - learned_values)))
        learned_values = updated
        if delta < 1e-10:
            break
    else:
        raise RuntimeError("Learned policy evaluation did not converge")
    learned_steps = float(np.mean(learned_values[starts]))

    baseline = route_initialized_q(track)
    baseline_chosen = np.asarray([greedy_action(baseline, state) for state in states])
    baseline_intended = intended[np.arange(len(states)), baseline_chosen]
    baseline_values = np.zeros(len(states), dtype=float)
    for baseline_iterations in range(1, 100_001):
        restart = float(np.mean(baseline_values[starts]))
        updated = (1 + 0.9 * continuation(baseline_intended, baseline_values, restart)
                   + 0.1 * continuation(failure, baseline_values, restart))
        delta = float(np.max(np.abs(updated - baseline_values)))
        baseline_values = updated
        if delta < 1e-10:
            break
    else:
        raise RuntimeError("Baseline policy evaluation did not converge")

    return {
        "states": len(states),
        "optimal_expected_steps": optimal_steps,
        "route_initializer_expected_steps": float(np.mean(baseline_values[starts])),
        "trained_mc_expected_steps": learned_steps,
        "trained_excess_steps": learned_steps - optimal_steps,
        "trained_vs_initializer_steps": learned_steps - float(np.mean(baseline_values[starts])),
        "optimal_value_iterations": iterations,
        "trained_policy_iterations": policy_iterations,
        "optimal_start_actions": [ACTION_DELTAS[optimal_actions[i]] for i in starts],
        "trained_start_actions": [ACTION_DELTAS[chosen[i]] for i in starts],
        "optimal_start_expected_steps": [float(values[i]) for i in starts],
        "trained_start_expected_steps": [float(learned_values[i]) for i in starts],
    }


def main():
    report = {}
    refined_report = {}
    start_rows = []
    for name in ("left", "right"):
        track = Track(name)
        q = np.load(OUT / f"q_{name}.npy")
        report[name] = solve(track, q)
        print(name, report[name])
        refined_path = OUT / f"q_{name}_refined.npy"
        if refined_path.exists():
            refined_report[name] = solve(track, np.load(refined_path))
            result = refined_report[name]
            for i, start in enumerate(track.starts):
                start_rows.append({
                    "track": name, "start_x": start[0], "start_y": start[1],
                    "optimal_expected_steps": result["optimal_start_expected_steps"][i],
                    "trained_expected_steps": result["trained_start_expected_steps"][i],
                    "optimal_action_x": result["optimal_start_actions"][i][0],
                    "optimal_action_y": result["optimal_start_actions"][i][1],
                    "trained_action_x": result["trained_start_actions"][i][0],
                    "trained_action_y": result["trained_start_actions"][i][1],
                })
    (OUT / "optimality_check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if refined_report:
        (OUT / "refined_optimality_check.json").write_text(
            json.dumps(refined_report, indent=2) + "\n", encoding="utf-8"
        )
        with (OUT / "refined_start_values.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(start_rows[0]))
            writer.writeheader()
            writer.writerows(start_rows)


if __name__ == "__main__":
    main()
