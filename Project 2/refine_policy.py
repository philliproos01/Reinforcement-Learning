"""Longer low-exploration MC training for near-optimal example policies."""

from __future__ import annotations

import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_optimality import solve
from racetrack import ACTION_DELTAS, Track, episode, mc_update, route_initialized_q
from run_experiments import (OUT, all_start_rollouts, evaluate, make_gif,
                             plot_rollouts, write_csv)


EPISODES = 300_000
CHECKPOINT = 10_000
EPSILON = 0.02
ALPHA = 0.005
SEED = 11


def main():
    OUT.mkdir(exist_ok=True)
    selected = {}
    curve_rows = []
    report = {}
    for name in ("left", "right"):
        track = Track(name)
        q = route_initialized_q(track)
        rng = random.Random(SEED)
        baseline = evaluate(track, q, 500_000, 500)
        curve_rows.append({"track": name, "episodes": 0,
                           "training_mean_steps": "",
                           "greedy_mean_steps": baseline["mean_steps"]})
        for checkpoint in range(CHECKPOINT, EPISODES + 1, CHECKPOINT):
            lengths = []
            for _ in range(CHECKPOINT):
                visits, steps, _, done, _ = episode(
                    track, q, rng, EPSILON, 500, failure_probability=0.1,
                )
                if done:
                    mc_update(q, visits, ALPHA)
                lengths.append(steps)
            evaluated = evaluate(track, q, 500_000 + checkpoint, 500)
            curve_rows.append({"track": name, "episodes": checkpoint,
                               "training_mean_steps": float(np.mean(lengths)),
                               "greedy_mean_steps": evaluated["mean_steps"]})
            if checkpoint % 50_000 == 0:
                print(name, checkpoint, "greedy mean", evaluated["mean_steps"], flush=True)
        np.save(OUT / f"q_{name}_refined.npy", q)
        selected[name] = (track, all_start_rollouts(track, q))
        report[name] = solve(track, q)
        report[name]["simulation_1000"] = evaluate(track, q, 900_000, 1_000)
        report[name]["noise_free_start_steps"] = [
            route["steps"] for route in selected[name][1]
        ]
        report[name]["noise_free_all_finished"] = all(
            route["finished"] and route["crashes"] == 0
            for route in selected[name][1]
        )
        print(name, "exact expected", report[name]["trained_mc_expected_steps"],
              "optimal", report[name]["optimal_expected_steps"], flush=True)

    write_csv(OUT / "refinement_curve.csv", curve_rows,
              ["track", "episodes", "training_mean_steps", "greedy_mean_steps"])
    (OUT / "refined_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    rollout_rows = []
    for name, (_, rollouts) in selected.items():
        for route in rollouts:
            for i, ((state, action), next_position) in enumerate(
                    zip(route["visits"], route["positions"][1:]), start=1):
                ax, ay = ACTION_DELTAS[action]
                rollout_rows.append({"track": name, "start_x": route["start"][0],
                                     "step": i, "x": state[0], "y": state[1],
                                     "vx": state[2], "vy": state[3],
                                     "action_x": ax, "action_y": ay,
                                     "next_x": next_position[0],
                                     "next_y": next_position[1],
                                     "finished": route["finished"] and i == route["steps"]})
    write_csv(OUT / "refined_rollouts.csv", rollout_rows,
              ["track", "start_x", "step", "x", "y", "vx", "vy",
               "action_x", "action_y", "next_x", "next_y", "finished"])
    plot_rollouts(selected, "refined_policy_rollouts.png")
    make_gif(selected["left"][0], selected["left"][1][2],
             "refined_example_run.gif")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout="constrained")
    for ax, name in zip(axes, ("left", "right")):
        rows = [r for r in curve_rows if r["track"] == name]
        ax.plot([r["episodes"] for r in rows],
                [r["greedy_mean_steps"] for r in rows],
                color="#177ca7", label="MC greedy policy")
        ax.axhline(report[name]["optimal_expected_steps"], color="#d67443",
                   linestyle="--", label="Model based optimum")
        ax.set_title(f"{name.title()} track")
        ax.set_xlabel("Training episodes")
        ax.set_ylabel("Mean steps to finish")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Longer Monte Carlo training at epsilon = 0.02\n"
                 "Greedy evaluation includes 10% action failure",
                 fontsize=13)
    fig.savefig(OUT / "refinement_curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
