"""Reproduce Monte Carlo control experiments and all result figures."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from racetrack import ACTION_DELTAS, Track, episode, mc_update, route_initialized_q


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SEEDS = (11, 23, 37, 53, 71)
EPSILONS = (0.05, 0.20)
EPISODES = 20_000
CHECKPOINT = 1_000
ALPHA = 0.02
EVALUATION_EPISODES = 200
FINAL_EVALUATION_EPISODES = 1_000
MAX_STEPS = 500
T95_4DF = 2.776445


def evaluate(track, q, rng_seed, count):
    rng = random.Random(rng_seed)
    lengths, crashes, successes = [], [], []
    for _ in range(count):
        _, steps, n_crashes, done, _ = episode(
            track, q, rng, 0.0, MAX_STEPS, failure_probability=0.1,
        )
        lengths.append(steps)
        crashes.append(n_crashes)
        successes.append(done)
    return {
        "mean_steps": float(np.mean(lengths)),
        "mean_crashes": float(np.mean(crashes)),
        "success_rate": float(np.mean(successes)),
    }


def all_start_rollouts(track, q):
    rollouts = []
    for start in track.starts:
        visits, steps, crashes, done, positions = episode(
            track, q, random.Random(8000 + start[0]), 0.0, MAX_STEPS,
            failure_probability=0.0, start=start, record=True,
        )
        rollouts.append({"start": start, "visits": visits,
                         "positions": positions, "steps": steps,
                         "crashes": crashes, "finished": done})
    return rollouts


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half_width = float(T95_4DF * np.std(values, ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "ci95_low": mean - half_width,
            "ci95_high": mean + half_width}


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def track_background(ax, track):
    image = np.zeros((track.height, track.width, 3), dtype=float)
    image[:] = (0.13, 0.18, 0.23)
    for x, y in track.cells:
        image[y, x] = (0.92, 0.94, 0.96)
    for x, y in track.starts:
        image[y, x] = (0.34, 0.67, 0.95)
    for x, y in track.finishes:
        image[y, x] = (0.43, 0.81, 0.52)
    ax.imshow(image, origin="lower", interpolation="nearest")
    ax.set_xlim(-0.5, track.width - 0.5)
    ax.set_ylim(-0.5, track.height - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    ax.grid(color="white", alpha=0.15, linewidth=0.3)


def plot_learning_curves(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout="constrained")
    colors = {0.05: "#177ca7", 0.20: "#df8051"}
    for ax, name in zip(axes, ("left", "right")):
        for epsilon in EPSILONS:
            records = [r for r in rows if r["track"] == name and r["epsilon"] == epsilon
                       and r["episodes"] > 0]
            episodes = sorted({r["episodes"] for r in records})
            means, lows, highs = [], [], []
            for n in episodes:
                values = [r["training_mean_steps"] for r in records if r["episodes"] == n]
                ci = mean_ci(values)
                means.append(ci["mean"])
                lows.append(ci["ci95_low"])
                highs.append(ci["ci95_high"])
            ax.plot(episodes, means, color=colors[epsilon], linewidth=2,
                    label=f"epsilon = {epsilon:.2f}")
            ax.fill_between(episodes, lows, highs, color=colors[epsilon], alpha=0.16)
        ax.set_title(f"{name.title()} track")
        ax.set_xlabel("Training episodes")
        ax.set_ylabel("Behavior policy: mean steps to finish")
        ax.grid(alpha=0.24)
        ax.legend(frameon=False)
        ax.set_xlim(0, EPISODES)
    fig.suptitle("Monte Carlo racetrack learning curves\n"
                 "Each point averages the previous 1,000 training episodes; bands are 95% t intervals",
                 fontsize=13)
    fig.savefig(OUT / "learning_curves.png", dpi=180)
    plt.close(fig)


def plot_greedy_curves(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout="constrained")
    colors = {0.05: "#177ca7", 0.20: "#df8051"}
    for ax, name in zip(axes, ("left", "right")):
        for epsilon in EPSILONS:
            records = [r for r in rows if r["track"] == name and r["epsilon"] == epsilon]
            episodes = sorted({r["episodes"] for r in records})
            medians, lows, highs = [], [], []
            for n in episodes:
                values = [r["greedy_mean_steps"] for r in records if r["episodes"] == n]
                lows.append(float(np.quantile(values, 0.25)))
                medians.append(float(np.median(values)))
                highs.append(float(np.quantile(values, 0.75)))
            ax.plot(episodes, medians, color=colors[epsilon], linewidth=2,
                    label=f"epsilon = {epsilon:.2f}")
            ax.fill_between(episodes, lows, highs, color=colors[epsilon], alpha=0.16)
        ax.set_title(f"{name.title()} track")
        ax.set_xlabel("Training episodes")
        ax.set_ylabel("Greedy policy: mean steps to finish")
        ax.grid(alpha=0.24)
        ax.legend(frameon=False)
        ax.set_xlim(0, EPISODES)
    fig.suptitle("Greedy policy evaluation during training\n"
                 "Median and interquartile range across 5 seeds; 10% action failure",
                 fontsize=13)
    fig.savefig(OUT / "greedy_evaluation_curves.png", dpi=180)
    plt.close(fig)


def plot_rollouts(selected, filename="policy_rollouts.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), layout="constrained")
    for ax, name in zip(axes, ("left", "right")):
        track, rollouts = selected[name]
        track_background(ax, track)
        chosen = rollouts if name == "left" else rollouts[::4]
        palette = plt.cm.plasma(np.linspace(0.12, 0.88, len(chosen)))
        for route, color in zip(chosen, palette):
            xy = np.asarray(route["positions"], dtype=float)
            ax.plot(xy[:, 0], xy[:, 1], "-o", color=color, linewidth=1.7,
                    markersize=2.5, label=f"x={route['start'][0]}: {route['steps']} steps")
            ax.scatter(*route["start"], s=35, color=color, edgecolor="black", zorder=5)
        ax.set_title(f"{name.title()} track | no action failures")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    csv_name = "refined_rollouts.csv" if filename.startswith("refined_") else "rollouts.csv"
    fig.suptitle("Example trajectories of the trained greedy policies\n"
                 f"Blue = start, green = finish; all starting states are in {csv_name}",
                 fontsize=13)
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def make_gif(track, rollout, filename="example_run.gif"):
    scale = 16
    top = 50
    width = track.width * scale
    height = track.height * scale + top
    frames = []
    positions = rollout["positions"]
    for index in range(len(positions)):
        frame = Image.new("RGB", (width, height), "#1f2d3a")
        draw = ImageDraw.Draw(frame)
        draw.text((8, 9), f"Trained policy | step {index}/{rollout['steps']}",
                  fill="white")
        for x, y in track.cells:
            xx, yy = x * scale, top + (track.height - 1 - y) * scale
            fill = "#eef1f4"
            if (x, y) in track.start_set:
                fill = "#56acef"
            if (x, y) in track.finishes:
                fill = "#6fcf83"
            draw.rectangle((xx, yy, xx + scale - 1, yy + scale - 1), fill=fill)
        for a, b in zip(positions[:index], positions[1:index + 1]):
            draw.line(((a[0] + 0.5) * scale,
                       top + (track.height - a[1] - 0.5) * scale,
                       (b[0] + 0.5) * scale,
                       top + (track.height - b[1] - 0.5) * scale),
                      fill="#ff8b54", width=3)
        x, y = positions[index]
        cx = (x + 0.5) * scale
        cy = top + (track.height - y - 0.5) * scale
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill="#e33440",
                     outline="white", width=2)
        frames.append(frame)
    frames[0].save(OUT / filename, save_all=True,
                   append_images=frames[1:], duration=350, loop=0)


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    final_rows = []
    selected = {}
    for name in ("left", "right"):
        track = Track(name)
        for epsilon in EPSILONS:
            for seed in SEEDS:
                q = route_initialized_q(track)
                rng = random.Random(seed)
                initial = evaluate(track, q, 100_000 + seed, EVALUATION_EPISODES)
                rows.append({"track": name, "epsilon": epsilon, "seed": seed,
                             "episodes": 0, "training_mean_steps": "",
                             "greedy_mean_steps": initial["mean_steps"],
                             "greedy_success_rate": initial["success_rate"],
                             "greedy_mean_crashes": initial["mean_crashes"]})
                for checkpoint in range(CHECKPOINT, EPISODES + 1, CHECKPOINT):
                    lengths = []
                    for _ in range(CHECKPOINT):
                        visits, steps, _, done, _ = episode(
                            track, q, rng, epsilon, MAX_STEPS,
                            failure_probability=0.1,
                        )
                        if done:
                            mc_update(q, visits, ALPHA)
                        lengths.append(steps)
                    result = evaluate(track, q, 100_000 + seed + checkpoint,
                                      EVALUATION_EPISODES)
                    rows.append({"track": name, "epsilon": epsilon, "seed": seed,
                                 "episodes": checkpoint,
                                 "training_mean_steps": float(np.mean(lengths)),
                                 "greedy_mean_steps": result["mean_steps"],
                                 "greedy_success_rate": result["success_rate"],
                                 "greedy_mean_crashes": result["mean_crashes"]})
                final = evaluate(track, q, 700_000 + seed, FINAL_EVALUATION_EPISODES)
                final_rows.append({"track": name, "epsilon": epsilon, "seed": seed,
                                   **final})
                if epsilon == 0.05 and seed == SEEDS[0]:
                    np.save(OUT / f"q_{name}.npy", q)
                    selected[name] = (track, all_start_rollouts(track, q))
                print(f"{name} epsilon={epsilon:.2f} seed={seed} "
                      f"final mean={final['mean_steps']:.2f} "
                      f"success={final['success_rate']:.3f}", flush=True)

    write_csv(OUT / "learning_curves.csv", rows,
              ["track", "epsilon", "seed", "episodes", "training_mean_steps",
               "greedy_mean_steps", "greedy_success_rate", "greedy_mean_crashes"])
    write_csv(OUT / "final_evaluation.csv", final_rows,
              ["track", "epsilon", "seed", "mean_steps", "mean_crashes", "success_rate"])
    summary = {}
    for name in ("left", "right"):
        summary[name] = {}
        for epsilon in EPSILONS:
            records = [r for r in final_rows if r["track"] == name and r["epsilon"] == epsilon]
            summary[name][str(epsilon)] = {
                "steps": mean_ci([r["mean_steps"] for r in records]),
                "crashes": mean_ci([r["mean_crashes"] for r in records]),
                "success_rate": mean_ci([r["success_rate"] for r in records]),
            }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

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
    write_csv(OUT / "rollouts.csv", rollout_rows,
              ["track", "start_x", "step", "x", "y", "vx", "vy",
               "action_x", "action_y", "next_x", "next_y", "finished"])
    plot_learning_curves(rows)
    plot_greedy_curves(rows)
    plot_rollouts(selected)
    left_track, left_routes = selected["left"]
    make_gif(left_track, left_routes[2])
    for name, (_, routes) in selected.items():
        print(name, "noise-free starts:", len(routes), "finished:",
              sum(r["finished"] for r in routes), "mean steps:",
              round(float(np.mean([r["steps"] for r in routes])), 2),
              "crashes:", sum(r["crashes"] for r in routes), flush=True)
    print("Wrote results to", OUT, flush=True)


if __name__ == "__main__":
    main()
