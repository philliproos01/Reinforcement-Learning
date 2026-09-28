from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters for the nonstationary bandit experiment."""

    runs: int = 2_000
    steps: int = 10_000
    arms: int = 10
    epsilon: float = 0.1
    alpha: float = 0.1
    random_walk_std: float = 0.01
    reward_std: float = 1.0
    seed: int = 2026

    def validate(self) -> None:
        if self.runs <= 0 or self.steps <= 0 or self.arms <= 0:
            raise ValueError("runs, steps, and arms must all be positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.random_walk_std < 0.0 or self.reward_std < 0.0:
            raise ValueError("standard deviations cannot be negative")


@dataclass(frozen=True)
class ExperimentResult:
    """Per-step averages for both action-value methods."""

    sample_average_reward: FloatArray
    constant_step_reward: FloatArray
    sample_average_optimal: FloatArray
    constant_step_optimal: FloatArray


def random_argmax(values: FloatArray, rng: np.random.Generator) -> IntArray:
    """Return a uniformly random maximizing column from each matrix row."""

    maxima = values.max(axis=1, keepdims=True)
    is_maximum = values == maxima
    tie_counts = is_maximum.sum(axis=1)
    tie_choices = (rng.random(values.shape[0]) * tie_counts).astype(np.int64)
    return np.argmax(
        np.cumsum(is_maximum, axis=1) > tie_choices[:, None], axis=1
    ).astype(np.int64)


def epsilon_greedy_actions(
    estimates: FloatArray,
    epsilon: float,
    rng: np.random.Generator,
) -> IntArray:
    """Select one epsilon-greedy action for every independent run."""

    actions = random_argmax(estimates, rng)
    explore = rng.random(estimates.shape[0]) < epsilon
    actions[explore] = rng.integers(
        estimates.shape[1], size=int(explore.sum())
    )
    return actions


def apply_action_value_update(
    estimates: FloatArray,
    counts: IntArray,
    actions: IntArray,
    rewards: FloatArray,
    *,
    alpha: float | None,
) -> None:
    """Apply an in-place sample-average or constant-step-size update.

    ``alpha=None`` selects the incremental sample-average step size 1/N(a).
    Otherwise the supplied constant step size is used.
    """

    rows = np.arange(estimates.shape[0])
    counts[rows, actions] += 1
    step_sizes: float | FloatArray
    if alpha is None:
        step_sizes = 1.0 / counts[rows, actions]
    else:
        step_sizes = alpha
    estimates[rows, actions] += step_sizes * (
        rewards - estimates[rows, actions]
    )


def optimal_action_indicator(
    true_values: FloatArray, actions: IntArray
) -> FloatArray:
    """Indicate whether each action is currently optimal, including ties."""

    rows = np.arange(true_values.shape[0])
    return (
        true_values[rows, actions] == true_values.max(axis=1)
    ).astype(np.float64)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run the paired nonstationary bandit experiment.

    Both methods face the same true-value random walks.  They have separate
    action-selection and reward-noise streams, preventing one method's action
    choices from changing the other method's observations.

    At each step an agent acts against the current true values, receives and
    learns from a reward, and then every true action value takes one random-walk
    step.  On step 1 all actions are correctly treated as optimal because the
    true values are tied at zero.
    """

    config.validate()
    seed_sequence = np.random.SeedSequence(config.seed)
    walk_seed, sample_seed, constant_seed = seed_sequence.spawn(3)
    walk_rng = np.random.default_rng(walk_seed)
    sample_rng = np.random.default_rng(sample_seed)
    constant_rng = np.random.default_rng(constant_seed)

    true_values = np.zeros((config.runs, config.arms), dtype=np.float64)
    sample_estimates = np.zeros_like(true_values)
    constant_estimates = np.zeros_like(true_values)
    sample_counts = np.zeros((config.runs, config.arms), dtype=np.int64)
    constant_counts = np.zeros_like(sample_counts)
    rows = np.arange(config.runs)

    sample_average_reward = np.empty(config.steps, dtype=np.float64)
    constant_step_reward = np.empty(config.steps, dtype=np.float64)
    sample_average_optimal = np.empty(config.steps, dtype=np.float64)
    constant_step_optimal = np.empty(config.steps, dtype=np.float64)

    for step in range(config.steps):
        sample_actions = epsilon_greedy_actions(
            sample_estimates, config.epsilon, sample_rng
        )
        constant_actions = epsilon_greedy_actions(
            constant_estimates, config.epsilon, constant_rng
        )

        sample_average_optimal[step] = optimal_action_indicator(
            true_values, sample_actions
        ).mean()
        constant_step_optimal[step] = optimal_action_indicator(
            true_values, constant_actions
        ).mean()

        sample_rewards = true_values[rows, sample_actions] + sample_rng.normal(
            0.0, config.reward_std, size=config.runs
        )
        constant_rewards = true_values[rows, constant_actions] + constant_rng.normal(
            0.0, config.reward_std, size=config.runs
        )
        sample_average_reward[step] = sample_rewards.mean()
        constant_step_reward[step] = constant_rewards.mean()

        apply_action_value_update(
            sample_estimates,
            sample_counts,
            sample_actions,
            sample_rewards,
            alpha=None,
        )
        apply_action_value_update(
            constant_estimates,
            constant_counts,
            constant_actions,
            constant_rewards,
            alpha=config.alpha,
        )

        true_values += walk_rng.normal(
            0.0,
            config.random_walk_std,
            size=true_values.shape,
        )

    return ExperimentResult(
        sample_average_reward=sample_average_reward,
        constant_step_reward=constant_step_reward,
        sample_average_optimal=sample_average_optimal,
        constant_step_optimal=constant_step_optimal,
    )


def save_results_csv(result: ExperimentResult, output_path: Path) -> None:
    """Save all per-step averages in a portable CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "sample_average_reward",
                "constant_step_reward",
                "sample_average_optimal_percent",
                "constant_step_optimal_percent",
            ]
        )
        for index in range(result.sample_average_reward.size):
            writer.writerow(
                [
                    index + 1,
                    result.sample_average_reward[index],
                    result.constant_step_reward[index],
                    100.0 * result.sample_average_optimal[index],
                    100.0 * result.constant_step_optimal[index],
                ]
            )


def trailing_mean(values: FloatArray, window: int) -> float:
    """Return the mean over the final ``window`` entries."""

    return float(values[-min(window, values.size) :].mean())


def build_summary(
    config: ExperimentConfig, result: ExperimentResult
) -> dict[str, object]:
    """Build a compact, machine-readable experiment summary."""

    window = min(1_000, config.steps)
    sample_reward = trailing_mean(result.sample_average_reward, window)
    constant_reward = trailing_mean(result.constant_step_reward, window)
    sample_optimal = 100.0 * trailing_mean(
        result.sample_average_optimal, window
    )
    constant_optimal = 100.0 * trailing_mean(
        result.constant_step_optimal, window
    )
    return {
        "configuration": asdict(config),
        "metric_window": f"last {window} steps",
        "sample_average": {
            "average_reward": sample_reward,
            "optimal_action_percent": sample_optimal,
        },
        "constant_step_size": {
            "average_reward": constant_reward,
            "optimal_action_percent": constant_optimal,
        },
        "constant_minus_sample": {
            "average_reward": constant_reward - sample_reward,
            "optimal_action_percentage_points": constant_optimal
            - sample_optimal,
        },
    }


def save_summary_json(summary: dict[str, object], output_path: Path) -> None:
    """Save the configuration and headline metrics as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")


def plot_results(
    config: ExperimentConfig,
    result: ExperimentResult,
    output_path: Path,
) -> None:
    """Create the Figure 2.2-style two-panel comparison."""

    steps = np.arange(1, config.steps + 1)
    sample_label = r"Sample average: $\alpha_n(a)=1/N_n(a)$"
    constant_label = rf"Constant step size: $\alpha={config.alpha:g}$"
    sample_color = "#D55E00"
    constant_color = "#0072B2"

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        steps,
        result.sample_average_reward,
        color=sample_color,
        linewidth=1.0,
        label=sample_label,
    )
    axes[0].plot(
        steps,
        result.constant_step_reward,
        color=constant_color,
        linewidth=1.0,
        label=constant_label,
    )
    axes[0].set_ylabel("Average reward")
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].plot(
        steps,
        100.0 * result.sample_average_optimal,
        color=sample_color,
        linewidth=1.0,
        label=sample_label,
    )
    axes[1].plot(
        steps,
        100.0 * result.constant_step_optimal,
        color=constant_color,
        linewidth=1.0,
        label=constant_label,
    )
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Optimal action (%)")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].legend(loc="lower right", frameon=True)

    for axis in axes:
        axis.set_xlim(1, config.steps)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(True, color="#D9D9D9", linewidth=0.7)
        axis.set_axisbelow(True)

    figure.suptitle(
        "Nonstationary 10-Armed Testbed",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.965,
        (
            rf"$\epsilon={config.epsilon:g}$, {config.runs:,} runs, "
            rf"random-walk $\sigma={config.random_walk_std:g}$"
        ),
        ha="center",
        va="top",
        fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2_000)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--arms", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--random-walk-std", type=float, default=0.01)
    parser.add_argument("--reward-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for the PNG, CSV, and JSON outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = ExperimentConfig(
        runs=args.runs,
        steps=args.steps,
        arms=args.arms,
        epsilon=args.epsilon,
        alpha=args.alpha,
        random_walk_std=args.random_walk_std,
        reward_std=args.reward_std,
        seed=args.seed,
    )
    result = run_experiment(config)
    summary = build_summary(config, result)

    output_dir = args.output_dir.resolve()
    plot_path = output_dir / "nonstationary_bandit_results.png"
    csv_path = output_dir / "nonstationary_bandit_results.csv"
    summary_path = output_dir / "nonstationary_bandit_summary.json"
    plot_results(config, result, plot_path)
    save_results_csv(result, csv_path)
    save_summary_json(summary, summary_path)

    print(json.dumps(summary, indent=2))
    print(f"Plot: {plot_path}")
    print(f"Data: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
