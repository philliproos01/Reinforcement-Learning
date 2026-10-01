from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

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
    distribution_step: int | None = None

    def validate(self) -> None:
        if self.runs <= 0 or self.steps <= 0 or self.arms <= 0:
            raise ValueError("runs, steps, and arms must all be positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.random_walk_std < 0.0 or self.reward_std < 0.0:
            raise ValueError("standard deviations cannot be negative")
        if self.distribution_step is not None and not (
            1 <= self.distribution_step <= self.steps
        ):
            raise ValueError("distribution_step must be between 1 and steps")


@dataclass(frozen=True)
class ExperimentResult:
    """Per-step averages for both action-value methods."""

    sample_average_reward: FloatArray
    constant_step_reward: FloatArray
    sample_average_optimal: FloatArray
    constant_step_optimal: FloatArray
    distribution_step: int
    representative_true_values: FloatArray


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
    distribution_step = config.steps if config.distribution_step is None else (
        config.distribution_step
    )
    representative_true_values = np.empty(config.arms, dtype=np.float64)

    for step in range(config.steps):
        if step + 1 == distribution_step:
            representative_true_values[:] = true_values[0]

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
        distribution_step=distribution_step,
        representative_true_values=representative_true_values,
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
) -> None:
    """Create the Figure 2.2-style two-panel comparison for display."""

    steps = np.arange(1, config.steps + 1)
    sample_label = r"Sample average: $\alpha_n(a)=1/N_n(a)$"
    constant_label = rf"Constant step size: $\alpha={config.alpha:g}$"
    sample_color = "#FF0000"
    constant_color = "#48FF00"

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
        f"Nonstationary Testbed",
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


def plot_reward_distributions(
    config: ExperimentConfig,
    result: ExperimentResult,
) -> None:
    """Create a reward-distribution plot for display."""

    positions = np.arange(1, config.arms + 1)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)

    for position, mean in zip(positions, result.representative_true_values):
        if config.reward_std > 0.0:
            rewards = np.linspace(
                mean - 4.0 * config.reward_std,
                mean + 4.0 * config.reward_std,
                400,
            )
            standardized = (rewards - mean) / config.reward_std
            half_width = 0.42 * np.exp(-0.5 * standardized**2)
            axis.fill_betweenx(
                rewards,
                position - half_width,
                position + half_width,
                facecolor="#9ECAE1",
                edgecolor="#6BAED6",
                linewidth=0.8,
                alpha=0.8,
            )
        axis.hlines(mean, position - 0.30, position + 0.30)
        axis.text(
            position,
            mean + 0.10,
            rf"$q_t({position})={mean:.2f}$",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis.axhline(0.0, color="#888888", linewidth=1.0, linestyle=(0, (6, 4)))
    axis.set_xlim(0.4, config.arms + 0.6)
    axis.set_xticks(positions)
    axis.set_xlabel("Action", fontweight="bold")
    axis.set_ylabel("Reward distribution", fontweight="bold")
    axis.set_title(
        (
            f"Nonstationary {config.arms}-Armed Testbed at Step "
            f"{result.distribution_step:,}\n"
            rf"$R_t\mid A_t=a \sim \mathcal{{N}}(q_t(a), "
            rf"{config.reward_std:g}^2)$ for one representative run"
        ),
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)


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
        "--distribution-step",
        type=int,
        default=None,
        help="Step shown in the reward-distribution plot (default: final step).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for the CSV and JSON outputs.",
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
        distribution_step=args.distribution_step,
    )
    result = run_experiment(config)
    summary = build_summary(config, result)

    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "nonstationary_bandit_results.csv"
    summary_path = output_dir / "nonstationary_bandit_summary.json"
    plot_results(config, result)
    plot_reward_distributions(config, result)
    save_results_csv(result, csv_path)
    save_summary_json(summary, summary_path)

    print(json.dumps(summary, indent=2))
    print(f"Data: {csv_path}")
    print(f"Summary: {summary_path}")
    plt.show()


if __name__ == "__main__":
    main()
