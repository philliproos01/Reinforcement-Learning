"""Compare decaying epsilon-greedy and decaying softmax bandit agents."""

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
    """Configuration for the paired nonstationary-bandit comparison."""

    runs: int = 2_000
    steps: int = 10_000
    arms: int = 10
    alpha: float = 0.1
    epsilon_initial: float = 0.3
    epsilon_final: float = 0.01
    temperature_initial: float = 1.0
    temperature_final: float = 0.05
    random_walk_std: float = 0.01
    reward_std: float = 1.0
    seed: int = 2026

    def validate(self) -> None:
        if self.runs <= 0 or self.steps <= 0 or self.arms <= 0:
            raise ValueError("runs, steps, and arms must all be positive")

        numeric_values = (
            self.alpha,
            self.epsilon_initial,
            self.epsilon_final,
            self.temperature_initial,
            self.temperature_final,
            self.random_walk_std,
            self.reward_std,
        )
        if not all(np.isfinite(value) for value in numeric_values):
            raise ValueError("all numeric configuration values must be finite")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= self.epsilon_final <= self.epsilon_initial <= 1.0:
            raise ValueError(
                "epsilon values must satisfy 0 <= final <= initial <= 1"
            )
        if not 0.0 < self.temperature_final <= self.temperature_initial:
            raise ValueError(
                "temperature values must satisfy 0 < final <= initial"
            )
        if self.random_walk_std < 0.0 or self.reward_std < 0.0:
            raise ValueError("standard deviations cannot be negative")


@dataclass(frozen=True)
class ExperimentResult:
    """Per-step schedules and averages across all independent runs."""

    epsilon_schedule: FloatArray
    temperature_schedule: FloatArray
    epsilon_greedy_reward: FloatArray
    softmax_reward: FloatArray
    epsilon_greedy_optimal: FloatArray
    softmax_optimal: FloatArray


def geometric_decay(initial: float, final: float, steps: int) -> FloatArray:
    """Return a geometric schedule that includes both requested endpoints."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if initial <= 0.0 or final <= 0.0:
        raise ValueError("geometric-decay endpoints must be positive")
    if steps == 1:
        return np.array([initial], dtype=np.float64)
    return np.geomspace(initial, final, num=steps, dtype=np.float64)


def epsilon_decay(initial: float, final: float, steps: int) -> FloatArray:
    """Return an epsilon schedule, supporting a final value of exactly zero."""

    if final > 0.0:
        return geometric_decay(initial, final, steps)
    if steps == 1:
        return np.array([initial], dtype=np.float64)
    # A linear schedule is well-defined when geometric decay cannot reach zero.
    return np.linspace(initial, final, num=steps, dtype=np.float64)


def random_argmax(values: FloatArray, rng: np.random.Generator) -> IntArray:
    """Choose uniformly among the maximizing columns of every row."""

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


def softmax_actions(
    estimates: FloatArray,
    temperature: float,
    rng: np.random.Generator,
) -> IntArray:
    """Sample actions from a numerically stable Boltzmann distribution."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    logits = estimates / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    uniforms = rng.random(estimates.shape[0])
    return (cumulative < uniforms[:, None]).sum(axis=1).astype(np.int64)


class MABAgent:
    """A batch of independent MAB agents sharing one policy configuration."""

    def __init__(
        self,
        runs: int,
        arms: int,
        alpha: float,
        rng: np.random.Generator,
    ) -> None:
        self.estimates = np.zeros((runs, arms), dtype=np.float64)
        self.alpha = alpha
        self.rng = rng
        self._rows = np.arange(runs)

    def select_actions(self, step: int) -> IntArray:
        raise NotImplementedError

    def update(self, actions: IntArray, rewards: FloatArray) -> None:
        """Apply a constant-step-size action-value update in place."""

        selected = self.estimates[self._rows, actions]
        self.estimates[self._rows, actions] = selected + self.alpha * (
            rewards - selected
        )


class DecayingEpsilonGreedyAgent(MABAgent):
    """Epsilon-greedy agent whose epsilon decays over time."""

    def __init__(
        self,
        runs: int,
        arms: int,
        alpha: float,
        schedule: FloatArray,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(runs, arms, alpha, rng)
        self.schedule = schedule

    def select_actions(self, step: int) -> IntArray:
        return epsilon_greedy_actions(
            self.estimates, float(self.schedule[step]), self.rng
        )


class DecayingSoftmaxAgent(MABAgent):
    """Softmax agent whose temperature decays over time."""

    def __init__(
        self,
        runs: int,
        arms: int,
        alpha: float,
        schedule: FloatArray,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(runs, arms, alpha, rng)
        self.schedule = schedule

    def select_actions(self, step: int) -> IntArray:
        return softmax_actions(
            self.estimates, float(self.schedule[step]), self.rng
        )


def optimal_action_indicator(
    true_values: FloatArray, actions: IntArray
) -> FloatArray:
    """Indicate whether each selected action is currently optimal."""

    rows = np.arange(true_values.shape[0])
    return (
        true_values[rows, actions] == true_values.max(axis=1)
    ).astype(np.float64)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a paired comparison on shared nonstationary bandit instances."""

    config.validate()
    epsilon_schedule = epsilon_decay(
        config.epsilon_initial, config.epsilon_final, config.steps
    )
    temperature_schedule = geometric_decay(
        config.temperature_initial, config.temperature_final, config.steps
    )

    seed_sequence = np.random.SeedSequence(config.seed)
    walk_seed, epsilon_seed, softmax_seed = seed_sequence.spawn(3)
    walk_rng = np.random.default_rng(walk_seed)
    epsilon_agent = DecayingEpsilonGreedyAgent(
        config.runs,
        config.arms,
        config.alpha,
        epsilon_schedule,
        np.random.default_rng(epsilon_seed),
    )
    softmax_agent = DecayingSoftmaxAgent(
        config.runs,
        config.arms,
        config.alpha,
        temperature_schedule,
        np.random.default_rng(softmax_seed),
    )

    true_values = np.zeros((config.runs, config.arms), dtype=np.float64)
    epsilon_reward = np.empty(config.steps, dtype=np.float64)
    softmax_reward = np.empty(config.steps, dtype=np.float64)
    epsilon_optimal = np.empty(config.steps, dtype=np.float64)
    softmax_optimal = np.empty(config.steps, dtype=np.float64)
    rows = np.arange(config.runs)

    for step in range(config.steps):
        epsilon_actions = epsilon_agent.select_actions(step)
        softmax_selected_actions = softmax_agent.select_actions(step)

        epsilon_optimal[step] = optimal_action_indicator(
            true_values, epsilon_actions
        ).mean()
        softmax_optimal[step] = optimal_action_indicator(
            true_values, softmax_selected_actions
        ).mean()

        epsilon_rewards = true_values[rows, epsilon_actions] + (
            epsilon_agent.rng.normal(0.0, config.reward_std, config.runs)
        )
        softmax_rewards = true_values[rows, softmax_selected_actions] + (
            softmax_agent.rng.normal(0.0, config.reward_std, config.runs)
        )
        epsilon_reward[step] = epsilon_rewards.mean()
        softmax_reward[step] = softmax_rewards.mean()

        epsilon_agent.update(epsilon_actions, epsilon_rewards)
        softmax_agent.update(softmax_selected_actions, softmax_rewards)

        # Match Part 1: values walk between decisions, after the current reward.
        true_values += walk_rng.normal(
            0.0, config.random_walk_std, size=true_values.shape
        )

    return ExperimentResult(
        epsilon_schedule=epsilon_schedule,
        temperature_schedule=temperature_schedule,
        epsilon_greedy_reward=epsilon_reward,
        softmax_reward=softmax_reward,
        epsilon_greedy_optimal=epsilon_optimal,
        softmax_optimal=softmax_optimal,
    )


def metric_window(
    result: ExperimentResult, start: int, stop: int
) -> dict[str, dict[str, float]]:
    """Summarize reward and optimal-action rate over a half-open interval."""

    epsilon_reward = float(result.epsilon_greedy_reward[start:stop].mean())
    softmax_reward = float(result.softmax_reward[start:stop].mean())
    epsilon_optimal = float(
        100.0 * result.epsilon_greedy_optimal[start:stop].mean()
    )
    softmax_optimal = float(100.0 * result.softmax_optimal[start:stop].mean())
    return {
        "epsilon_greedy": {
            "average_reward": epsilon_reward,
            "optimal_action_percent": epsilon_optimal,
        },
        "softmax": {
            "average_reward": softmax_reward,
            "optimal_action_percent": softmax_optimal,
        },
        "softmax_minus_epsilon_greedy": {
            "average_reward": softmax_reward - epsilon_reward,
            "optimal_action_percentage_points": softmax_optimal
            - epsilon_optimal,
        },
    }


def build_summary(
    config: ExperimentConfig, result: ExperimentResult
) -> dict[str, object]:
    """Build summaries for early, middle, and late portions of the run."""

    window = min(1_000, config.steps)
    middle_start = max(0, (config.steps - window) // 2)
    return {
        "configuration": asdict(config),
        "decay": {
            "epsilon": "geometric" if config.epsilon_final > 0.0 else "linear",
            "temperature": "geometric",
        },
        "windows": {
            f"first_{window}_steps": metric_window(result, 0, window),
            f"middle_{window}_steps": metric_window(
                result, middle_start, middle_start + window
            ),
            f"last_{window}_steps": metric_window(
                result, config.steps - window, config.steps
            ),
        },
    }


def save_results_csv(result: ExperimentResult, output_path: Path) -> None:
    """Save schedules and per-step averages as CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "epsilon",
                "temperature",
                "epsilon_greedy_reward",
                "softmax_reward",
                "epsilon_greedy_optimal_percent",
                "softmax_optimal_percent",
            ]
        )
        for index in range(result.epsilon_greedy_reward.size):
            writer.writerow(
                [
                    index + 1,
                    result.epsilon_schedule[index],
                    result.temperature_schedule[index],
                    result.epsilon_greedy_reward[index],
                    result.softmax_reward[index],
                    100.0 * result.epsilon_greedy_optimal[index],
                    100.0 * result.softmax_optimal[index],
                ]
            )


def save_json(data: dict[str, object], output_path: Path) -> None:
    """Save a dictionary as indented JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def plot_performance(
    config: ExperimentConfig,
    result: ExperimentResult,
) -> None:
    """Plot reward and optimal-action rate for both action selectors."""

    steps = np.arange(1, config.steps + 1)
    epsilon_label = (
        rf"Decaying $\epsilon$-greedy "
        rf"(${config.epsilon_initial:g}\to{config.epsilon_final:g}$)"
    )
    softmax_label = (
        rf"Decaying softmax $T$ "
        rf"(${config.temperature_initial:g}\to{config.temperature_final:g}$)"
    )
    epsilon_color = "#D55E00"
    softmax_color = "#0072B2"

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        steps,
        result.epsilon_greedy_reward,
        color=epsilon_color,
        linewidth=1.0,
        label=epsilon_label,
    )
    axes[0].plot(
        steps,
        result.softmax_reward,
        color=softmax_color,
        linewidth=1.0,
        label=softmax_label,
    )
    axes[0].set_ylabel("Average reward")
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].plot(
        steps,
        100.0 * result.epsilon_greedy_optimal,
        color=epsilon_color,
        linewidth=1.0,
        label=epsilon_label,
    )
    axes[1].plot(
        steps,
        100.0 * result.softmax_optimal,
        color=softmax_color,
        linewidth=1.0,
        label=softmax_label,
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
        f"Action Selection on a Nonstationary {config.arms}-Armed Bandit",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.965,
        (
            rf"$\alpha={config.alpha:g}$, {config.runs:,} runs, "
            rf"random-walk $\sigma={config.random_walk_std:g}$"
        ),
        ha="center",
        va="top",
        fontsize=10,
    )
def plot_schedules(
    config: ExperimentConfig,
    result: ExperimentResult,
) -> None:
    """Plot the two decay schedules on separate, correctly scaled axes."""

    steps = np.arange(1, config.steps + 1)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, constrained_layout=True
    )
    axes[0].plot(steps, result.epsilon_schedule, color="#D55E00", linewidth=2)
    axes[0].set_ylabel(r"Exploration rate $\epsilon$")
    axes[0].set_title(r"$\epsilon$-greedy schedule")
    axes[1].plot(
        steps, result.temperature_schedule, color="#0072B2", linewidth=2
    )
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel(r"Temperature $T$")
    axes[1].set_title("Softmax schedule")
    for axis in axes:
        axis.set_xlim(1, config.steps)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(True, color="#D9D9D9", linewidth=0.7)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Geometric Exploration Decay",
        fontsize=15,
        fontweight="bold",
    )
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2_000)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--arms", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon-initial", type=float, default=0.3)
    parser.add_argument("--epsilon-final", type=float, default=0.01)
    parser.add_argument("--temperature-initial", type=float, default=1.0)
    parser.add_argument("--temperature-final", type=float, default=0.05)
    parser.add_argument("--random-walk-std", type=float, default=0.01)
    parser.add_argument("--reward-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for the CSV data and JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = ExperimentConfig(
        runs=args.runs,
        steps=args.steps,
        arms=args.arms,
        alpha=args.alpha,
        epsilon_initial=args.epsilon_initial,
        epsilon_final=args.epsilon_final,
        temperature_initial=args.temperature_initial,
        temperature_final=args.temperature_final,
        random_walk_std=args.random_walk_std,
        reward_std=args.reward_std,
        seed=args.seed,
    )
    result = run_experiment(config)
    summary = build_summary(config, result)

    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "action_selection_results.csv"
    summary_path = output_dir / "action_selection_summary.json"

    plot_performance(config, result)
    plot_schedules(config, result)
    save_results_csv(result, csv_path)
    save_json(summary, summary_path)

    print(json.dumps(summary, indent=2))
    print(f"Data: {csv_path}")
    print(f"Summary: {summary_path}")
    plt.show()


if __name__ == "__main__":
    main()
