"""Evaluate bandit agents under abrupt best/worst-arm payout swaps.

True action values are stationary between change points. At each change point,
the current best and worst arms exchange their expected rewards independently
in every run. The script compares sample-average and constant-step updates,
several epsilon-greedy and softmax hyperparameters, and decaying exploration.
"""

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
    runs: int = 1_000
    steps: int = 10_000
    arms: int = 10
    alpha: float = 0.1
    change_interval: int = 2_500
    initial_value_std: float = 1.0
    reward_std: float = 1.0
    epsilon_values: tuple[float, ...] = (0.01, 0.1, 0.2)
    temperature_values: tuple[float, ...] = (0.05, 0.1, 0.2)
    epsilon_decay_initial: float = 0.3
    epsilon_decay_final: float = 0.01
    temperature_decay_initial: float = 1.0
    temperature_decay_final: float = 0.05
    recovery_window: int = 250
    seed: int = 2027

    def validate(self) -> None:
        if self.runs <= 0 or self.steps <= 0 or self.arms < 2:
            raise ValueError("runs and steps must be positive; arms must be at least 2")
        if self.change_interval <= 0 or self.recovery_window <= 0:
            raise ValueError("change_interval and recovery_window must be positive")
        if self.change_interval >= self.steps:
            raise ValueError("change_interval must produce at least one change")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.initial_value_std < 0.0 or self.reward_std < 0.0:
            raise ValueError("standard deviations cannot be negative")
        if not self.epsilon_values or not self.temperature_values:
            raise ValueError("epsilon and temperature sweeps cannot be empty")
        if any(not 0.0 <= value <= 1.0 for value in self.epsilon_values):
            raise ValueError("epsilon values must be in [0, 1]")
        if any(value <= 0.0 for value in self.temperature_values):
            raise ValueError("temperatures must be positive")
        if not (
            0.0
            < self.epsilon_decay_final
            <= self.epsilon_decay_initial
            <= 1.0
        ):
            raise ValueError("decaying epsilon endpoints are invalid")
        if not (
            0.0
            < self.temperature_decay_final
            <= self.temperature_decay_initial
        ):
            raise ValueError("decaying temperature endpoints are invalid")


@dataclass(frozen=True)
class AgentSpec:
    identifier: str
    label: str
    policy: str
    update_rule: str
    exploration_initial: float
    exploration_final: float


@dataclass(frozen=True)
class ExperimentResult:
    specs: tuple[AgentSpec, ...]
    rewards: FloatArray
    optimal: FloatArray
    regret: FloatArray
    change_steps: tuple[int, ...]


def number_id(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_agent_specs(config: ExperimentConfig) -> tuple[AgentSpec, ...]:
    specs: list[AgentSpec] = [
        AgentSpec(
            "sample_average_epsilon_0p1",
            r"Sample average, $\epsilon=0.1$",
            "epsilon_greedy",
            "sample_average",
            0.1,
            0.1,
        )
    ]
    for epsilon in config.epsilon_values:
        specs.append(
            AgentSpec(
                f"constant_alpha_epsilon_{number_id(epsilon)}",
                rf"$\epsilon$-greedy, $\epsilon={epsilon:g}$",
                "epsilon_greedy",
                "constant_step",
                epsilon,
                epsilon,
            )
        )
    specs.append(
        AgentSpec(
            "decaying_epsilon",
            rf"Decaying $\epsilon$: {config.epsilon_decay_initial:g}"
            rf"$\to${config.epsilon_decay_final:g}",
            "epsilon_greedy",
            "constant_step",
            config.epsilon_decay_initial,
            config.epsilon_decay_final,
        )
    )
    for temperature in config.temperature_values:
        specs.append(
            AgentSpec(
                f"constant_alpha_softmax_{number_id(temperature)}",
                rf"Softmax, $T={temperature:g}$",
                "softmax",
                "constant_step",
                temperature,
                temperature,
            )
        )
    specs.append(
        AgentSpec(
            "decaying_temperature",
            rf"Decaying $T$: {config.temperature_decay_initial:g}"
            rf"$\to${config.temperature_decay_final:g}",
            "softmax",
            "constant_step",
            config.temperature_decay_initial,
            config.temperature_decay_final,
        )
    )
    return tuple(specs)


def exploration_schedule(spec: AgentSpec, steps: int) -> FloatArray:
    if spec.exploration_initial == spec.exploration_final:
        return np.full(steps, spec.exploration_initial, dtype=np.float64)
    return np.geomspace(
        spec.exploration_initial,
        spec.exploration_final,
        num=steps,
        dtype=np.float64,
    )


def random_argmax(values: FloatArray, rng: np.random.Generator) -> IntArray:
    maxima = values.max(axis=1, keepdims=True)
    is_maximum = values == maxima
    tie_counts = is_maximum.sum(axis=1)
    choices = (rng.random(values.shape[0]) * tie_counts).astype(np.int64)
    return np.argmax(
        np.cumsum(is_maximum, axis=1) > choices[:, None], axis=1
    ).astype(np.int64)


def epsilon_greedy_actions(
    estimates: FloatArray, epsilon: float, rng: np.random.Generator
) -> IntArray:
    actions = random_argmax(estimates, rng)
    explore = rng.random(estimates.shape[0]) < epsilon
    actions[explore] = rng.integers(
        estimates.shape[1], size=int(explore.sum())
    )
    return actions


def softmax_actions(
    estimates: FloatArray, temperature: float, rng: np.random.Generator
) -> IntArray:
    logits = estimates / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    draws = rng.random(estimates.shape[0])
    return (cumulative < draws[:, None]).sum(axis=1).astype(np.int64)


class BatchedAgent:
    def __init__(
        self,
        spec: AgentSpec,
        config: ExperimentConfig,
        selection_rng: np.random.Generator,
        reward_rng: np.random.Generator,
    ) -> None:
        self.spec = spec
        self.alpha = config.alpha
        self.schedule = exploration_schedule(spec, config.steps)
        self.estimates = np.zeros((config.runs, config.arms), dtype=np.float64)
        self.counts = np.zeros((config.runs, config.arms), dtype=np.int64)
        self.selection_rng = selection_rng
        self.reward_rng = reward_rng
        self.rows = np.arange(config.runs)

    def select_actions(self, step: int) -> IntArray:
        exploration = float(self.schedule[step])
        if self.spec.policy == "epsilon_greedy":
            return epsilon_greedy_actions(
                self.estimates, exploration, self.selection_rng
            )
        return softmax_actions(self.estimates, exploration, self.selection_rng)

    def update(self, actions: IntArray, rewards: FloatArray) -> None:
        self.counts[self.rows, actions] += 1
        if self.spec.update_rule == "sample_average":
            step_sizes: float | FloatArray = 1.0 / self.counts[self.rows, actions]
        else:
            step_sizes = self.alpha
        selected = self.estimates[self.rows, actions]
        self.estimates[self.rows, actions] = selected + step_sizes * (
            rewards - selected
        )


def swap_best_and_worst(true_values: FloatArray) -> None:
    """Swap each run's current best and worst expected payouts in place."""

    rows = np.arange(true_values.shape[0])
    best = true_values.argmax(axis=1)
    worst = true_values.argmin(axis=1)
    best_values = true_values[rows, best].copy()
    true_values[rows, best] = true_values[rows, worst]
    true_values[rows, worst] = best_values


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    config.validate()
    specs = build_agent_specs(config)
    seed_children = np.random.SeedSequence(config.seed).spawn(1 + 2 * len(specs))
    environment_rng = np.random.default_rng(seed_children[0])
    agents = [
        BatchedAgent(
            spec,
            config,
            np.random.default_rng(seed_children[1 + 2 * index]),
            np.random.default_rng(seed_children[2 + 2 * index]),
        )
        for index, spec in enumerate(specs)
    ]

    true_values = environment_rng.normal(
        0.0, config.initial_value_std, size=(config.runs, config.arms)
    )
    rewards = np.empty((len(specs), config.steps), dtype=np.float64)
    optimal = np.empty_like(rewards)
    regret = np.empty_like(rewards)
    rows = np.arange(config.runs)
    change_steps = tuple(range(config.change_interval, config.steps, config.change_interval))
    change_step_set = set(change_steps)

    for step_index in range(config.steps):
        step_number = step_index + 1
        if step_number in change_step_set:
            swap_best_and_worst(true_values)

        best_values = true_values.max(axis=1)
        for agent_index, agent in enumerate(agents):
            actions = agent.select_actions(step_index)
            selected_values = true_values[rows, actions]
            observed_rewards = selected_values + agent.reward_rng.normal(
                0.0, config.reward_std, size=config.runs
            )
            rewards[agent_index, step_index] = observed_rewards.mean()
            optimal[agent_index, step_index] = np.mean(selected_values == best_values)
            regret[agent_index, step_index] = np.mean(best_values - selected_values)
            agent.update(actions, observed_rewards)

    return ExperimentResult(specs, rewards, optimal, regret, change_steps)


def mean_metrics(
    result: ExperimentResult, indices: NDArray[np.int64]
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for index, spec in enumerate(result.specs):
        metrics[spec.identifier] = {
            "average_reward": float(result.rewards[index, indices].mean()),
            "optimal_action_percent": float(
                100.0 * result.optimal[index, indices].mean()
            ),
            "dynamic_regret": float(result.regret[index, indices].mean()),
        }
    return metrics


def build_summary(
    config: ExperimentConfig, result: ExperimentResult
) -> dict[str, object]:
    final_window = min(1_000, config.steps)
    recovery_indices = np.concatenate(
        [
            np.arange(
                step - 1,
                min(config.steps, step - 1 + config.recovery_window),
                dtype=np.int64,
            )
            for step in result.change_steps
        ]
    )
    return {
        "configuration": asdict(config),
        "agents": [asdict(spec) for spec in result.specs],
        "change_steps": list(result.change_steps),
        "metrics": {
            "overall": mean_metrics(
                result, np.arange(config.steps, dtype=np.int64)
            ),
            f"first_{config.recovery_window}_steps_after_each_change": mean_metrics(
                result, recovery_indices
            ),
            f"last_{final_window}_steps": mean_metrics(
                result,
                np.arange(config.steps - final_window, config.steps, dtype=np.int64),
            ),
        },
    }


def save_csv(result: ExperimentResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["step"]
    for spec in result.specs:
        header.extend(
            [
                f"{spec.identifier}_reward",
                f"{spec.identifier}_optimal_percent",
                f"{spec.identifier}_dynamic_regret",
            ]
        )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for step in range(result.rewards.shape[1]):
            row: list[float | int] = [step + 1]
            for agent_index in range(len(result.specs)):
                row.extend(
                    [
                        result.rewards[agent_index, step],
                        100.0 * result.optimal[agent_index, step],
                        result.regret[agent_index, step],
                    ]
                )
            writer.writerow(row)


def plot_group(
    config: ExperimentConfig,
    result: ExperimentResult,
    identifiers: list[str],
    title: str,
    output_path: Path,
) -> None:
    index_by_id = {spec.identifier: index for index, spec in enumerate(result.specs)}
    steps = np.arange(1, config.steps + 1)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000"]
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        3, 1, figsize=(10, 10), sharex=True, constrained_layout=True
    )
    for plot_index, identifier in enumerate(identifiers):
        agent_index = index_by_id[identifier]
        spec = result.specs[agent_index]
        color = colors[plot_index % len(colors)]
        axes[0].plot(steps, result.rewards[agent_index], label=spec.label, color=color)
        axes[1].plot(
            steps, 100.0 * result.optimal[agent_index], label=spec.label, color=color
        )
        axes[2].plot(steps, result.regret[agent_index], label=spec.label, color=color)

    axes[0].set_ylabel("Average reward")
    axes[1].set_ylabel("Optimal action (%)")
    axes[1].set_ylim(0.0, 100.0)
    axes[2].set_ylabel("Dynamic regret")
    axes[2].set_xlabel("Steps")
    for axis in axes:
        for change_step in result.change_steps:
            axis.axvline(change_step, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_xlim(1, config.steps)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(True, color="#D9D9D9", linewidth=0.7)
        axis.legend(loc="best", frameon=True)
    figure.suptitle(title, fontsize=15, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    config: ExperimentConfig,
    result: ExperimentResult,
    output_dir: Path,
) -> dict[str, object]:
    summary = build_summary(config, result)
    epsilon_ids = [
        spec.identifier
        for spec in result.specs
        if spec.policy == "epsilon_greedy" and spec.update_rule == "constant_step"
    ]
    softmax_ids = [spec.identifier for spec in result.specs if spec.policy == "softmax"]
    constant_epsilon_id = min(
        epsilon_ids,
        key=lambda identifier: abs(
            next(
                spec.exploration_initial
                for spec in result.specs
                if spec.identifier == identifier
            )
            - 0.1
        ),
    )
    plot_group(
        config,
        result,
        ["sample_average_epsilon_0p1", constant_epsilon_id],
        "Abrupt Swaps: Update-Rule Comparison",
        output_dir / "update_rule_comparison.png",
    )
    plot_group(
        config,
        result,
        epsilon_ids,
        "Abrupt Swaps: Epsilon-Greedy Hyperparameters",
        output_dir / "epsilon_greedy_sweep.png",
    )
    plot_group(
        config,
        result,
        softmax_ids,
        "Abrupt Swaps: Softmax Hyperparameters",
        output_dir / "softmax_sweep.png",
    )
    save_csv(result, output_dir / "abrupt_regime_results.csv")
    with (output_dir / "abrupt_regime_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary


def build_report(config: ExperimentConfig, summary: dict[str, object]) -> str:
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    recovery_key = f"first_{config.recovery_window}_steps_after_each_change"
    recovery = metrics[recovery_key]
    final_key = f"last_{min(1_000, config.steps)}_steps"
    final_metrics = metrics[final_key]
    assert isinstance(recovery, dict) and isinstance(final_metrics, dict)
    best_recovery = min(
        recovery, key=lambda identifier: recovery[identifier]["dynamic_regret"]
    )
    best_final = min(
        final_metrics,
        key=lambda identifier: final_metrics[identifier]["dynamic_regret"],
    )
    labels = {
        spec["identifier"]: spec["label"]
        for spec in summary["agents"]
        if isinstance(spec, dict)
    }
    return f"""# Abrupt Regime-Switching Bandit

The expected rewards are stationary between change points. At steps
{', '.join(str(step) for step in summary['change_steps'])}, the current best
and worst arms exchange payouts independently in every run. Dashed vertical
lines in the plots mark these shocks.

The experiment averages {config.runs:,} runs of {config.steps:,} steps. It
compares sample-average and constant-step updates, three fixed epsilon values,
three fixed temperatures, and decaying epsilon and temperature schedules.

## Plots

![Update comparison](update_rule_comparison.png)

![Epsilon sweep](epsilon_greedy_sweep.png)

![Softmax sweep](softmax_sweep.png)

## Results

The lowest average dynamic regret during the first {config.recovery_window}
steps after each shock belongs to **{labels[best_recovery]}**
({recovery[best_recovery]['dynamic_regret']:.4f}). The lowest regret over the
final {min(1_000, config.steps):,} steps belongs to **{labels[best_final]}**
({final_metrics[best_final]['dynamic_regret']:.4f}).

Abrupt swaps punish methods that give too much weight to old observations.
Sample averages retain the complete pre-change history and therefore usually
reverse their rankings slowly. Constant step sizes forget old evidence
geometrically and should recover faster. Exploration floors matter because the
new best arm was the old worst arm; a nearly greedy agent may rarely revisit
it. Larger epsilon or temperature should improve recovery immediately after a
shock but sacrifice reward during stable intervals. The best setting therefore
depends on both shock frequency and the desired recovery speed.
"""


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1_000)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--arms", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--change-interval", type=int, default=2_500)
    parser.add_argument("--initial-value-std", type=float, default=1.0)
    parser.add_argument("--reward-std", type=float, default=1.0)
    parser.add_argument("--epsilon-values", type=parse_float_list, default=(0.01, 0.1, 0.2))
    parser.add_argument(
        "--temperature-values", type=parse_float_list, default=(0.05, 0.1, 0.2)
    )
    parser.add_argument("--epsilon-decay-initial", type=float, default=0.3)
    parser.add_argument("--epsilon-decay-final", type=float, default=0.01)
    parser.add_argument("--temperature-decay-initial", type=float, default=1.0)
    parser.add_argument("--temperature-decay-final", type=float, default=0.05)
    parser.add_argument("--recovery-window", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = ExperimentConfig(
        runs=args.runs,
        steps=args.steps,
        arms=args.arms,
        alpha=args.alpha,
        change_interval=args.change_interval,
        initial_value_std=args.initial_value_std,
        reward_std=args.reward_std,
        epsilon_values=args.epsilon_values,
        temperature_values=args.temperature_values,
        epsilon_decay_initial=args.epsilon_decay_initial,
        epsilon_decay_final=args.epsilon_decay_final,
        temperature_decay_initial=args.temperature_decay_initial,
        temperature_decay_final=args.temperature_decay_final,
        recovery_window=args.recovery_window,
        seed=args.seed,
    )
    result = run_experiment(config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = write_outputs(config, result, output_dir)
    report_path = output_dir / "experiment_report.md"
    report_path.write_text(build_report(config, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
