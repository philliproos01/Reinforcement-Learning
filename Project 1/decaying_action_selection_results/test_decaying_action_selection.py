"""Focused tests for the decaying action-selection experiment."""

from __future__ import annotations

import unittest

import numpy as np

from decaying_action_selection import (
    ExperimentConfig,
    MABAgent,
    epsilon_decay,
    run_experiment,
    softmax_actions,
)


class ScheduleTests(unittest.TestCase):
    def test_positive_epsilon_decay_includes_endpoints(self) -> None:
        schedule = epsilon_decay(0.3, 0.01, 101)
        self.assertAlmostEqual(schedule[0], 0.3)
        self.assertAlmostEqual(schedule[-1], 0.01)
        self.assertTrue(np.all(np.diff(schedule) < 0.0))

    def test_zero_epsilon_endpoint_uses_linear_decay(self) -> None:
        schedule = epsilon_decay(0.3, 0.0, 4)
        np.testing.assert_allclose(schedule, [0.3, 0.2, 0.1, 0.0])


class AgentTests(unittest.TestCase):
    def test_constant_step_update(self) -> None:
        agent = MABAgent(1, 2, 0.1, np.random.default_rng(1))
        actions = np.array([1], dtype=np.int64)
        agent.update(actions, np.array([2.0], dtype=np.float64))
        self.assertAlmostEqual(agent.estimates[0, 1], 0.2)

    def test_equal_softmax_values_are_approximately_uniform(self) -> None:
        estimates = np.zeros((50_000, 5), dtype=np.float64)
        actions = softmax_actions(estimates, 0.5, np.random.default_rng(2))
        frequencies = np.bincount(actions, minlength=5) / actions.size
        np.testing.assert_allclose(frequencies, np.full(5, 0.2), atol=0.01)

    def test_experiment_is_reproducible_and_bounded(self) -> None:
        config = ExperimentConfig(runs=64, steps=50, seed=7)
        first = run_experiment(config)
        second = run_experiment(config)
        for field in first.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(first, field), getattr(second, field)
            )
        self.assertTrue(np.all((first.softmax_optimal >= 0.0)))
        self.assertTrue(np.all((first.softmax_optimal <= 1.0)))
        self.assertEqual(first.softmax_optimal[0], 1.0)
        self.assertEqual(first.epsilon_greedy_optimal[0], 1.0)


if __name__ == "__main__":
    unittest.main()
