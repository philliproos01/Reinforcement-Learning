# Exercise 5.12: Racetrack

This project implements both tracks in Figure 5.5 of *Reinforcement Learning: An
Introduction* (2020 PDF, printed pages 111–112). The grid cells, starting lines,
and finish lines were extracted from the vector drawing on PDF page 134 and saved
in `tracks.json`. Training does not need the PDF.

## Rules implemented

- State: grid cell `(x, y)` and integer velocity `(vx, vy)`, with `y` increasing
  upward. Both velocity components are in `0..4`. Both can be zero only on the
  starting line.
- Action: one of the nine `(ax, ay)` pairs from `{-1, 0, +1}²`, subject to the
  velocity limits. Independently on each step, the intended increment is
  replaced by `(0, 0)` with probability `0.1`.
- The car moves along the full projected line segment. A swept-grid traversal
  checks every touched cell, including both side cells at an exact grid-corner
  crossing. Touching a finish cell ends the episode; touching another boundary
  sends the car to a uniformly random start cell with zero velocity.
- Every action gives reward `-1`; a crash does not end the episode. Episodes
  start from a uniformly random starting-line cell with zero velocity.

`racetrack.py` contains the simulator and **every-visit, on-policy Monte Carlo
control** (`gamma = 1`). The behavior is epsilon-greedy; the reported policy is
greedy. The value update uses a constant step size. Episodes capped at 500 steps
are excluded from MC updates, so truncation is never mistaken for termination.

Monte Carlo cannot update from an episode that has never reached the finish.
A cold-start pilot produced no finishes in 5,000 capped episodes. To provide
complete episodes, `route_initialized_q` supplies action values from shortest
*noise-free* routes on the known track. This is an initialization, not the
reported policy: all subsequent control updates use sampled returns under
the required 10% action failure. The initializer and the later model-based
benchmark are disclosed because they use transition knowledge.

## Experiments

The main comparison trains five independent seeds (`11, 23, 37, 53, 71`) for
20,000 episodes on each track. The conditions differ only in epsilon (`0.05`
versus `0.20`); both use step size `0.02`. Every 1,000 episodes, the current
greedy policy is evaluated with action failure enabled. The learning curves
show the mean training episode length over the preceding 1,000 episodes with
95% t intervals across seeds. A second plot shows median greedy evaluation
length with interquartile bands; this is robust to occasional temporary
policies that cycle until action noise lets them escape.

Final greedy evaluation uses 1,000 noisy episodes per seed. Mean steps and 95%
t intervals across the five seed means:

| Track | epsilon = 0.05 | epsilon = 0.20 |
| --- | ---: | ---: |
| Left | 14.48 [14.02, 14.94] | 16.04 [14.65, 17.43] |
| Right | 9.37 [9.28, 9.46] | 10.44 [10.17, 10.71] |

Every final evaluation episode reached the finish within the 500-step cap.
The lower exploration rate produced shorter greedy runs on both tracks. On
the left track its training behavior also improved from about 21 steps in
the first 1,000 episodes to about 17 in the last 1,000.

For a more complete policy, `refine_policy.py` trains 300,000 episodes per
track from the same route initialization with epsilon `0.02`, step size
`0.005`, and seed `11`. An independent value-iteration calculation using the
exact simulator transition probabilities provides the optimal expected steps
from a uniformly sampled starting cell:

| Track | Route initializer | Refined MC policy | Model-based optimum | MC gap |
| --- | ---: | ---: | ---: | ---: |
| Left | 15.18 | 12.65 | 12.11 | 0.55 steps |
| Right | 9.65 | 9.02 | 8.95 | 0.07 steps |

The refined policy is **near optimal**, not exactly optimal. With action
failure turned off, it finishes without a crash from all 6 left-track and
all 23 right-track starting cells. The left runs take 11–12 steps and the
right runs take 8–10 steps. `refined_start_values.csv` reports the expected
steps and first action for each start under both the refined and exact optimal
policies.

## Results to inspect

- `results/learning_curves.png`: training episode lengths for the two epsilon
  conditions on both tracks.
- `results/greedy_evaluation_curves.png`: policy quality during those runs.
- `results/refinement_curve.png`: longer MC training against the exact optimum.
- `results/refined_policy_rollouts.png`: example noise-free routes from several
  starts on both tracks.
- `results/refined_example_run.gif`: animated left-track example run.
- `results/refined_rollouts.csv`: every action and position for **all** starts.
- `results/refined_start_values.csv`: per-start optimality check.
- `results/q_left_refined.npy`, `results/q_right_refined.npy`: trained action
  values. The greedy policy is `argmax` over the nine actions in `racetrack.py`.
- Raw experiment data and summaries are in the other CSV and JSON files in
  `results/`.

## Reproduce

From this folder, using Python 3.11 or newer:

```powershell
python -m pip install -r requirements.txt
python -m unittest -v test_racetrack.py
python run_experiments.py
python refine_policy.py
python analyze_optimality.py
```

To regenerate the track asset from the PDF in the parent `RL` folder, run
`python extract_tracks.py` first. The experiment scripts overwrite their
corresponding result files with the same seeded runs.
