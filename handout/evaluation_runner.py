"""HMM localisation assignment — evaluation runner.

AUTHORS: 
    1. Prince Samuel Kyeremanteng
    2. Cecilia Quinn

Runs the three evaluation comparisons requested in HMM_assignment2026.pdf:
  1) 4x4 grid: Forward Filtering with NUF vs UF sensor model
  2) 20x20 grid: Forward Filtering with NUF vs UF sensor model
  3) 10x20 grid (NUF): Filtering vs fixed-lag smoothing (lag=5) vs sensor-only baseline

Outputs plots of cumulative average Manhattan distance over time.

How to run (from root of the repo):
  python handout/evaluation_runner.py
  or, you may use python3 if you're on a mac or linux:
    python3 handout/evaluation_runner.py

Notes on fairness / “same trajectory” requirements:
    - For comparisons 1 & 2, we run two simulations with the same seed and the same
    call pattern (move then sense each step). Because both runs consume the RNG in
    the same way, the *motion trajectory* is identical between NUF and UF.
    - For comparison 3, we simulate once (NUF) and then replay the exact same true
    states + sensor readings through filtering, smoothing, and the sensor-only
    baseline.

"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


# So we can import the handout folder regardless of the current working directory. 
# This is needed to import the models and FilterSmoother for the simulations and evaluations.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(_THIS_DIR))

# these are imported here to avoid circular imports with Filters.py
# which imports the same models for its own use. 
# The models are not used directly in this file, 
# but they are needed for the simulate_move_sense() function, which creates the simulations that we evaluate.
from models import StateModel, TransitionModel, ObservationModel_NUF, ObservationModel_UF, RobotSim
from Filters import FilterSmoother


@dataclass(frozen=True)
class Simulation:
    """A simulated run: aligned true states and sensor readings."""

    sm: StateModel
    tm: TransitionModel
    readings: list[int | None]  # length = total_steps
    true_states: list[int]  # length = total_steps


def _belief_to_position(sm: StateModel, state_belief: np.ndarray) -> tuple[int, int]:
    """Convert belief over pose states into most-likely *position* estimate.

    The hidden state includes heading (4 per cell). The sensor reports position
    only, and the viewer expects us to sum the 4 heading probabilities per cell.
    """

    state_belief = np.asarray(state_belief, dtype=float).reshape(-1)
    rows, cols, head = sm.get_grid_dimensions()
    if head != 4:
        raise ValueError(f"Expected 4 headings per cell, got {head}.")

    # Shape (rows*cols, 4) and sum headings -> position belief.
    pos_belief = state_belief.reshape(rows * cols, head).sum(axis=1)
    r_hat = int(np.argmax(pos_belief))
    return sm.reading_to_position(r_hat)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _cumulative_average(values: Iterable[float]) -> np.ndarray:
    """Return cumulative mean array: avg[t] = mean(values[0..t])."""

    vals = np.asarray(list(values), dtype=float)
    return np.cumsum(vals) / (np.arange(vals.size) + 1.0)


def simulate_move_sense(
    *,
    rows: int,
    cols: int,
    observation_model: str,
    total_steps: int,
    seed: int,
) -> Simulation:
    """Simulate a run using the provided RobotSim, TransitionModel and observation model."""

    random.seed(seed)
    np.random.seed(seed)

    sm = StateModel(rows, cols)
    tm = TransitionModel(sm)

    if observation_model.upper() == "NUF":
        om = ObservationModel_NUF.ObservationModel(sm)
    elif observation_model.upper() == "UF":
        om = ObservationModel_UF.ObservationModelUF(sm)
    else:
        raise ValueError("observation_model must be 'NUF' or 'UF'")

    start_state = random.randrange(sm.get_num_of_states())
    rs = RobotSim(start_state, sm)

    true_states: list[int] = []
    readings: list[int | None] = []

    # One update step is: move -> sense.
    for _ in range(total_steps):
        s = rs.move_once(tm)
        r = rs.sense_in_current_state(om)
        true_states.append(int(s))
        readings.append(r)

    return Simulation(sm=sm, tm=tm, readings=readings, true_states=true_states)


def run_filtering_errors(sim: Simulation, *, observation_model: str) -> np.ndarray:
    """Run forward filtering on a pre-simulated sequence, return per-step Manhattan errors."""

    sm = sim.sm
    tm = sim.tm

    if observation_model.upper() == "NUF":
        om = ObservationModel_NUF.ObservationModel(sm)
    elif observation_model.upper() == "UF":
        om = ObservationModel_UF.ObservationModelUF(sm)
    else:
        raise ValueError("observation_model must be 'NUF' or 'UF'")

    f = np.ones(sm.get_num_of_states(), dtype=float) / float(sm.get_num_of_states())
    hmm = FilterSmoother(f, tm, om, sm)

    errors: list[float] = []

    for s_true, r in zip(sim.true_states, sim.readings, strict=True):
        f = hmm.filter(r, f)
        est_pos = _belief_to_position(sm, f)
        true_pos = sm.state_to_position(int(s_true))
        errors.append(float(_manhattan(true_pos, est_pos)))

    return np.asarray(errors, dtype=float)


def run_fixed_lag_smoothing_errors(
    sim: Simulation,
    *,
    lag: int,
    observation_model: str,
    eval_steps: int,
) -> np.ndarray:
    """Compute fixed-lag smoothing errors for the first `eval_steps` steps.

    Requires `sim` to contain at least `eval_steps + lag + 1` readings, because
    at time t we use future evidence r_{t+1..t+lag} and we pass a buffer of
    length lag+1 (matching the Localizer interface).
    """

    if lag < 0:
        raise ValueError("lag must be >= 0")

    sm = sim.sm
    tm = sim.tm

    if observation_model.upper() == "NUF":
        om = ObservationModel_NUF.ObservationModel(sm)
    elif observation_model.upper() == "UF":
        om = ObservationModel_UF.ObservationModelUF(sm)
    else:
        raise ValueError("observation_model must be 'NUF' or 'UF'")

    if len(sim.readings) < eval_steps + lag + 1:
        raise ValueError(
            f"Need at least {eval_steps + lag + 1} readings, got {len(sim.readings)}."
        )

    f = np.ones(sm.get_num_of_states(), dtype=float) / float(sm.get_num_of_states())
    hmm = FilterSmoother(f, tm, om, sm)

    errors: list[float] = []

    for t in range(eval_steps):
        r_t = sim.readings[t]
        s_true = sim.true_states[t]

        # Forward filter up to time t
        f = hmm.filter(r_t, f)

        # Buffer of future readings: length lag+1.
        # Localizer passes a buffer whose last element is kept for the next step; our
        # `smooth()` uses only the first `len(buffer)-1` entries.
        future_buffer = sim.readings[t + 1 : t + lag + 2]
        fb = hmm.smooth(np.array(future_buffer, dtype=object), f)

        est_pos = _belief_to_position(sm, fb)
        true_pos = sm.state_to_position(int(s_true))
        errors.append(float(_manhattan(true_pos, est_pos)))

    return np.asarray(errors, dtype=float)


def run_sensor_only_baseline(sim: Simulation, *, eval_steps: int) -> np.ndarray:
    """Sensor-only baseline: Manhattan error using the sensor reading as estimate.

    Per handout hint, we compute the average only over steps with a valid sensor
    output (reading != None). For plotting over time, we show the cumulative
    average up to each time step.
    """

    sm = sim.sm

    cum_sum = 0.0
    count = 0
    avg_over_time: list[float] = []

    for t in range(eval_steps):
        s_true = sim.true_states[t]
        r = sim.readings[t]
        true_pos = sm.state_to_position(int(s_true))

        if r is not None:
            sensed_pos = sm.reading_to_position(int(r))
            cum_sum += float(_manhattan(true_pos, sensed_pos))
            count += 1

        # If count==0 (very early), define average as NaN to make the plot honest.
        avg_over_time.append(cum_sum / count if count > 0 else float("nan"))

    return np.asarray(avg_over_time, dtype=float)


def _plot_lines(
    *,
    title: str,
    series: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for label, y in series.items():
        plt.plot(y, label=label)

    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Cumulative average Manhattan distance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    out_dir = _THIS_DIR / "evaluation_outputs"

    steps = 300

    # --- Setting 1: 4x4, Filtering, NUF vs UF ---
    seed_4 = 20260223
    sim_nuf_4 = simulate_move_sense(rows=4, cols=4, observation_model="NUF", total_steps=steps, seed=seed_4)
    sim_uf_4 = simulate_move_sense(rows=4, cols=4, observation_model="UF", total_steps=steps, seed=seed_4)

    err_nuf_4 = run_filtering_errors(sim_nuf_4, observation_model="NUF")
    err_uf_4 = run_filtering_errors(sim_uf_4, observation_model="UF")

    _plot_lines(
        title="4x4 grid — Forward Filtering — NUF vs UF",
        series={
            "Filtering (NUF)": _cumulative_average(err_nuf_4),
            "Filtering (UF)": _cumulative_average(err_uf_4),
        },
        out_path=out_dir / "eval1_4x4_filtering_nuf_vs_uf.png",
    )

    # --- Setting 2: 20x20, Filtering, NUF vs UF ---
    seed_20 = 20260224
    sim_nuf_20 = simulate_move_sense(rows=20, cols=20, observation_model="NUF", total_steps=steps, seed=seed_20)
    sim_uf_20 = simulate_move_sense(rows=20, cols=20, observation_model="UF", total_steps=steps, seed=seed_20)

    err_nuf_20 = run_filtering_errors(sim_nuf_20, observation_model="NUF")
    err_uf_20 = run_filtering_errors(sim_uf_20, observation_model="UF")

    _plot_lines(
        title="20x20 grid — Forward Filtering — NUF vs UF",
        series={
            "Filtering (NUF)": _cumulative_average(err_nuf_20),
            "Filtering (UF)": _cumulative_average(err_uf_20),
        },
        out_path=out_dir / "eval2_20x20_filtering_nuf_vs_uf.png",
    )

    # --- Setting 3: 10x20, NUF, Filtering vs Smoothing (lag=5) vs Sensor-only ---
    lag = 5
    seed_10x20 = 20260225

    # We need extra steps to have future evidence for smoothing buffers.
    sim_10x20 = simulate_move_sense(
        rows=10,
        cols=20,
        observation_model="NUF",
        total_steps=steps + lag + 1,
        seed=seed_10x20,
    )

    # Filtering errors for the first `steps`.
    err_filter = run_filtering_errors(
        Simulation(sim_10x20.sm, sim_10x20.tm, sim_10x20.readings[:steps], sim_10x20.true_states[:steps]),
        observation_model="NUF",
    )

    # Fixed-lag smoothing errors.
    err_smooth = run_fixed_lag_smoothing_errors(sim_10x20, lag=lag, observation_model="NUF", eval_steps=steps)

    # Sensor-only baseline (cumulative avg only over valid readings).
    sensor_avg = run_sensor_only_baseline(sim_10x20, eval_steps=steps)

    _plot_lines(
        title="10x20 grid — NUF — Filtering vs Fixed-lag Smoothing (lag=5) vs Sensor-only",
        series={
            "Filtering (NUF)": _cumulative_average(err_filter),
            f"Smoothing (lag={lag})": _cumulative_average(err_smooth),
            "Sensor-only (valid readings only)": sensor_avg,
        },
        out_path=out_dir / "eval3_10x20_filter_vs_smooth_vs_sensor.png",
    )

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()
