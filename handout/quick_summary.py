"""
This script runs the three evaluations described in the handout 
and prints out a quick summary of the results, including mean error, 
median error, last-50 mean, and final cumulative average for each evaluation. 
The code uses the `evaluation_runner` module to simulate the environments and compute the errors for filtering and smoothing.

AUTHORS: 
    1. Prince Samuel Kyeremanteng
    2. Cecilia Quinn

This code was optimized by an LLM (ChatGPT-4) and we used it to quickly summarize the results of the evaluations, which was very useful in our discussions and analysis. 
The LLM helped us to quickly extract key statistics from the error data, which allowed us to focus on interpreting the results rather than spending time on manual calculations.
"""

from __future__ import annotations

import numpy as np

import evaluation_runner as er


def summarize(label: str, errors: np.ndarray) -> None:
    errors = np.asarray(errors, dtype=float)
    print(label)
    print(f"  mean error: {errors.mean():.4f}")
    print(f"  median error: {np.median(errors):.4f}")
    print(f"  last-50 mean: {errors[-50:].mean():.4f}")
    print(f"  final cumulative avg: {(errors.cumsum()[-1] / len(errors)):.4f}")


def main() -> None:
    steps = 300

    # Eval 1
    seed_4 = 20260223
    sim_nuf_4 = er.simulate_move_sense(rows=4, cols=4, observation_model="NUF", total_steps=steps, seed=seed_4)
    sim_uf_4 = er.simulate_move_sense(rows=4, cols=4, observation_model="UF", total_steps=steps, seed=seed_4)
    err_nuf_4 = er.run_filtering_errors(sim_nuf_4, observation_model="NUF")
    err_uf_4 = er.run_filtering_errors(sim_uf_4, observation_model="UF")

    # Eval 2
    seed_20 = 20260224
    sim_nuf_20 = er.simulate_move_sense(rows=20, cols=20, observation_model="NUF", total_steps=steps, seed=seed_20)
    sim_uf_20 = er.simulate_move_sense(rows=20, cols=20, observation_model="UF", total_steps=steps, seed=seed_20)
    err_nuf_20 = er.run_filtering_errors(sim_nuf_20, observation_model="NUF")
    err_uf_20 = er.run_filtering_errors(sim_uf_20, observation_model="UF")

    # Eval 3
    lag = 5
    seed_10x20 = 20260225
    sim_10x20 = er.simulate_move_sense(
        rows=10, cols=20, observation_model="NUF", total_steps=steps + lag + 1, seed=seed_10x20
    )
    err_filter = er.run_filtering_errors(
        er.Simulation(sim_10x20.sm, sim_10x20.tm, sim_10x20.readings[:steps], sim_10x20.true_states[:steps]),
        observation_model="NUF",
    )
    err_smooth = er.run_fixed_lag_smoothing_errors(sim_10x20, lag=lag, observation_model="NUF", eval_steps=steps)
    sensor_avg = er.run_sensor_only_baseline(sim_10x20, eval_steps=steps)

    print("\n=== Eval 1: 4x4 Filtering (NUF vs UF) ===")
    summarize("Filtering NUF", err_nuf_4)
    summarize("Filtering UF", err_uf_4)

    print("\n=== Eval 2: 20x20 Filtering (NUF vs UF) ===")
    summarize("Filtering NUF", err_nuf_20)
    summarize("Filtering UF", err_uf_20)

    print("\n=== Eval 3: 10x20 NUF (Filtering vs Smoothing lag=5) ===")
    summarize("Filtering NUF", err_filter)
    summarize("Smoothing lag=5", err_smooth)
    print("Sensor-only baseline (cumulative avg over valid readings only)")
    print(f"  last value: {sensor_avg[-1]:.4f}")


if __name__ == "__main__":
    main()
