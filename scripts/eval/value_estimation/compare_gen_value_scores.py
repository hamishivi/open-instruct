#!/usr/bin/env python3
"""Compare two generative-value scorers on the same fixed Monte Carlo panel."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    parser.add_argument("--output_json", type=pathlib.Path)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mse_noninferiority_margin", type=float, default=0.01)
    parser.add_argument("--auc_noninferiority_margin", type=float, default=0.02)
    parser.add_argument(
        "--prediction_column",
        default="predicted_values",
        help="Nested parquet column to evaluate, for example soft_predicted_values.",
    )
    return parser.parse_args()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def _trajectory_band(position: int, final_position: int) -> str:
    fraction = position / max(final_position, 1)
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "middle"
    return "late"


def _absolute_prefix_band(position: int) -> str:
    if position < 1024:
        return "lt_1024"
    if position < 2048:
        return "1024_2047"
    if position < 4096:
        return "2048_4095"
    return "ge_4096"


def _flatten_scores(
    path: pathlib.Path, *, prediction_column: str = "predicted_values"
) -> dict[tuple[str, bool, str, int], dict[str, Any]]:
    frame = pd.read_parquet(path)
    required = {"problem", "rollout_tokens", "rollout_is_correct", "probe_positions", "mc_values", prediction_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    flattened: dict[tuple[str, bool, str, int], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        positions = list(row["probe_positions"])
        targets = list(row["mc_values"])
        predictions = list(row[prediction_column])
        if not (len(positions) == len(targets) == len(predictions)):
            raise ValueError(f"Mismatched probe arrays for problem {row['problem']!r} in {path}.")
        final_position = len(row["rollout_tokens"]) - 1
        is_correct = bool(row["rollout_is_correct"])
        for position, target, prediction in zip(positions, targets, predictions, strict=True):
            position = int(position)
            state_kind = "final_action" if position == final_position else "intermediate"
            key = (str(row["problem"]), is_correct, state_kind, position)
            if key in flattened:
                raise ValueError(f"Duplicate probe key {key!r} in {path}.")
            flattened[key] = {
                "problem": str(row["problem"]),
                "rollout_is_correct": is_correct,
                "state_kind": state_kind,
                "trajectory_band": _trajectory_band(position, final_position),
                "absolute_prefix_band": _absolute_prefix_band(position),
                "target": float(target),
                "prediction": _optional_float(prediction),
            }
    return flattened


def _pairwise_auc(correct: list[float], incorrect: list[float]) -> float | None:
    if not correct or not incorrect:
        return None
    correct_array = np.asarray(correct)[:, None]
    incorrect_array = np.asarray(incorrect)[None, :]
    return float(np.mean((correct_array > incorrect_array) + 0.5 * (correct_array == incorrect_array)))


def _average_ranks(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array, kind="stable")
    ranks = np.empty(len(values_array), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman_correlation(left: list[float], right: list[float]) -> float | None:
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _quantized_decile(value: float) -> int:
    """Map a normalized value to the direct-MC critic's integer score."""
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Expected a finite normalized score in [0, 1], got {value}.")
    return int(np.floor(value * 10.0 + 0.5))


def _score_scale_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Report use and calibration of the critic's discrete 0--10 score scale."""
    parsed_rows = [row for row in rows if row["prediction"] is not None]
    prediction_counts = np.zeros(11, dtype=np.int64)
    for row in parsed_rows:
        prediction_counts[_quantized_decile(float(row["prediction"]))] += 1

    metrics: dict[str, float | None] = {
        "prediction_decile_coverage": float(np.count_nonzero(prediction_counts)),
    }
    if parsed_rows:
        probabilities = prediction_counts[prediction_counts > 0] / len(parsed_rows)
        metrics["prediction_decile_entropy_bits"] = float(-np.sum(probabilities * np.log2(probabilities)))
    else:
        metrics["prediction_decile_entropy_bits"] = None

    for decile in range(11):
        metrics[f"prediction_decile_{decile}_examples"] = float(prediction_counts[decile])
        target_group = [row for row in rows if _quantized_decile(float(row["target"])) == decile]
        parsed_target_group = [row for row in target_group if row["prediction"] is not None]
        prefix = f"target_decile_{decile}"
        metrics[f"{prefix}_examples"] = float(len(target_group))
        if parsed_target_group:
            target_mean = float(np.mean([row["target"] for row in parsed_target_group]))
            prediction_mean = float(np.mean([row["prediction"] for row in parsed_target_group]))
            metrics[f"{prefix}_target_mean"] = target_mean
            metrics[f"{prefix}_prediction_mean"] = prediction_mean
            metrics[f"{prefix}_calibration_bias"] = prediction_mean - target_mean
        else:
            metrics[f"{prefix}_target_mean"] = None
            metrics[f"{prefix}_prediction_mean"] = None
            metrics[f"{prefix}_calibration_bias"] = None
        errors = [
            1.0 if row["prediction"] is None else (row["prediction"] - row["target"]) ** 2
            for row in target_group
        ]
        metrics[f"{prefix}_penalized_mse"] = float(np.mean(errors)) if errors else None
    return metrics


def _mc_selection_metrics(
    rows: list[dict[str, Any]], *, trajectory_band: str | None
) -> dict[str, float | None]:
    problem_metrics = []
    total_pairs = 0
    for problem in sorted({row["problem"] for row in rows}):
        correct = [
            row
            for row in rows
            if row["problem"] == problem
            and row["state_kind"] == "intermediate"
            and row["rollout_is_correct"]
            and row["prediction"] is not None
        ]
        incorrect = [
            row
            for row in rows
            if row["problem"] == problem
            and row["state_kind"] == "intermediate"
            and not row["rollout_is_correct"]
            and row["prediction"] is not None
        ]
        pair_metrics = []
        for correct_row in correct:
            for incorrect_row in incorrect:
                if trajectory_band is None:
                    if correct_row["trajectory_band"] != incorrect_row["trajectory_band"]:
                        continue
                elif (
                    correct_row["trajectory_band"] != trajectory_band
                    or incorrect_row["trajectory_band"] != trajectory_band
                ):
                    continue
                correct_target = correct_row["target"]
                incorrect_target = incorrect_row["target"]
                if np.isclose(correct_target, incorrect_target):
                    continue
                prediction_delta = correct_row["prediction"] - incorrect_row["prediction"]
                target_delta = correct_target - incorrect_target
                if prediction_delta == 0.0:
                    accuracy = 0.5
                    selected_target = (correct_target + incorrect_target) / 2.0
                else:
                    accuracy = float(np.sign(prediction_delta) == np.sign(target_delta))
                    selected_target = correct_target if prediction_delta > 0.0 else incorrect_target
                random_target = (correct_target + incorrect_target) / 2.0
                pair_metrics.append(
                    (
                        accuracy,
                        max(correct_target, incorrect_target) - selected_target,
                        selected_target - random_target,
                    )
                )
        if pair_metrics:
            total_pairs += len(pair_metrics)
            problem_metrics.append(np.mean(pair_metrics, axis=0))

    if not problem_metrics:
        return {
            "accuracy": None,
            "regret": None,
            "gain_over_random": None,
            "problems": 0.0,
            "pairs": 0.0,
        }
    problem_metrics_array = np.asarray(problem_metrics)
    return {
        "accuracy": float(np.mean(problem_metrics_array[:, 0])),
        "regret": float(np.mean(problem_metrics_array[:, 1])),
        "gain_over_random": float(np.mean(problem_metrics_array[:, 2])),
        "problems": float(len(problem_metrics)),
        "pairs": float(total_pairs),
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    parsed_rows = [row for row in rows if row["prediction"] is not None]
    penalized_errors = [1.0 if row["prediction"] is None else (row["prediction"] - row["target"]) ** 2 for row in rows]
    metrics: dict[str, float | None] = {
        "examples": float(len(rows)),
        "parse_rate": float(len(parsed_rows) / len(rows)) if rows else None,
        "penalized_mse": float(np.mean(penalized_errors)) if rows else None,
        "prediction_mean": float(np.mean([row["prediction"] for row in parsed_rows])) if parsed_rows else None,
    }
    if len(parsed_rows) > 1:
        metrics["spearman"] = _spearman_correlation(
            [row["prediction"] for row in parsed_rows], [row["target"] for row in parsed_rows]
        )
    else:
        metrics["spearman"] = None
    metrics.update(_score_scale_metrics(rows))

    for state_kind in ("final_action", "intermediate"):
        for outcome_name, outcome in (("correct", True), ("incorrect", False)):
            group = [row for row in rows if row["state_kind"] == state_kind and row["rollout_is_correct"] is outcome]
            errors = [1.0 if row["prediction"] is None else (row["prediction"] - row["target"]) ** 2 for row in group]
            prefix = f"{state_kind}_{outcome_name}"
            metrics[f"{prefix}_examples"] = float(len(group))
            metrics[f"{prefix}_penalized_mse"] = float(np.mean(errors)) if errors else None
            parsed = [row["prediction"] for row in group if row["prediction"] is not None]
            metrics[f"{prefix}_prediction_mean"] = float(np.mean(parsed)) if parsed else None

    intermediate_correct = [
        row["prediction"]
        for row in rows
        if row["state_kind"] == "intermediate" and row["rollout_is_correct"] and row["prediction"] is not None
    ]
    intermediate_incorrect = [
        row["prediction"]
        for row in rows
        if row["state_kind"] == "intermediate" and not row["rollout_is_correct"] and row["prediction"] is not None
    ]
    if intermediate_correct and intermediate_incorrect:
        metrics["intermediate_outcome_gap"] = float(np.mean(intermediate_correct) - np.mean(intermediate_incorrect))
    else:
        metrics["intermediate_outcome_gap"] = None
    metrics["intermediate_outcome_auc"] = _pairwise_auc(intermediate_correct, intermediate_incorrect)

    # The policy compares continuations of the same prompt. A pooled AUC also
    # compares states from unrelated problems, so it can reward problem-level
    # calibration offsets that never help an actual policy decision. Average a
    # within-problem AUC instead, giving every held-out problem equal weight.
    within_problem_aucs = []
    within_problem_pairs = 0
    for problem in sorted({row["problem"] for row in rows}):
        problem_correct = [
            row["prediction"]
            for row in rows
            if row["problem"] == problem
            and row["state_kind"] == "intermediate"
            and row["rollout_is_correct"]
            and row["prediction"] is not None
        ]
        problem_incorrect = [
            row["prediction"]
            for row in rows
            if row["problem"] == problem
            and row["state_kind"] == "intermediate"
            and not row["rollout_is_correct"]
            and row["prediction"] is not None
        ]
        problem_auc = _pairwise_auc(problem_correct, problem_incorrect)
        if problem_auc is not None:
            within_problem_aucs.append(problem_auc)
            within_problem_pairs += len(problem_correct) * len(problem_incorrect)
    metrics["intermediate_within_problem_auc"] = (
        float(np.mean(within_problem_aucs)) if within_problem_aucs else None
    )
    metrics["intermediate_within_problem_auc_problems"] = float(len(within_problem_aucs))
    metrics["intermediate_within_problem_auc_pairs"] = float(within_problem_pairs)
    for trajectory_band in (None, "early", "middle", "late"):
        selection_metrics = _mc_selection_metrics(rows, trajectory_band=trajectory_band)
        band_suffix = "" if trajectory_band is None else f"_{trajectory_band}"
        for name, value in selection_metrics.items():
            metrics[f"intermediate_mc_selection{band_suffix}_{name}"] = value

    for prefix_band in ("lt_1024", "1024_2047", "2048_4095", "ge_4096"):
        group = [row for row in rows if row["absolute_prefix_band"] == prefix_band]
        if not group:
            continue
        parsed_group = [row for row in group if row["prediction"] is not None]
        penalized_errors = [
            1.0 if row["prediction"] is None else (row["prediction"] - row["target"]) ** 2 for row in group
        ]
        metric_prefix = f"absolute_prefix_{prefix_band}"
        metrics[f"{metric_prefix}_examples"] = float(len(group))
        metrics[f"{metric_prefix}_parse_rate"] = float(len(parsed_group) / len(group))
        metrics[f"{metric_prefix}_penalized_mse"] = float(np.mean(penalized_errors))
        metrics[f"{metric_prefix}_target_mean"] = float(np.mean([row["target"] for row in group]))
        metrics[f"{metric_prefix}_prediction_mean"] = (
            float(np.mean([row["prediction"] for row in parsed_group])) if parsed_group else None
        )
        selection_metrics = _mc_selection_metrics(group, trajectory_band=None)
        for name, value in selection_metrics.items():
            metrics[f"{metric_prefix}_mc_selection_{name}"] = value
        for outcome_name, outcome in (("correct", True), ("incorrect", False)):
            outcome_group = [
                row for row in group if row["state_kind"] == "intermediate" and row["rollout_is_correct"] is outcome
            ]
            outcome_parsed = [row for row in outcome_group if row["prediction"] is not None]
            outcome_prefix = f"{metric_prefix}_intermediate_{outcome_name}"
            metrics[f"{outcome_prefix}_examples"] = float(len(outcome_group))
            outcome_errors = [
                1.0 if row["prediction"] is None else (row["prediction"] - row["target"]) ** 2
                for row in outcome_group
            ]
            metrics[f"{outcome_prefix}_penalized_mse"] = (
                float(np.mean(outcome_errors)) if outcome_errors else None
            )
            if outcome_parsed:
                target_mean = float(np.mean([row["target"] for row in outcome_parsed]))
                prediction_mean = float(np.mean([row["prediction"] for row in outcome_parsed]))
                metrics[f"{outcome_prefix}_target_mean"] = target_mean
                metrics[f"{outcome_prefix}_prediction_mean"] = prediction_mean
                metrics[f"{outcome_prefix}_calibration_bias"] = prediction_mean - target_mean
            else:
                metrics[f"{outcome_prefix}_target_mean"] = None
                metrics[f"{outcome_prefix}_prediction_mean"] = None
                metrics[f"{outcome_prefix}_calibration_bias"] = None
    return metrics


def _clustered_delta_summary(
    per_problem_deltas: dict[str, list[float]], *, bootstrap_samples: int, rng: np.random.Generator
) -> dict[str, Any] | None:
    """Summarize paired error deltas while treating problems as the independent unit."""
    if not per_problem_deltas:
        return None
    problem_means = np.asarray([np.mean(values) for values in per_problem_deltas.values()], dtype=float)
    samples = rng.choice(
        problem_means, size=(bootstrap_samples, len(problem_means)), replace=True
    ).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "problems": len(problem_means),
        "problem_balanced_mse_delta_candidate_minus_baseline": float(np.mean(problem_means)),
        "problem_cluster_bootstrap_95pct_ci": [float(lower), float(upper)],
    }


def compare_scores(
    baseline_path: pathlib.Path,
    candidate_path: pathlib.Path,
    *,
    bootstrap_samples: int,
    seed: int,
    mse_noninferiority_margin: float,
    auc_noninferiority_margin: float,
    prediction_column: str = "predicted_values",
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1.")
    baseline = _flatten_scores(baseline_path, prediction_column=prediction_column)
    candidate = _flatten_scores(candidate_path, prediction_column=prediction_column)
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())
        extra = sorted(candidate.keys() - baseline.keys())
        raise ValueError(f"Panel probe keys differ; missing={missing[:3]}, extra={extra[:3]}.")

    baseline_rows = []
    candidate_rows = []
    problem_deltas: dict[str, list[float]] = {}
    absolute_prefix_problem_deltas: dict[str, dict[str, list[float]]] = {
        "lt_1024": {},
        "1024_2047": {},
        "2048_4095": {},
        "ge_4096": {},
        "ge_2048": {},
    }
    # Keep a separate intermediate-only view for the long-prefix acceptance
    # gate.  Final-action states are deliberately repeated during SFT and are
    # much easier to fit; allowing their gains into this gate could hide a
    # regression at the segment-start values that provide most actor credit.
    absolute_prefix_intermediate_problem_deltas: dict[str, dict[str, list[float]]] = {
        "lt_1024": {},
        "1024_2047": {},
        "2048_4095": {},
        "ge_4096": {},
        "ge_2048": {},
    }
    absolute_prefix_intermediate_outcome_problem_deltas: dict[str, dict[str, dict[str, list[float]]]] = {
        outcome: {
            "lt_1024": {},
            "1024_2047": {},
            "2048_4095": {},
            "ge_4096": {},
            "ge_2048": {},
        }
        for outcome in ("correct", "incorrect")
    }
    for key in baseline:
        baseline_row = baseline[key]
        candidate_row = candidate[key]
        if not np.isclose(baseline_row["target"], candidate_row["target"], atol=0.0, rtol=0.0):
            raise ValueError(f"Monte Carlo target differs for probe {key!r}.")
        baseline_rows.append(baseline_row)
        candidate_rows.append(candidate_row)
        baseline_error = (
            1.0 if baseline_row["prediction"] is None else (baseline_row["prediction"] - baseline_row["target"]) ** 2
        )
        candidate_error = (
            1.0
            if candidate_row["prediction"] is None
            else (candidate_row["prediction"] - candidate_row["target"]) ** 2
        )
        problem = baseline_row["problem"]
        error_delta = candidate_error - baseline_error
        problem_deltas.setdefault(problem, []).append(error_delta)
        prefix_band = baseline_row["absolute_prefix_band"]
        absolute_prefix_problem_deltas[prefix_band].setdefault(problem, []).append(error_delta)
        if prefix_band in {"2048_4095", "ge_4096"}:
            absolute_prefix_problem_deltas["ge_2048"].setdefault(problem, []).append(error_delta)
        if baseline_row["state_kind"] == "intermediate":
            absolute_prefix_intermediate_problem_deltas[prefix_band].setdefault(problem, []).append(error_delta)
            outcome = "correct" if baseline_row["rollout_is_correct"] else "incorrect"
            absolute_prefix_intermediate_outcome_problem_deltas[outcome][prefix_band].setdefault(
                problem, []
            ).append(error_delta)
            if prefix_band in {"2048_4095", "ge_4096"}:
                absolute_prefix_intermediate_problem_deltas["ge_2048"].setdefault(problem, []).append(error_delta)
                absolute_prefix_intermediate_outcome_problem_deltas[outcome]["ge_2048"].setdefault(
                    problem, []
                ).append(error_delta)

    baseline_metrics = _metrics(baseline_rows)
    candidate_metrics = _metrics(candidate_rows)
    rng = np.random.default_rng(seed)
    overall_delta = _clustered_delta_summary(
        problem_deltas, bootstrap_samples=bootstrap_samples, rng=rng
    )
    if overall_delta is None:
        raise ValueError("Cannot compare empty score panels.")
    absolute_prefix_deltas = {
        band: summary
        for band, deltas in absolute_prefix_problem_deltas.items()
        if (
            summary := _clustered_delta_summary(deltas, bootstrap_samples=bootstrap_samples, rng=rng)
        )
        is not None
    }
    absolute_prefix_intermediate_deltas = {
        band: summary
        for band, deltas in absolute_prefix_intermediate_problem_deltas.items()
        if (
            summary := _clustered_delta_summary(deltas, bootstrap_samples=bootstrap_samples, rng=rng)
        )
        is not None
    }
    absolute_prefix_intermediate_outcome_deltas = {
        outcome: {
            band: summary
            for band, deltas in band_deltas.items()
            if (
                summary := _clustered_delta_summary(deltas, bootstrap_samples=bootstrap_samples, rng=rng)
            )
            is not None
        }
        for outcome, band_deltas in absolute_prefix_intermediate_outcome_problem_deltas.items()
    }

    baseline_correct_mse = baseline_metrics["intermediate_correct_penalized_mse"]
    candidate_correct_mse = candidate_metrics["intermediate_correct_penalized_mse"]
    baseline_incorrect_mse = baseline_metrics["intermediate_incorrect_penalized_mse"]
    candidate_incorrect_mse = candidate_metrics["intermediate_incorrect_penalized_mse"]
    baseline_auc = baseline_metrics["intermediate_within_problem_auc"]
    candidate_auc = candidate_metrics["intermediate_within_problem_auc"]
    baseline_selection_accuracy = baseline_metrics["intermediate_mc_selection_accuracy"]
    candidate_selection_accuracy = candidate_metrics["intermediate_mc_selection_accuracy"]
    long_prefix_intermediate_delta = absolute_prefix_intermediate_deltas.get("ge_2048")
    long_successful_prefix_delta = absolute_prefix_intermediate_outcome_deltas["correct"].get("ge_2048")
    long_failed_prefix_delta = absolute_prefix_intermediate_outcome_deltas["incorrect"].get("ge_2048")
    checks = {
        "candidate_parse_rate_at_least_0_99": candidate_metrics["parse_rate"] >= 0.99,
        "problem_balanced_mse_noninferior": (
            overall_delta["problem_cluster_bootstrap_95pct_ci"][1] <= mse_noninferiority_margin
        ),
        "long_prefix_intermediate_mse_noninferior": bool(
            long_prefix_intermediate_delta is not None
            and long_prefix_intermediate_delta["problem_cluster_bootstrap_95pct_ci"][1]
            <= mse_noninferiority_margin
        ),
        "successful_intermediate_mse_noninferior": (
            candidate_correct_mse <= baseline_correct_mse + mse_noninferiority_margin
        ),
        "failed_intermediate_mse_improved": candidate_incorrect_mse < baseline_incorrect_mse,
        "long_successful_prefix_mse_noninferior": bool(
            long_successful_prefix_delta is not None
            and long_successful_prefix_delta["problem_cluster_bootstrap_95pct_ci"][1]
            <= mse_noninferiority_margin
        ),
        "long_failed_prefix_mse_improved": bool(
            long_failed_prefix_delta is not None
            and long_failed_prefix_delta["problem_balanced_mse_delta_candidate_minus_baseline"] < 0.0
        ),
        "intermediate_within_problem_auc_noninferior": (
            candidate_auc >= baseline_auc - auc_noninferiority_margin
        ),
        "intermediate_mc_selection_accuracy_noninferior": (
            candidate_selection_accuracy >= baseline_selection_accuracy - auc_noninferiority_margin
        ),
    }
    return {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "prediction_column": prediction_column,
        "problems": len(problem_deltas),
        "probes": len(baseline_rows),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "problem_balanced_mse_delta_candidate_minus_baseline": overall_delta[
            "problem_balanced_mse_delta_candidate_minus_baseline"
        ],
        "problem_cluster_bootstrap_95pct_ci": overall_delta["problem_cluster_bootstrap_95pct_ci"],
        "absolute_prefix_mse_deltas": absolute_prefix_deltas,
        "absolute_prefix_intermediate_mse_deltas": absolute_prefix_intermediate_deltas,
        "absolute_prefix_intermediate_outcome_mse_deltas": absolute_prefix_intermediate_outcome_deltas,
        "intermediate_within_problem_auc_delta_candidate_minus_baseline": candidate_auc - baseline_auc,
        "intermediate_mc_selection_accuracy_delta_candidate_minus_baseline": (
            candidate_selection_accuracy - baseline_selection_accuracy
        ),
        "bootstrap_samples": bootstrap_samples,
        "gate": {
            "accepted": all(checks.values()),
            "checks": checks,
            "mse_noninferiority_margin": mse_noninferiority_margin,
            "auc_noninferiority_margin": auc_noninferiority_margin,
        },
    }


def main() -> None:
    args = parse_args()
    comparison = compare_scores(
        args.baseline,
        args.candidate,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        mse_noninferiority_margin=args.mse_noninferiority_margin,
        auc_noninferiority_margin=args.auc_noninferiority_margin,
        prediction_column=args.prediction_column,
    )
    rendered = json.dumps(comparison, indent=2, sort_keys=True)
    if args.output_json is not None:
        if args.output_json.exists():
            raise FileExistsError(f"Refusing to overwrite comparison output: {args.output_json}.")
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
