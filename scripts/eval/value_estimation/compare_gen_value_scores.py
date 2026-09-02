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


def _flatten_scores(path: pathlib.Path) -> dict[tuple[str, bool, str, int], dict[str, Any]]:
    frame = pd.read_parquet(path)
    required = {"problem", "rollout_tokens", "rollout_is_correct", "probe_positions", "mc_values", "predicted_values"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    flattened: dict[tuple[str, bool, str, int], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        positions = list(row["probe_positions"])
        targets = list(row["mc_values"])
        predictions = list(row["predicted_values"])
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
    return metrics


def compare_scores(
    baseline_path: pathlib.Path,
    candidate_path: pathlib.Path,
    *,
    bootstrap_samples: int,
    seed: int,
    mse_noninferiority_margin: float,
    auc_noninferiority_margin: float,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1.")
    baseline = _flatten_scores(baseline_path)
    candidate = _flatten_scores(candidate_path)
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())
        extra = sorted(candidate.keys() - baseline.keys())
        raise ValueError(f"Panel probe keys differ; missing={missing[:3]}, extra={extra[:3]}.")

    baseline_rows = []
    candidate_rows = []
    problem_deltas: dict[str, list[float]] = {}
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
        problem_deltas.setdefault(baseline_row["problem"], []).append(candidate_error - baseline_error)

    baseline_metrics = _metrics(baseline_rows)
    candidate_metrics = _metrics(candidate_rows)
    per_problem_deltas = np.asarray([np.mean(values) for values in problem_deltas.values()])
    rng = np.random.default_rng(seed)
    samples = rng.choice(per_problem_deltas, size=(bootstrap_samples, len(per_problem_deltas)), replace=True).mean(
        axis=1
    )
    delta_ci = np.quantile(samples, [0.025, 0.975])
    problem_balanced_delta = float(np.mean(per_problem_deltas))

    baseline_correct_mse = baseline_metrics["intermediate_correct_penalized_mse"]
    candidate_correct_mse = candidate_metrics["intermediate_correct_penalized_mse"]
    baseline_incorrect_mse = baseline_metrics["intermediate_incorrect_penalized_mse"]
    candidate_incorrect_mse = candidate_metrics["intermediate_incorrect_penalized_mse"]
    baseline_auc = baseline_metrics["intermediate_within_problem_auc"]
    candidate_auc = candidate_metrics["intermediate_within_problem_auc"]
    baseline_selection_accuracy = baseline_metrics["intermediate_mc_selection_accuracy"]
    candidate_selection_accuracy = candidate_metrics["intermediate_mc_selection_accuracy"]
    checks = {
        "candidate_parse_rate_at_least_0_99": candidate_metrics["parse_rate"] >= 0.99,
        "problem_balanced_mse_noninferior": float(delta_ci[1]) <= mse_noninferiority_margin,
        "successful_intermediate_mse_improved": candidate_correct_mse < baseline_correct_mse,
        "failed_intermediate_mse_noninferior": (
            candidate_incorrect_mse <= baseline_incorrect_mse + mse_noninferiority_margin
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
        "problems": len(problem_deltas),
        "probes": len(baseline_rows),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "problem_balanced_mse_delta_candidate_minus_baseline": problem_balanced_delta,
        "problem_cluster_bootstrap_95pct_ci": [float(delta_ci[0]), float(delta_ci[1])],
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
