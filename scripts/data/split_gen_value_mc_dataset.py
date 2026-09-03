#!/usr/bin/env python3
"""Split paired Monte Carlo value data into problem-disjoint train and holdout parquets."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from open_instruct import value_estimation


def split_paired_mc_rows(
    rows: Sequence[dict[str, Any]], *, heldout_problem_count: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Split whole normalized problem groups while validating paired outcomes."""
    if isinstance(heldout_problem_count, bool) or not isinstance(heldout_problem_count, int):
        raise ValueError(f"heldout_problem_count must be a positive integer, got {heldout_problem_count!r}.")
    if heldout_problem_count <= 0:
        raise ValueError(f"heldout_problem_count must be positive, got {heldout_problem_count}.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a nonnegative integer, got {seed!r}.")

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_problem_strings: dict[str, set[str]] = defaultdict(set)
    for row_index, row in enumerate(rows):
        problem = row.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"MC row {row_index} has no nonempty problem string.")
        identity = value_estimation.normalize_problem_identity(problem)
        grouped_rows[identity].append(row)
        raw_problem_strings[identity].add(problem)

    if heldout_problem_count >= len(grouped_rows):
        raise ValueError(
            f"heldout_problem_count={heldout_problem_count} must leave at least one of "
            f"{len(grouped_rows)} problem identities for training."
        )

    for identity, problem_rows in grouped_rows.items():
        outcomes = []
        ground_truths = set()
        for row in problem_rows:
            outcome = row.get("rollout_is_correct")
            if outcome not in (False, True, 0, 1):
                raise ValueError(
                    f"Problem identity {identity!r} has an invalid rollout_is_correct value: {outcome!r}."
                )
            outcomes.append(bool(outcome))
            ground_truth = row.get("ground_truth")
            if not isinstance(ground_truth, str) or not ground_truth.strip():
                raise ValueError(f"Problem identity {identity!r} has no nonempty ground truth string.")
            ground_truths.add(ground_truth)
        if len(problem_rows) != 2 or set(outcomes) != {False, True}:
            raise ValueError(
                f"Problem identity {identity!r} must have exactly one correct and one incorrect rollout; "
                f"found {len(problem_rows)} rows with outcomes {outcomes}."
            )
        if len(ground_truths) != 1:
            raise ValueError(
                f"Problem identity {identity!r} must have one consistent nonempty ground truth; "
                f"found {sorted(ground_truths)}."
            )

    identities = sorted(grouped_rows)
    random.Random(seed).shuffle(identities)
    heldout_identities = set(identities[:heldout_problem_count])
    train_rows = [row for identity in identities if identity not in heldout_identities for row in grouped_rows[identity]]
    heldout_rows = [row for identity in identities if identity in heldout_identities for row in grouped_rows[identity]]
    summary = {
        "input_rows": len(rows),
        "unique_problem_identities": len(grouped_rows),
        "formatting_variant_identities": sum(len(strings) > 1 for strings in raw_problem_strings.values()),
        "train_rows": len(train_rows),
        "train_problem_identities": len(grouped_rows) - heldout_problem_count,
        "heldout_rows": len(heldout_rows),
        "heldout_problem_identities": heldout_problem_count,
        "seed": seed,
    }
    return train_rows, heldout_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_parquet", type=pathlib.Path)
    parser.add_argument("--train_output", required=True, type=pathlib.Path)
    parser.add_argument("--heldout_output", required=True, type=pathlib.Path)
    parser.add_argument("--heldout_problem_count", required=True, type=int)
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args()


def main() -> None:
    import pandas as pd  # noqa: PLC0415

    args = parse_args()
    resolved_paths = {args.input_parquet.resolve(), args.train_output.resolve(), args.heldout_output.resolve()}
    if len(resolved_paths) != 3:
        raise ValueError("Input, training output, and held-out output paths must be distinct.")
    frame = pd.read_parquet(args.input_parquet)
    train_rows, heldout_rows, summary = split_paired_mc_rows(
        frame.to_dict(orient="records"), heldout_problem_count=args.heldout_problem_count, seed=args.seed
    )

    for output_path in (args.train_output, args.heldout_output):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    train_temporary = args.train_output.with_name(f"{args.train_output.name}.tmp")
    heldout_temporary = args.heldout_output.with_name(f"{args.heldout_output.name}.tmp")
    pd.DataFrame(train_rows).to_parquet(train_temporary, index=False)
    pd.DataFrame(heldout_rows).to_parquet(heldout_temporary, index=False)
    train_temporary.replace(args.train_output)
    heldout_temporary.replace(args.heldout_output)

    print(
        json.dumps(
            {
                **summary,
                "input": str(args.input_parquet),
                "train_output": str(args.train_output),
                "heldout_output": str(args.heldout_output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
