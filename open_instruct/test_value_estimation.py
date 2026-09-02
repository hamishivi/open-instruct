import argparse
import dataclasses
import math
import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from scripts.data import (
    prepare_gen_value_mc_sft,
    prepare_gen_value_sft,
    split_gen_value_mc_dataset,
    synthesize_gen_value_sft,
)
from scripts.eval.value_estimation import compare_gen_value_scores

from open_instruct import value_estimation


class _FakeTokenizer:
    def __init__(self):
        self.skip_special_tokens_calls = []

    def decode(self, token_ids, skip_special_tokens=True):
        self.skip_special_tokens_calls.append(skip_special_tokens)
        return ":".join(str(token_id) for token_id in token_ids)


class TestValueEstimationStates(unittest.TestCase):
    def test_generative_score_defaults_match_online_long_context_serving(self):
        config = value_estimation.ScoreDatasetConfig(
            input_dataset_path="input.parquet", output_path="scores.parquet", value_model_path="critic"
        )

        self.assertEqual(config.vllm_max_model_len, 32768)
        self.assertTrue(config.vllm_enable_prefix_caching)
        self.assertFalse(config.vllm_disable_custom_all_reduce)

    def test_generative_score_uses_actor_tokenizer_instead_of_critic_tokenizer(self):
        config = value_estimation.ScoreDatasetConfig(
            input_dataset_path="input.parquet", output_path="scores.parquet", value_model_path="critic"
        )
        self.assertEqual(
            value_estimation._resolve_generative_value_actor_tokenizer_path(
                config, ["Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B-Base"]
            ),
            "Qwen/Qwen3-4B-Base",
        )

        config.actor_tokenizer_name_or_path = "explicit-actor-tokenizer"
        self.assertEqual(
            value_estimation._resolve_generative_value_actor_tokenizer_path(config, ["actor-metadata"]),
            "explicit-actor-tokenizer",
        )

    def test_generative_score_rejects_ambiguous_or_missing_actor_tokenizer(self):
        config = value_estimation.ScoreDatasetConfig(
            input_dataset_path="input.parquet", output_path="scores.parquet", value_model_path="critic"
        )
        with self.assertRaisesRegex(ValueError, "multiple actor models"):
            value_estimation._resolve_generative_value_actor_tokenizer_path(config, ["actor-a", "actor-b"])
        with self.assertRaisesRegex(ValueError, "needs the tokenizer"):
            value_estimation._resolve_generative_value_actor_tokenizer_path(config, [])

    def test_generative_score_wrapper_forwards_actor_tokenizer(self):
        repository_root = pathlib.Path(__file__).parents[1]
        with tempfile.TemporaryDirectory(prefix="gen-value-score-wrapper-") as directory:
            test_root = pathlib.Path(directory)
            captured_args = test_root / "args.txt"
            fake_python = test_root / "fake-python"
            fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
            fake_python.chmod(0o755)
            environment = {
                **os.environ,
                "PYTHON_EXECUTABLE": str(fake_python),
                "CAPTURE_ARGS": str(captured_args),
                "ACTOR_TOKENIZER_NAME_OR_PATH": "actor-tokenizer",
                "VLLM_DISABLE_CUSTOM_ALL_REDUCE": "1",
            }

            result = subprocess.run(
                [
                    "bash",
                    "scripts/eval/value_estimation/score_generative_value.sh",
                    "critic",
                    "input.parquet",
                    "output.parquet",
                    "gt",
                ],
                cwd=repository_root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = captured_args.read_text().splitlines()
            tokenizer_index = arguments.index("--actor_tokenizer_name_or_path")
            self.assertEqual(arguments[tokenizer_index + 1], "actor-tokenizer")
            self.assertIn("--vllm_disable_custom_all_reduce", arguments)

    def test_mc_dataset_split_keeps_normalized_problem_pairs_disjoint(self):
        rows = [
            {"id": "a1", "problem": "problem  a\n", "ground_truth": "1", "rollout_is_correct": True},
            {"id": "a0", "problem": "problem a", "ground_truth": "1", "rollout_is_correct": False},
            {"id": "b1", "problem": "problem b", "ground_truth": "2", "rollout_is_correct": True},
            {"id": "b0", "problem": "problem b", "ground_truth": "2", "rollout_is_correct": False},
            {"id": "c1", "problem": "problem c", "ground_truth": "3", "rollout_is_correct": True},
            {"id": "c0", "problem": "problem c", "ground_truth": "3", "rollout_is_correct": False},
        ]

        train_rows, heldout_rows, summary = split_gen_value_mc_dataset.split_paired_mc_rows(
            rows, heldout_problem_count=1, seed=37
        )

        train_identities = {value_estimation.normalize_problem_identity(row["problem"]) for row in train_rows}
        heldout_identities = {value_estimation.normalize_problem_identity(row["problem"]) for row in heldout_rows}
        self.assertFalse(train_identities & heldout_identities)
        self.assertCountEqual([row["id"] for row in train_rows + heldout_rows], [row["id"] for row in rows])
        self.assertEqual(len(train_rows), 4)
        self.assertEqual(len(heldout_rows), 2)
        self.assertEqual(summary["formatting_variant_identities"], 1)

    def test_mc_dataset_split_rejects_unpaired_problem(self):
        rows = [
            {"problem": "broken", "ground_truth": "1", "rollout_is_correct": True},
            {"problem": "broken", "ground_truth": "1", "rollout_is_correct": True},
            {"problem": "other", "ground_truth": "2", "rollout_is_correct": True},
            {"problem": "other", "ground_truth": "2", "rollout_is_correct": False},
        ]

        with self.assertRaisesRegex(ValueError, "exactly one correct and one incorrect"):
            split_gen_value_mc_dataset.split_paired_mc_rows(rows, heldout_problem_count=1, seed=37)

    def test_mc_dataset_split_rejects_missing_ground_truth(self):
        rows = [
            {"problem": "broken", "ground_truth": None, "rollout_is_correct": True},
            {"problem": "broken", "ground_truth": None, "rollout_is_correct": False},
            {"problem": "other", "ground_truth": "2", "rollout_is_correct": True},
            {"problem": "other", "ground_truth": "2", "rollout_is_correct": False},
        ]

        with self.assertRaisesRegex(ValueError, "no nonempty ground truth"):
            split_gen_value_mc_dataset.split_paired_mc_rows(rows, heldout_problem_count=1, seed=37)

    def test_mc_value_sft_wrapper_forwards_absolute_prefix_gate(self):
        repository_root = pathlib.Path(__file__).parents[1]
        with tempfile.TemporaryDirectory(prefix="gen-value-wrapper-test-") as directory:
            test_root = pathlib.Path(directory)
            train_parquet = test_root / "train.parquet"
            heldout_parquet = test_root / "heldout.parquet"
            model_path = test_root / "model"
            captured_args = test_root / "args.txt"
            fake_python = test_root / "fake-python"
            train_parquet.touch()
            heldout_parquet.touch()
            model_path.mkdir()
            fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\nexit 73\n')
            fake_python.chmod(0o755)
            environment = {
                **os.environ,
                "MC_VALUE_PARQUET": str(train_parquet),
                "HELDOUT_VALUE_PARQUET": str(heldout_parquet),
                "MODEL_PATH": str(model_path),
                "PYTHON_EXECUTABLE": str(fake_python),
                "CAPTURE_ARGS": str(captured_args),
                "ACTOR_TOKENIZER_NAME_OR_PATH": "actor-tokenizer",
                "LONG_PREFIX_TOKEN_THRESHOLD": "3072",
                "MIN_LONG_PREFIX_FRACTION": "0.15",
            }

            result = subprocess.run(
                ["bash", "scripts/train/debug/genac_math_mc_value_sft_h200.sh"],
                cwd=repository_root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 73, result.stderr)
            arguments = captured_args.read_text().splitlines()
            threshold_index = arguments.index("--long_prefix_token_threshold")
            fraction_index = arguments.index("--min_long_prefix_fraction")
            tokenizer_index = arguments.index("--tokenizer_name_or_path")
            self.assertEqual(arguments[threshold_index + 1], "3072")
            self.assertEqual(arguments[fraction_index + 1], "0.15")
            self.assertEqual(arguments[tokenizer_index + 1], "actor-tokenizer")

    def test_mc_sft_holdout_pipeline_preserves_conditioning_and_scores_every_epoch(self):
        repository_root = pathlib.Path(__file__).parents[1]
        with tempfile.TemporaryDirectory(prefix="gen-value-holdout-pipeline-") as directory:
            test_root = pathlib.Path(directory)
            source_parquet = test_root / "source.parquet"
            model_path = test_root / "model"
            experiment_root = test_root / "experiment"
            call_log = test_root / "calls.log"
            fake_python = test_root / "fake-python"
            fake_bash = test_root / "fake-bash"
            source_parquet.touch()
            model_path.mkdir()
            fake_python.write_text(
                """#!/bin/sh
set -eu
printf 'python' >> "$CALL_LOG"
printf ' <%s>' "$@" >> "$CALL_LOG"
printf '\n' >> "$CALL_LOG"
case "$1" in
*split_gen_value_mc_dataset.py)
    shift
    while [ "$#" -gt 0 ]; do
        case "$1" in
        --train_output) train_output="$2"; shift 2 ;;
        --heldout_output) heldout_output="$2"; shift 2 ;;
        *) shift ;;
        esac
    done
    : > "$train_output"
    : > "$heldout_output"
    ;;
*compare_gen_value_scores.py)
    while [ "$#" -gt 0 ]; do
        if [ "$1" = --output_json ]; then
            : > "$2"
            break
        fi
        shift
    done
    ;;
esac
"""
            )
            fake_bash.write_text(
                """#!/bin/sh
set -eu
printf 'bash' >> "$CALL_LOG"
printf ' <%s>' "$@" >> "$CALL_LOG"
printf '\n' >> "$CALL_LOG"
case "$1" in
*score_generative_value.sh)
    [ "$5" = gt ]
    [ "$ACTOR_TOKENIZER_NAME_OR_PATH" = Qwen/Qwen3-4B-Base ]
    : > "$4"
    ;;
*genac_math_mc_value_sft_h200.sh)
    [ "$GEN_VALUE_CONDITIONING" = gt ]
    [ "$MIN_LONG_PREFIX_FRACTION" = 0.15 ]
    [ "$ACTOR_TOKENIZER_NAME_OR_PATH" = Qwen/Qwen3-4B-Base ]
    mkdir -p "$OUTPUT_DIR/epoch_1_model" "$OUTPUT_DIR/epoch_2_model"
    ;;
*) exit 97 ;;
esac
"""
            )
            fake_python.chmod(0o755)
            fake_bash.chmod(0o755)
            environment = {
                **os.environ,
                "MC_VALUE_PARQUET": str(source_parquet),
                "MODEL_PATH": str(model_path),
                "EXPERIMENT_ROOT": str(experiment_root),
                "GEN_VALUE_CONDITIONING": "gt",
                "PYTHON_EXECUTABLE": str(fake_python),
                "BASH_EXECUTABLE": str(fake_bash),
                "CALL_LOG": str(call_log),
            }

            result = subprocess.run(
                ["/bin/bash", "scripts/eval/value_estimation/run_genac_mc_sft_holdout_h200.sh"],
                cwd=repository_root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = call_log.read_text().splitlines()
            self.assertEqual(sum("score_generative_value.sh" in call for call in calls), 3)
            self.assertEqual(sum("genac_math_mc_value_sft_h200.sh" in call for call in calls), 1)
            self.assertEqual(sum("compare_gen_value_scores.py" in call for call in calls), 2)
            self.assertTrue((experiment_root / "data/train.parquet").is_file())
            self.assertTrue((experiment_root / "data/heldout.parquet").is_file())
            self.assertTrue((experiment_root / "comparisons/epoch_1_model.json").is_file())
            self.assertTrue((experiment_root / "comparisons/epoch_2_model.json").is_file())

    def test_trace_sft_can_target_prefix_states_without_discarding_other_kinds_by_default(self):
        examples = [{"id": "early", "state_kind": "segment_start"}, {"id": "terminal", "state_kind": "final_action"}]

        self.assertIs(prepare_gen_value_sft.filter_state_kinds(examples, None), examples)
        self.assertEqual(
            prepare_gen_value_sft.filter_state_kinds(examples, ["segment_start"]),
            [{"id": "early", "state_kind": "segment_start"}],
        )
        with self.assertRaisesRegex(RuntimeError, "No input traces matched state kinds"):
            prepare_gen_value_sft.filter_state_kinds(examples, ["missing"])

    def test_sae_mc_probes_match_online_segment_starts_and_final_action(self):
        logprobs = [math.log(0.9), math.log(0.1), math.log(0.9), math.log(0.1), math.log(0.9), math.log(0.9)]

        positions = value_estimation._sae_probe_positions(
            rollout_tokens=[10, 11, 12, 13, 14, 15],
            response_logprobs=logprobs,
            sae_threshold=0.2,
            max_segments=16,
            include_final_action_probe=True,
        )

        # Boundaries are [1, 3, 5], so online GenAC queries segment starts
        # [0, 2, 4], followed by the causal final-action state at 5.
        self.assertEqual(positions, [0, 2, 4, 5])

    def test_sae_mc_probe_limit_applies_to_segments_not_final_override(self):
        positions = value_estimation._sae_probe_positions(
            rollout_tokens=[10, 11, 12, 13, 14, 15],
            response_logprobs=[math.log(0.1)] * 6,
            sae_threshold=0.2,
            max_segments=2,
            include_final_action_probe=True,
        )

        self.assertEqual(positions, [0, 1, 5])

    def test_sae_mc_probes_require_aligned_logprobs(self):
        with self.assertRaisesRegex(ValueError, "align one-to-one"):
            value_estimation._sae_probe_positions(
                rollout_tokens=[10, 11],
                response_logprobs=[math.log(0.1)],
                sae_threshold=0.2,
                max_segments=16,
                include_final_action_probe=True,
            )

    def test_mc_sft_targets_supervise_only_the_direct_score(self):
        tokenizer = _FakeTokenizer()
        examples = prepare_gen_value_mc_sft.build_mc_sft_examples(
            [
                {
                    "problem": "Compute the answer.",
                    "rollout_tokens": [10, 11, 12],
                    "probe_positions": [1, 2],
                    "mc_values": [0.0, 0.5625],
                    "num_continuations": 16,
                    "response_token_limit": 8192,
                    "actor_model_name": "Qwen/Qwen3-4B-Base",
                    "actor_success_rate": 0.1,
                }
            ],
            tokenizer=tokenizer,
            min_continuations=16,
        )

        self.assertEqual(
            [example["generation"] for example in examples], [" <answer>0</answer>", " <answer>6</answer>"]
        )
        self.assertEqual([example["target"] for example in examples], [0.0, 0.5625])
        self.assertIn("<rollout>10</rollout>", examples[0]["prompt"])
        self.assertIn("<rollout>10:11</rollout>", examples[1]["prompt"])
        self.assertTrue(all(example["direct_mc_score_supervision"] for example in examples))
        self.assertEqual([example["trajectory_fraction"] for example in examples], [0.5, 1.0])
        self.assertEqual(tokenizer.skip_special_tokens_calls, [False, False])

    def test_mc_sft_score_grid_can_match_sixteen_continuation_targets_exactly(self):
        examples = prepare_gen_value_mc_sft.build_mc_sft_examples(
            [
                {
                    "problem": "Compute the answer.",
                    "rollout_tokens": [10, 11],
                    "probe_positions": [0, 1],
                    "mc_values": [1 / 16, 15 / 16],
                    "num_continuations": 16,
                }
            ],
            tokenizer=_FakeTokenizer(),
            min_continuations=16,
            score_max=16,
        )

        self.assertEqual(
            [example["generation"] for example in examples], [" <answer>1</answer>", " <answer>15</answer>"]
        )
        self.assertEqual([example["prediction"] for example in examples], [1 / 16, 15 / 16])
        self.assertEqual([example["squared_error"] for example in examples], [0.0, 0.0])

    def test_mc_sft_can_upweight_late_and_final_states(self):
        examples = prepare_gen_value_mc_sft.build_mc_sft_examples(
            [
                {
                    "problem": "Compute the answer.",
                    "rollout_tokens": [10, 11, 12, 13, 14, 15],
                    "probe_positions": [1, 3, 5],
                    "mc_values": [0.25, 0.5, 1.0],
                    "num_continuations": 16,
                }
            ],
            tokenizer=_FakeTokenizer(),
            min_continuations=16,
        )

        repeated = prepare_gen_value_mc_sft.repeat_examples_for_horizon(
            examples, final_action_repeat=3, late_state_repeat=2, late_state_fraction=0.5
        )

        self.assertEqual(len(repeated), 6)
        self.assertEqual([example["response_tokens_used"] for example in repeated], [1, 3, 3, 5, 5, 5])
        self.assertEqual([example["horizon_repeat_index"] for example in repeated], [0, 0, 1, 0, 1, 2])

    def test_mc_sft_trajectory_coverage_is_measured_before_replay(self):
        raw_examples = [
            {"state_kind": "segment_start", "trajectory_fraction": 0.0},
            {"state_kind": "segment_start", "trajectory_fraction": 0.5},
            {"state_kind": "segment_start", "trajectory_fraction": 0.8},
            {"state_kind": "final_action", "trajectory_fraction": 1.0},
        ]

        coverage = prepare_gen_value_mc_sft.require_trajectory_coverage(raw_examples, min_early_middle_fraction=0.5)

        self.assertEqual(coverage["early"], 1)
        self.assertEqual(coverage["middle"], 1)
        self.assertEqual(coverage["late_nonterminal"], 1)
        self.assertEqual(coverage["final_action"], 1)
        self.assertEqual(coverage["early_middle_examples"], 2)
        self.assertEqual(coverage["early_middle_fraction"], 0.5)
        with self.assertRaisesRegex(ValueError, "too late-heavy"):
            prepare_gen_value_mc_sft.require_trajectory_coverage(raw_examples, min_early_middle_fraction=0.75)

    def test_mc_sft_prefix_length_coverage_is_measured_before_replay(self):
        raw_examples = [
            {"response_tokens_used": 0},
            {"response_tokens_used": 1023},
            {"response_tokens_used": 1024},
            {"response_tokens_used": 2048},
            {"response_tokens_used": 3072},
        ]

        coverage = prepare_gen_value_mc_sft.require_prefix_length_coverage(
            raw_examples, long_prefix_token_threshold=2048, min_long_prefix_fraction=0.4
        )

        self.assertEqual(coverage["zero_prefix_examples"], 1)
        self.assertEqual(coverage["at_least_1024_tokens"], 3)
        self.assertEqual(coverage["at_least_2048_tokens"], 2)
        self.assertEqual(coverage["at_least_3072_tokens"], 1)
        self.assertEqual(coverage["long_prefix_examples"], 2)
        self.assertEqual(coverage["long_prefix_fraction"], 0.4)
        with self.assertRaisesRegex(ValueError, "too short-context"):
            prepare_gen_value_mc_sft.require_prefix_length_coverage(
                raw_examples, long_prefix_token_threshold=2048, min_long_prefix_fraction=0.5
            )

    def test_mc_sft_audits_final_action_continuation_shift_before_replay(self):
        raw_examples = [
            {
                "state_kind": "final_action",
                "response_tokens_used": 2,
                "target": 0.75,
                "source_rollout_outcomes": [0.0],
            },
            {
                "state_kind": "final_action",
                "response_tokens_used": 3072,
                "target": 0.5,
                "source_rollout_outcomes": [1.0],
            },
            {"state_kind": "final_action", "response_tokens_used": 4096, "target": 1.0, "source_rollout_outcomes": []},
            {"state_kind": "segment_start", "response_tokens_used": 3, "target": 0.0},
        ]

        summary = prepare_gen_value_mc_sft.summarize_final_action_continuation_shift(
            raw_examples, long_prefix_token_threshold=2048
        )

        self.assertEqual(summary["final_action_examples"], 3)
        self.assertEqual(summary["final_action_examples_with_sampled_outcomes"], 2)
        self.assertEqual(summary["final_action_source_outcomes"], 2)
        self.assertEqual(summary["final_action_prefix_lt_1024"], 1)
        self.assertEqual(summary["final_action_prefix_ge_2048"], 2)
        self.assertEqual(summary["final_action_prefix_ge_long_threshold"], 2)
        self.assertEqual(summary["continuation_minus_sampled_outcome_mean"], 0.125)
        self.assertEqual(summary["continuation_sampled_outcome_mae"], 0.625)
        self.assertEqual(summary["continuation_sampled_outcome_abs_gap_gt_0_25"], 2)
        self.assertEqual(summary["high_value_after_failed_sample"], 1)
        self.assertEqual(summary["low_value_after_correct_sample"], 0)

    def test_mc_sft_target_position_balance_is_deterministic_and_does_not_duplicate_states(self):
        examples = []
        identifier = 0
        for trajectory_fraction, target_counts in (
            (0.1, ((0.0, 4), (0.3, 1), (0.6, 1), (0.9, 1))),
            (0.5, ((0.0, 2), (0.3, 1), (0.6, 1), (0.9, 1))),
        ):
            for target, count in target_counts:
                for _ in range(count):
                    examples.append(
                        {
                            "id": identifier,
                            "prompt": f"state-{identifier}",
                            "state_kind": "segment_start",
                            "trajectory_fraction": trajectory_fraction,
                            "target": target,
                        }
                    )
                    identifier += 1
        examples.extend(
            [
                {
                    "id": identifier + offset,
                    "prompt": f"final-{offset}",
                    "state_kind": "final_action",
                    "trajectory_fraction": 1.0,
                    "target": float(offset),
                }
                for offset in range(2)
            ]
        )

        balanced = prepare_gen_value_mc_sft.balance_examples_by_target_and_position(examples, seed=7)

        self.assertEqual(balanced, prepare_gen_value_mc_sft.balance_examples_by_target_and_position(examples, seed=7))
        self.assertEqual(len(balanced), 10)
        self.assertEqual(len({example["prompt"] for example in balanced}), len(balanced))
        self.assertEqual(sum(example["state_kind"] == "final_action" for example in balanced), 2)
        cell_counts = {}
        for example in balanced:
            cell = prepare_gen_value_mc_sft._target_position_cell(example)
            if cell is not None:
                cell_counts[cell] = cell_counts.get(cell, 0) + 1
        self.assertEqual(set(cell_counts.values()), {1})
        self.assertEqual(len(cell_counts), 8)
        self.assertTrue(all(example["target_position_balanced"] for example in balanced))

    def test_mc_sft_pools_independent_targets_for_shared_exact_states(self):
        examples = prepare_gen_value_mc_sft.build_mc_sft_examples(
            [
                {
                    "problem": "Compute the answer.",
                    "rollout_is_correct": True,
                    "rollout_tokens": [10, 11, 12],
                    "probe_positions": [0, 2],
                    "mc_values": [0.25, 1.0],
                    "num_continuations": 16,
                },
                {
                    "problem": "Compute the answer.",
                    "rollout_is_correct": False,
                    "rollout_tokens": [10, 13],
                    "probe_positions": [0, 1],
                    "mc_values": [0.5, 0.0],
                    "num_continuations": 32,
                },
            ],
            tokenizer=_FakeTokenizer(),
            min_continuations=16,
        )

        self.assertEqual(len(examples), 3)
        pooled = examples[0]
        self.assertAlmostEqual(pooled["target"], (0.25 * 16 + 0.5 * 32) / 48)
        self.assertEqual(pooled["generation"], " <answer>4</answer>")
        self.assertEqual(pooled["num_continuations"], 48)
        self.assertEqual(pooled["mc_source_count"], 2)
        self.assertEqual(pooled["source_rollout_lengths"], [3, 2])
        self.assertEqual(pooled["source_rollout_outcomes"], [1.0, 0.0])
        self.assertEqual(pooled["source_trajectory_fractions"], [0.0, 0.0])

    def test_mc_sft_rejects_prompt_collisions_with_inconsistent_metadata(self):
        with self.assertRaisesRegex(ValueError, "inconsistent metadata fields"):
            prepare_gen_value_mc_sft.build_mc_sft_examples(
                [
                    {
                        "problem": "First problem identity.",
                        "prompt_token_ids": [1],
                        "rollout_tokens": [10],
                        "probe_positions": [0],
                        "mc_values": [0.25],
                        "num_continuations": 16,
                    },
                    {
                        "problem": "Second problem identity.",
                        "prompt_token_ids": [1],
                        "rollout_tokens": [11],
                        "probe_positions": [0],
                        "mc_values": [0.5],
                        "num_continuations": 16,
                    },
                ],
                tokenizer=_FakeTokenizer(),
                min_continuations=16,
            )

    def test_mc_sft_uses_the_exact_decoded_actor_prompt_as_the_problem(self):
        tokenizer = _FakeTokenizer()

        examples = prepare_gen_value_mc_sft.build_mc_sft_examples(
            [
                {
                    "problem": "Plain problem identity.",
                    "prompt_token_ids": [1, 2, 3],
                    "rollout_tokens": [10, 11],
                    "probe_positions": [1],
                    "mc_values": [0.25],
                    "num_continuations": 16,
                }
            ],
            tokenizer=tokenizer,
            min_continuations=16,
        )

        self.assertIn("Problem:\n1:2:3\n\n", examples[0]["prompt"])
        self.assertNotIn("Problem:\nPlain problem identity.\n\n", examples[0]["prompt"])
        self.assertEqual(examples[0]["problem"], "Plain problem identity.")
        self.assertEqual(examples[0]["critic_problem"], "1:2:3")
        self.assertEqual(tokenizer.skip_special_tokens_calls, [True, False])

    def test_mc_sft_can_condition_on_the_reference_answer(self):
        examples = prepare_gen_value_mc_sft.build_mc_sft_examples(
            [
                {
                    "problem": "Compute the answer.",
                    "ground_truth": "42",
                    "rollout_tokens": [10, 11],
                    "probe_positions": [1],
                    "mc_values": [0.0],
                    "num_continuations": 16,
                }
            ],
            tokenizer=_FakeTokenizer(),
            min_continuations=16,
            gen_value_conditioning="gt",
        )

        self.assertIn("The correct answer is 42.", examples[0]["prompt"])
        self.assertEqual(examples[0]["ground_truth"], "42")
        self.assertEqual(examples[0]["gen_value_conditioning"], "gt")
        self.assertTrue(synthesize_gen_value_sft.prompt_has_ground_truth_conditioning(examples[0]["prompt"]))

    def test_mc_sft_rejects_missing_reference_answer(self):
        with self.assertRaisesRegex(ValueError, "no ground truth"):
            prepare_gen_value_mc_sft.build_mc_sft_examples(
                [
                    {
                        "problem": "Compute the answer.",
                        "rollout_tokens": [10],
                        "probe_positions": [0],
                        "mc_values": [0.0],
                        "num_continuations": 16,
                    }
                ],
                tokenizer=_FakeTokenizer(),
                min_continuations=16,
                gen_value_conditioning="gt",
            )

    def test_mc_sft_rejects_invalid_horizon_repeat_configuration(self):
        with self.assertRaisesRegex(ValueError, "positive integers"):
            prepare_gen_value_mc_sft.repeat_examples_for_horizon(
                [], final_action_repeat=0, late_state_repeat=1, late_state_fraction=0.75
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            prepare_gen_value_mc_sft.repeat_examples_for_horizon(
                [], final_action_repeat=1, late_state_repeat=1, late_state_fraction=1.1
            )

    def test_sft_audit_recognizes_only_complete_declared_horizon_repeats(self):
        repeated = [
            {
                "prompt": "same state",
                "generation": " <answer>0</answer>",
                "horizon_repeat_count": 3,
                "horizon_repeat_index": index,
            }
            for index in range(3)
        ]

        self.assertTrue(synthesize_gen_value_sft.is_declared_horizon_repeat_group(repeated))
        self.assertFalse(synthesize_gen_value_sft.is_declared_horizon_repeat_group(repeated[:-1]))
        repeated[-1]["generation"] = " <answer>1</answer>"
        self.assertFalse(synthesize_gen_value_sft.is_declared_horizon_repeat_group(repeated))

    def test_mc_sft_rejects_under_sampled_targets(self):
        with self.assertRaisesRegex(ValueError, "only 8 continuations"):
            prepare_gen_value_mc_sft.build_mc_sft_examples(
                [
                    {
                        "problem": "Compute the answer.",
                        "rollout_tokens": [10],
                        "probe_positions": [0],
                        "mc_values": [0.5],
                        "num_continuations": 8,
                    }
                ],
                tokenizer=_FakeTokenizer(),
                min_continuations=16,
            )

    def test_problem_exclusion_happens_before_sampling(self):
        dataset = [
            {"messages": [{"role": "user", "content": "held out"}]},
            {"messages": [{"role": "user", "content": "training one"}]},
            {"prompt": "training two"},
        ]

        indices = value_estimation._sample_record_indices(
            dataset, num_to_sample=3, seed=7, excluded_problems={"held out"}
        )

        self.assertCountEqual(indices, [1, 2])

    def test_problem_exclusion_ignores_formatting_only_whitespace(self):
        dataset = [{"problem": "held  out\nproblem  "}, {"problem": "training problem"}]

        indices = value_estimation._sample_record_indices(
            dataset, num_to_sample=2, seed=7, excluded_problems={"held out problem"}
        )

        self.assertEqual(indices, [1])

    def test_mc_sft_overlap_guard_ignores_formatting_only_whitespace(self):
        overlaps = prepare_gen_value_mc_sft.normalized_problem_overlaps(
            [{"problem": "held  out\nproblem  "}, {"problem": "training problem"}], {"held out problem"}
        )

        self.assertEqual(overlaps, ["held out problem"])

    def test_extract_problem_reads_local_problem_column(self):
        self.assertEqual(
            value_estimation._extract_problem({"problem": "held-out question", "ground_truth": "42"}),
            "held-out question",
        )

    def test_problem_exclusion_fails_when_nothing_remains(self):
        with self.assertRaisesRegex(ValueError, "No dataset rows remain"):
            value_estimation._sample_record_indices(
                [{"prompt": "held\nout  "}], num_to_sample=1, seed=7, excluded_problems={"held out"}
            )

    def test_source_dataset_loads_hub_dataset_by_name(self):
        loader = mock.Mock(return_value="dataset")

        result = value_estimation._load_source_dataset("org/dataset", "train", loader)

        self.assertEqual(result, "dataset")
        loader.assert_called_once_with("org/dataset", split="train")

    def test_source_dataset_loads_local_parquet_file(self):
        loader = mock.Mock(return_value="dataset")
        dataset_path = "/tmp/heldout-problems.parquet"
        with mock.patch.object(value_estimation.pathlib.Path, "is_file", return_value=True):
            result = value_estimation._load_source_dataset(dataset_path, "train", loader)

        self.assertEqual(result, "dataset")
        loader.assert_called_once_with("parquet", data_files=dataset_path, split="train")

    def test_source_dataset_loads_local_jsonl_file(self):
        loader = mock.Mock(return_value="dataset")
        dataset_path = "/tmp/heldout-problems.jsonl"
        with mock.patch.object(value_estimation.pathlib.Path, "is_file", return_value=True):
            result = value_estimation._load_source_dataset(dataset_path, "validation", loader)

        self.assertEqual(result, "dataset")
        loader.assert_called_once_with("json", data_files=dataset_path, split="validation")

    def test_source_dataset_rejects_unsupported_local_file(self):
        loader = mock.Mock()
        with (
            mock.patch.object(value_estimation.pathlib.Path, "is_file", return_value=True),
            self.assertRaisesRegex(ValueError, "Unsupported local dataset format"),
        ):
            value_estimation._load_source_dataset("/tmp/heldout-problems.csv", "train", loader)

        loader.assert_not_called()

    def test_generative_scorer_default_matches_online_reasoning_budget(self):
        self.assertEqual(
            value_estimation.ScoreDatasetConfig.__dataclass_fields__["gen_value_max_new_tokens"].default, 1024
        )

    def test_mc_dataset_defaults_to_one_data_parallel_replica(self):
        self.assertEqual(value_estimation.MakeDatasetConfig.__dataclass_fields__["data_parallel_size"].default, 1)

    def test_dense_data_parallel_replicas_get_disjoint_cuda_device_groups(self):
        self.assertEqual(
            value_estimation._cuda_device_groups(
                data_parallel_size=2, tensor_parallel_size=2, visible_devices="4,7,2,9"
            ),
            ["4,7", "2,9"],
        )

    def test_dense_data_parallel_replicas_require_enough_visible_devices(self):
        with self.assertRaisesRegex(ValueError, "Need 4 visible CUDA devices"):
            value_estimation._cuda_device_groups(data_parallel_size=2, tensor_parallel_size=2, visible_devices="4,7,2")

    def test_dense_data_parallel_replicas_isolate_compile_caches(self):
        with mock.patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "4,7",
                "SLURM_JOB_ID": "123",
                "VLLM_CACHE_ROOT": "/cache/vllm",
                "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor",
                "TRITON_CACHE_DIR": "/cache/triton",
                "CUDA_CACHE_PATH": "/cache/cuda",
                "XDG_CACHE_HOME": "/cache/xdg",
                "TMPDIR": "/cache/tmp",
            },
        ):
            with mock.patch.object(value_estimation.pathlib.Path, "mkdir"):
                value_estimation._configure_data_replica_environment(1, "7")

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")
            self.assertEqual(os.environ["VLLM_CACHE_ROOT"], "/cache/vllm/job-123-replica-1")
            self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], "/cache/torchinductor/job-123-replica-1")
            self.assertEqual(os.environ["TRITON_CACHE_DIR"], "/cache/triton/job-123-replica-1")
            self.assertEqual(os.environ["CUDA_CACHE_PATH"], "/cache/cuda/job-123-replica-1")
            self.assertEqual(os.environ["XDG_CACHE_HOME"], "/cache/xdg/job-123-replica-1")
            self.assertEqual(os.environ["TMPDIR"], "/cache/tmp/job-123-replica-1")

    def test_mc_dataset_actor_identity_can_differ_from_rollout_checkpoint(self):
        config = value_estimation.MakeDatasetConfig(
            model_name_or_path="/checkpoints/step_100",
            output_path="values.parquet",
            actor_model_name="Qwen/Qwen3-4B-Base",
        )

        self.assertEqual(config.actor_model_name, "Qwen/Qwen3-4B-Base")

    def test_optional_float_cli_field_is_parsed_as_float(self):
        parser = argparse.ArgumentParser()
        field = next(
            field
            for field in dataclasses.fields(value_estimation.ScoreDatasetConfig)
            if field.name == "gen_value_actor_success_rate"
        )
        value_estimation._add_field(parser, field)

        args = parser.parse_args(["--gen_value_actor_success_rate", "0.125"])

        self.assertEqual(args.gen_value_actor_success_rate, 0.125)

    def test_actor_state_uses_exact_token_prefix(self):
        self.assertEqual(value_estimation._actor_state_token_ids([1, 2], [3, 4, 5], 2), [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            value_estimation._actor_state_token_ids([1], [2], 2)

    def test_full_continuation_includes_observed_prefix(self):
        tokenizer = _FakeTokenizer()
        decoded = value_estimation._decode_full_continuation(tokenizer, [10, 11], [12, 13])
        self.assertEqual(decoded, "10:11:12:13")
        self.assertEqual(tokenizer.skip_special_tokens_calls, [True])

    def test_mc_continuations_are_reduced_to_values_before_replica_ipc(self):
        tokenizer = _FakeTokenizer()
        results = [[{"token_ids": [12]}, {"token_ids": [13]}], [{"token_ids": [22]}, {"token_ids": [23]}]]
        metadata = [([10, 11], "first", "math"), ([20, 21], "second", "math")]

        with mock.patch.object(
            value_estimation,
            "_verify",
            side_effect=lambda response, ground_truth, _verifier: response.endswith("12")
            or (ground_truth == "second" and response.endswith("23")),
        ):
            reduced = value_estimation._reduce_mc_continuation_results(
                results, metadata, tokenizer=tokenizer, keep_continuation_texts=True
            )

        self.assertEqual([item["mc_value"] for item in reduced], [0.5, 0.5])
        self.assertEqual(
            [item["continuation_texts"] for item in reduced], [["10:11:12", "10:11:13"], ["20:21:22", "20:21:23"]]
        )
        self.assertEqual(tokenizer.skip_special_tokens_calls, [True] * 4)

    def test_mc_continuation_metadata_must_align_with_prompts(self):
        with self.assertRaisesRegex(ValueError, "one item per prompt"):
            value_estimation._run_rollouts(
                ["one", "two"],
                model_name_or_path="model",
                n=2,
                temperature=1.0,
                top_p=1.0,
                max_tokens=8,
                tensor_parallel_size=1,
                data_parallel_size=1,
                gpu_memory_utilization=0.9,
                mc_continuation_metadata=[([], "answer", "math")],
            )

    def test_sampled_eos_does_not_define_remaining_horizon(self):
        positions = value_estimation._fixed_probe_positions(
            rollout_length=1000,
            response_token_limit=8192,
            probe_interval=1000,
            min_remaining_tokens=64,
            max_probes=16,
            include_final_action_probe=True,
        )
        self.assertEqual(positions, [999])

    def test_make_dataset_rejects_incomplete_balanced_panel(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"found 1 prompts.*screening 64 prompts.*target_num_pairs=48.*success rate was 0\.002.*No partial",
        ):
            value_estimation._require_target_num_pairs(
                paired_count=1, target_num_pairs=48, screened_prompts=64, actor_success_rate=1 / 512
            )

        value_estimation._require_target_num_pairs(
            paired_count=48, target_num_pairs=48, screened_prompts=512, actor_success_rate=0.16
        )

    def test_near_budget_final_state_uses_its_true_remaining_budget(self):
        positions = value_estimation._fixed_probe_positions(
            rollout_length=8190,
            response_token_limit=8192,
            probe_interval=1000,
            min_remaining_tokens=64,
            max_probes=16,
            include_final_action_probe=True,
        )
        self.assertEqual(positions[-1], 8189)

    def test_rollout_cannot_exceed_response_budget(self):
        with self.assertRaisesRegex(ValueError, "no larger than"):
            value_estimation._fixed_probe_positions(
                rollout_length=8193,
                response_token_limit=8192,
                probe_interval=1000,
                min_remaining_tokens=64,
                max_probes=16,
                include_final_action_probe=True,
            )

    def test_probe_cap_preserves_latest_state(self):
        positions = value_estimation._fixed_probe_positions(
            rollout_length=8000,
            response_token_limit=8192,
            probe_interval=100,
            min_remaining_tokens=64,
            max_probes=4,
            include_final_action_probe=True,
        )
        self.assertEqual(len(positions), 4)
        self.assertEqual(positions[-1], 7999)

    def test_fraction_probes_cover_trajectory_and_preserve_final_action(self):
        positions = value_estimation._fraction_probe_positions(
            rollout_length=1000,
            response_token_limit=8192,
            probe_fractions=value_estimation._parse_probe_fractions("0,0.25,0.5,0.75"),
            include_final_action_probe=True,
        )

        self.assertEqual(positions, [0, 250, 500, 749, 999])

    def test_fraction_probes_deduplicate_short_trajectory_states(self):
        positions = value_estimation._fraction_probe_positions(
            rollout_length=2,
            response_token_limit=8192,
            probe_fractions=[0.0, 0.25, 0.5, 0.75, 1.0],
            include_final_action_probe=True,
        )

        self.assertEqual(positions, [0, 1])

    def test_fraction_probes_reject_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            value_estimation._parse_probe_fractions("")
        with self.assertRaisesRegex(ValueError, "nonnumeric"):
            value_estimation._parse_probe_fractions("0,quarter")
        with self.assertRaisesRegex(ValueError, "finite and in"):
            value_estimation._fraction_probe_positions(
                rollout_length=100,
                response_token_limit=8192,
                probe_fractions=[0.0, 1.1],
                include_final_action_probe=True,
            )

    def test_numpy_correlations_do_not_require_scipy(self):
        self.assertAlmostEqual(value_estimation._pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(value_estimation._spearman_correlation([1, 2, 2, 3], [3, 2, 2, 1]), -1.0)

    def test_constant_correlation_is_not_finite(self):
        self.assertTrue(math.isnan(value_estimation._pearson_correlation([1, 1], [0, 1])))

    def test_parquet_array_column_is_normalized_without_truth_testing(self):
        self.assertEqual(value_estimation._optional_sequence_as_list(np.array(["a", "b"])), ["a", "b"])
        self.assertEqual(value_estimation._optional_sequence_as_list(None), [])
        self.assertEqual(value_estimation._optional_sequence_as_list(float("nan")), [])

    def test_prediction_group_metrics_penalize_parse_failures(self):
        metrics = value_estimation._prediction_group_metrics(
            [0.1, None, float("nan"), 0.9], [0.0, 1.0, 0.0, 1.0], prefix="final_action"
        )

        self.assertEqual(metrics["final_action_examples"], 4.0)
        self.assertAlmostEqual(metrics["final_action_parse_rate"], 0.5)
        self.assertAlmostEqual(metrics["final_action_target_mean"], 0.5)
        self.assertAlmostEqual(metrics["final_action_penalized_mse"], (0.01 + 1.0 + 1.0 + 0.01) / 4)
        self.assertAlmostEqual(metrics["final_action_pred_mean"], 0.5)
        self.assertAlmostEqual(metrics["final_action_parsed_target_mean"], 0.5)
        self.assertAlmostEqual(metrics["final_action_mse"], 0.01)

    def test_prediction_group_metrics_reject_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "differ in length"):
            value_estimation._prediction_group_metrics([0.1], [], prefix="broken")

    def test_bucketed_prediction_metrics_separates_position_bands_and_penalizes_parse_failures(self):
        metrics = value_estimation._bucketed_prediction_metrics(
            [0.1, 0.5, None, 0.9], [0.0, 0.5, 1.0, 1.0], [0.0, 0.25, 0.5, 1.0], prefix="trajectory"
        )

        self.assertEqual(metrics["trajectory_early_examples"], 1.0)
        self.assertAlmostEqual(metrics["trajectory_early_mse"], 0.01)
        self.assertEqual(metrics["trajectory_middle_examples"], 2.0)
        self.assertAlmostEqual(metrics["trajectory_middle_parse_rate"], 0.5)
        self.assertAlmostEqual(metrics["trajectory_middle_penalized_mse"], 0.5)
        self.assertAlmostEqual(metrics["trajectory_middle_mc_mean"], 0.5)
        self.assertEqual(metrics["trajectory_late_examples"], 1.0)
        self.assertAlmostEqual(metrics["trajectory_late_mse"], 0.01)

    def test_bucketed_prediction_metrics_rejects_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            value_estimation._bucketed_prediction_metrics([0.1], [0.0], [], prefix="trajectory")

    def test_absolute_prefix_metrics_use_fixed_token_bands_and_penalize_parse_failures(self):
        metrics = value_estimation._bucketed_absolute_prefix_metrics(
            [0.1, 0.5, None, 0.9], [0.0, 0.5, 1.0, 1.0], [100, 1024, 2048, 4096]
        )

        self.assertEqual(metrics["prefix_tokens_lt_1024_examples"], 1.0)
        self.assertAlmostEqual(metrics["prefix_tokens_lt_1024_mse"], 0.01)
        self.assertEqual(metrics["prefix_tokens_1024_2047_examples"], 1.0)
        self.assertAlmostEqual(metrics["prefix_tokens_1024_2047_mc_mean"], 0.5)
        self.assertEqual(metrics["prefix_tokens_2048_4095_parse_rate"], 0.0)
        self.assertEqual(metrics["prefix_tokens_2048_4095_penalized_mse"], 1.0)
        self.assertEqual(metrics["prefix_tokens_ge_4096_examples"], 1.0)
        self.assertAlmostEqual(metrics["prefix_tokens_ge_4096_mse"], 0.01)

    def test_absolute_prefix_metrics_reject_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            value_estimation._bucketed_absolute_prefix_metrics([0.1], [0.0], [])


class TestGenerativeValueScoreComparison(unittest.TestCase):
    def test_clustered_delta_summary_uses_problem_means(self):
        summary = compare_gen_value_scores._clustered_delta_summary(
            {"a": [0.1, 0.3], "b": [-0.2]}, bootstrap_samples=100, rng=np.random.default_rng(0)
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary["problems"], 2)
        self.assertAlmostEqual(summary["problem_balanced_mse_delta_candidate_minus_baseline"], 0.0)
        self.assertEqual(len(summary["problem_cluster_bootstrap_95pct_ci"]), 2)

    def test_metrics_report_score_scale_coverage_and_calibration(self):
        rows = [
            {
                "problem": "a",
                "rollout_is_correct": True,
                "state_kind": "intermediate",
                "trajectory_band": "early",
                "absolute_prefix_band": "lt_1024",
                "target": 0.0,
                "prediction": 0.3,
            },
            {
                "problem": "a",
                "rollout_is_correct": False,
                "state_kind": "intermediate",
                "trajectory_band": "early",
                "absolute_prefix_band": "lt_1024",
                "target": 0.1,
                "prediction": 0.3,
            },
            {
                "problem": "b",
                "rollout_is_correct": True,
                "state_kind": "intermediate",
                "trajectory_band": "late",
                "absolute_prefix_band": "2048_4095",
                "target": 1.0,
                "prediction": 0.9,
            },
            {
                "problem": "b",
                "rollout_is_correct": False,
                "state_kind": "intermediate",
                "trajectory_band": "late",
                "absolute_prefix_band": "2048_4095",
                "target": 0.5,
                "prediction": None,
            },
        ]

        metrics = compare_gen_value_scores._metrics(rows)

        self.assertEqual(metrics["prediction_decile_coverage"], 2.0)
        self.assertEqual(metrics["prediction_decile_3_examples"], 2.0)
        self.assertEqual(metrics["prediction_decile_9_examples"], 1.0)
        self.assertAlmostEqual(metrics["target_decile_0_calibration_bias"], 0.3)
        self.assertAlmostEqual(metrics["target_decile_1_calibration_bias"], 0.2)
        self.assertAlmostEqual(metrics["target_decile_10_calibration_bias"], -0.1)
        self.assertEqual(metrics["target_decile_5_penalized_mse"], 1.0)

    def test_score_scale_metrics_reject_out_of_range_values(self):
        with self.assertRaisesRegex(ValueError, "normalized score"):
            compare_gen_value_scores._score_scale_metrics([{"target": 0.0, "prediction": 1.1}])

    def test_auc_is_problem_balanced_and_restricted_to_real_selection_pairs(self):
        rows = [
            {
                "problem": "a",
                "rollout_is_correct": True,
                "state_kind": "intermediate",
                "trajectory_band": "early",
                "absolute_prefix_band": "lt_1024",
                "target": 0.2,
                "prediction": 0.6,
            },
            {
                "problem": "a",
                "rollout_is_correct": False,
                "state_kind": "intermediate",
                "trajectory_band": "early",
                "absolute_prefix_band": "lt_1024",
                "target": 0.8,
                "prediction": 0.5,
            },
            {
                "problem": "b",
                "rollout_is_correct": True,
                "state_kind": "intermediate",
                "trajectory_band": "early",
                "absolute_prefix_band": "2048_4095",
                "target": 0.9,
                "prediction": 0.2,
            },
            {
                "problem": "b",
                "rollout_is_correct": False,
                "state_kind": "intermediate",
                "trajectory_band": "early",
                "absolute_prefix_band": "2048_4095",
                "target": 0.1,
                "prediction": 0.1,
            },
        ]

        metrics = compare_gen_value_scores._metrics(rows)

        # The pooled metric compares cross-problem pairs and loses one of four
        # comparisons. Both comparisons the policy could actually make are
        # correctly ranked.
        self.assertAlmostEqual(metrics["intermediate_outcome_auc"], 0.75)
        self.assertAlmostEqual(metrics["intermediate_within_problem_auc"], 1.0)
        self.assertEqual(metrics["intermediate_within_problem_auc_problems"], 2.0)
        self.assertEqual(metrics["intermediate_within_problem_auc_pairs"], 2.0)
        # Eventual rollout correctness is not the critic's value target. The
        # first problem ranks the correct rollout higher even though its exact
        # continuation value is lower, which the decision metric catches.
        self.assertAlmostEqual(metrics["intermediate_mc_selection_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["intermediate_mc_selection_regret"], 0.3)
        self.assertAlmostEqual(metrics["intermediate_mc_selection_gain_over_random"], 0.05)
        self.assertEqual(metrics["intermediate_mc_selection_problems"], 2.0)
        self.assertEqual(metrics["intermediate_mc_selection_pairs"], 2.0)
        self.assertEqual(metrics["absolute_prefix_lt_1024_examples"], 2.0)
        self.assertAlmostEqual(metrics["absolute_prefix_lt_1024_mc_selection_accuracy"], 0.0)
        self.assertEqual(metrics["absolute_prefix_2048_4095_examples"], 2.0)
        self.assertAlmostEqual(metrics["absolute_prefix_2048_4095_mc_selection_accuracy"], 1.0)

    def test_long_prefix_gate_cannot_hide_intermediate_regression_with_final_action_gains(self):
        with tempfile.TemporaryDirectory(prefix="gen-value-long-prefix-gate-") as directory:
            test_root = pathlib.Path(directory)
            baseline_path = test_root / "baseline.parquet"
            candidate_path = test_root / "candidate.parquet"
            shared = {"problem": "one problem", "rollout_tokens": list(range(4097)), "probe_positions": [2048, 4096]}
            pd.DataFrame(
                [
                    {**shared, "rollout_is_correct": True, "mc_values": [0.8, 1.0], "predicted_values": [0.8, 0.0]},
                    {**shared, "rollout_is_correct": False, "mc_values": [0.2, 0.0], "predicted_values": [0.2, 1.0]},
                ]
            ).to_parquet(baseline_path, index=False)
            pd.DataFrame(
                [
                    {**shared, "rollout_is_correct": True, "mc_values": [0.8, 1.0], "predicted_values": [1.0, 1.0]},
                    {**shared, "rollout_is_correct": False, "mc_values": [0.2, 0.0], "predicted_values": [0.0, 0.0]},
                ]
            ).to_parquet(candidate_path, index=False)

            comparison = compare_gen_value_scores.compare_scores(
                baseline_path,
                candidate_path,
                bootstrap_samples=100,
                seed=0,
                mse_noninferiority_margin=0.01,
                auc_noninferiority_margin=0.02,
            )

        # Easy final actions make the aggregate long-prefix score look much
        # better even though both intermediate values become worse.
        self.assertLess(
            comparison["absolute_prefix_mse_deltas"]["ge_2048"]["problem_balanced_mse_delta_candidate_minus_baseline"],
            0.0,
        )
        self.assertAlmostEqual(
            comparison["absolute_prefix_intermediate_mse_deltas"]["ge_2048"][
                "problem_balanced_mse_delta_candidate_minus_baseline"
            ],
            0.04,
        )
        self.assertFalse(comparison["gate"]["checks"]["long_prefix_intermediate_mse_noninferior"])


if __name__ == "__main__":
    unittest.main()
