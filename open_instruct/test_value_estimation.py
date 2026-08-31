import argparse
import dataclasses
import math
import os
import unittest
from unittest import mock

import numpy as np
from scripts.data import prepare_gen_value_mc_sft, prepare_gen_value_sft, synthesize_gen_value_sft

from open_instruct import value_estimation


class _FakeTokenizer:
    def __init__(self):
        self.skip_special_tokens_calls = []

    def decode(self, token_ids, skip_special_tokens=True):
        self.skip_special_tokens_calls.append(skip_special_tokens)
        return ":".join(str(token_id) for token_id in token_ids)


class TestValueEstimationStates(unittest.TestCase):
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

    def test_mc_sft_pools_independent_targets_for_shared_exact_states(self):
        examples = prepare_gen_value_mc_sft.build_mc_sft_examples(
            [
                {
                    "problem": "Compute the answer.",
                    "rollout_tokens": [10, 11, 12],
                    "probe_positions": [0, 2],
                    "mc_values": [0.25, 1.0],
                    "num_continuations": 16,
                },
                {
                    "problem": "Compute the answer.",
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

    def test_problem_exclusion_fails_when_nothing_remains(self):
        with self.assertRaisesRegex(ValueError, "No dataset rows remain"):
            value_estimation._sample_record_indices(
                [{"prompt": "held out"}], num_to_sample=1, seed=7, excluded_problems={"held out"}
            )

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


if __name__ == "__main__":
    unittest.main()
