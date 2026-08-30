import argparse
import dataclasses
import math
import unittest

import numpy as np

from open_instruct import value_estimation


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens
        return ":".join(str(token_id) for token_id in token_ids)


class TestValueEstimationStates(unittest.TestCase):
    def test_generative_scorer_default_matches_online_reasoning_budget(self):
        self.assertEqual(
            value_estimation.ScoreDatasetConfig.__dataclass_fields__["gen_value_max_new_tokens"].default, 1024
        )

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
        decoded = value_estimation._decode_full_continuation(_FakeTokenizer(), [10, 11], [12, 13])
        self.assertEqual(decoded, "10:11:12:13")

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

    def test_near_budget_final_state_requires_room_to_continue(self):
        positions = value_estimation._fixed_probe_positions(
            rollout_length=8190,
            response_token_limit=8192,
            probe_interval=1000,
            min_remaining_tokens=64,
            max_probes=16,
            include_final_action_probe=True,
        )
        self.assertEqual(positions[-1], 8000)
        self.assertNotIn(8189, positions)

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


if __name__ == "__main__":
    unittest.main()
