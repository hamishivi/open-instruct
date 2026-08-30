import unittest

from open_instruct import value_estimation


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens
        return ":".join(str(token_id) for token_id in token_ids)


class TestValueEstimationStates(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
