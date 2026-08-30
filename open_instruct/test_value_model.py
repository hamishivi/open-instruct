# Copyright 2026 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Smoke tests for the value-model code paths added on hamish/vip.

These tests deliberately avoid anything that requires vLLM, DeepSpeed, or a GPU so they run on a
laptop and in CI. They focus on:

(a) GAE variants (standard, SAE, VAPO, SAE+VAPO) on a tiny packed sequence;
(b) sibling-rollout assembly helpers in data_loader.py;
(c) the conditioning text builders in value_model_utils.py;
(d) score parsing for the generative value model.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from open_instruct import grpo_utils
from open_instruct.rl_utils import (
    PackedSequences,
    calculate_advantages_packed,
    calculate_advantages_packed_sae,
    calculate_advantages_packed_sae_vapo,
    calculate_advantages_packed_vapo,
    calculate_length_adaptive_lambda,
    estimate_sae_terminal_credit_retention,
)
from open_instruct.value_model_utils import (
    accumulation_group_token_counts,
    add_observation_segment_boundaries,
    balanced_accumulation_group_ids,
    bounded_value_prediction,
    build_conditioning_text,
    build_gen_value_validation_holdout,
    build_generative_value_prompt,
    causal_final_action_prefix_token_ids,
    causal_segment_start_prefix_token_ids,
    causal_value_mask,
    compute_value_loss,
    flatten_gen_value_pack,
    gen_value_validation_metrics,
    generative_value_reinforce_reward,
    grouped_token_counts,
    is_postfix_template,
    missing_value_fallback,
    normalize_value_loss,
    pack_gen_value_examples,
    parse_generative_value_score,
    predict_values,
    regression_metric_sums,
    regression_metrics_from_sums,
    resolve_num_siblings_to_sample,
    reward_to_unit_value,
    segment_rollout,
    select_gen_value_sft_traces,
    unit_value_to_reward,
    validate_terminal_rewards,
    value_clipped_mse_loss,
    value_metric_sums,
    value_metrics_from_sums,
    write_gen_value_training_trace_reservoir,
    write_gen_value_validation_snapshot,
)


def _packing_example(sequence_ids, generated_ids, rollout_logprobs, reward):
    return {
        "sequence_ids": sequence_ids,
        "generated_ids": generated_ids,
        "rollout_logprobs": rollout_logprobs,
        "reward": reward,
    }


class TestGenValuePacking(unittest.TestCase):
    def test_uses_policy_token_budget_without_truncation(self):
        examples = [
            _packing_example([1, 2, 3, 4], [4], [-0.1], 0.2),
            _packing_example([5, 6], [6], [-0.2], 0.4),
            _packing_example(list(range(7, 15)), [14], [-0.3], 0.6),
        ]

        packs = pack_gen_value_examples(examples, target_tokens=6)

        self.assertEqual([[len(example["sequence_ids"]) for example in pack] for pack in packs], [[4, 2], [8]])
        self.assertEqual(packs[1][0]["sequence_ids"], list(range(7, 15)))

    def test_resets_positions_and_selects_unequal_generated_tokens(self):
        examples = [
            _packing_example([10, 11, 12, 13, 14], [13, 14], [-0.1, -0.2], 0.25),
            _packing_example([20, 21, 22, 23, 24], [22, 23, 24], [-0.3, -0.4, -0.5], 0.75),
        ]

        flattened = flatten_gen_value_pack(examples)

        input_ids, position_ids, logit_positions, target_ids, rollout_logprobs, token_rewards = flattened
        self.assertEqual(input_ids, [10, 11, 12, 13, 14, 20, 21, 22, 23, 24])
        self.assertEqual(position_ids, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        self.assertEqual(logit_positions, [2, 3, 6, 7, 8])
        self.assertEqual(target_ids, [13, 14, 22, 23, 24])
        self.assertEqual(rollout_logprobs, [-0.1, -0.2, -0.3, -0.4, -0.5])
        self.assertEqual(token_rewards, [0.25, 0.25, 0.75, 0.75, 0.75])

    def test_packed_logits_match_isolated_sequences_with_unequal_token_counts(self):
        torch.manual_seed(1)
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            use_cache=False,
        )
        model = Qwen3ForCausalLM(config).eval()
        examples = [
            _packing_example([1, 2, 3, 4, 5], [4, 5], [-0.1, -0.2], 0.25),
            _packing_example([6, 7, 8, 9, 10, 11], [9, 10, 11], [-0.3, -0.4, -0.5], 0.75),
        ]

        serial_logits = []
        for example in examples:
            sequence_ids = torch.tensor([example["sequence_ids"]])
            generated_length = len(example["generated_ids"])
            prompt_length = sequence_ids.shape[1] - generated_length
            logit_positions = torch.arange(prompt_length - 1, sequence_ids.shape[1] - 1)
            serial_logits.append(
                model(
                    input_ids=sequence_ids,
                    attention_mask=None,
                    position_ids=torch.arange(sequence_ids.shape[1]).unsqueeze(0),
                    logits_to_keep=logit_positions,
                ).logits
            )

        input_ids, position_ids, logit_positions, *_ = flatten_gen_value_pack(examples)
        packed_logits = model(
            input_ids=torch.tensor([input_ids]),
            attention_mask=None,
            position_ids=torch.tensor([position_ids]),
            logits_to_keep=torch.tensor(logit_positions),
        ).logits

        torch.testing.assert_close(packed_logits, torch.cat(serial_logits, dim=1), rtol=1e-5, atol=1e-5)


class TestCausalGenValuePrefixes(unittest.TestCase):
    def test_observation_gap_starts_a_new_segment(self):
        response_mask = [False, True, False, False, True, True]

        boundaries = add_observation_segment_boundaries(response_mask, [2])

        self.assertEqual(boundaries, [0, 2])

    def test_observation_boundaries_merge_with_existing_segments(self):
        response_mask = [False, True, True, False, True, True, True]

        boundaries = add_observation_segment_boundaries(response_mask, [1, 4])

        self.assertEqual(boundaries, [1, 4])

    def test_scores_segment_starts_instead_of_segment_ends(self):
        # prompt=[10, 11], response actions=[20, 21, 22, 23, 24, 25]
        sequence = [10, 11, 20, 21, 22, 23, 24, 25]
        response_mask = [False, False, True, True, True, True, True, True]

        prefixes = causal_segment_start_prefix_token_ids(sequence, response_mask, [2, 5])

        self.assertEqual(prefixes, [[], [20, 21, 22]])

    def test_retains_masked_tool_observations_before_next_action(self):
        # The first segment contains action 20. Tokens 30 and 31 are a masked
        # tool observation that must remain visible before action 21.
        sequence = [10, 11, 20, 30, 31, 21, 22]
        response_mask = [False, False, True, False, False, True, True]

        prefixes = causal_segment_start_prefix_token_ids(sequence, response_mask, [0, 2])

        self.assertEqual(prefixes, [[], [20, 30, 31]])

    def test_requires_terminal_boundary(self):
        with self.assertRaisesRegex(ValueError, "final segment boundary"):
            causal_segment_start_prefix_token_ids([10, 20, 21], [False, True, True], [0])


class TestValueRewardBounds(unittest.TestCase):
    def test_missing_prediction_falls_back_to_zero_when_supported(self):
        self.assertEqual(missing_value_fallback(-2.0, 3.0), 0.0)

    def test_missing_prediction_uses_endpoint_closest_to_zero(self):
        self.assertEqual(missing_value_fallback(1.0, 3.0), 1.0)
        self.assertEqual(missing_value_fallback(-3.0, -1.0), -1.0)

    def test_accepts_terminal_rewards_inside_declared_support(self):
        rewards = torch.tensor([[0.0, -1.0, 0.0, 2.0]])
        dones = torch.tensor([[0, 1, 0, 1]])

        validate_terminal_rewards(rewards, dones, -1.0, 2.0)

    def test_rejects_terminal_rewards_outside_declared_support(self):
        rewards = torch.tensor([[0.0, 3.0]])
        dones = torch.tensor([[0, 1]])

        with self.assertRaisesRegex(ValueError, "outside the configured value range"):
            validate_terminal_rewards(rewards, dones, 0.0, 1.0)

    def test_ignores_nonterminal_values(self):
        rewards = torch.tensor([[100.0, 1.0]])
        dones = torch.tensor([[0, 1]])

        validate_terminal_rewards(rewards, dones, 0.0, 1.0)


class TestGAEVariants(unittest.TestCase):
    def _inputs(self):
        # Single packed sequence with one sub-sequence. Prompt tokens 0..2, response 3..7.
        # Reward of 1.0 at t=7 (terminal). One low-probability token at t=4 for SAE.
        B, T = 1, 8
        values = np.array([[0.1] * T])
        rewards = np.zeros((B, T))
        rewards[0, 7] = 1.0
        dones = np.zeros((B, T))
        dones[0, 7] = 1
        response_masks = np.zeros((B, T))
        response_masks[0, 3:8] = 1
        logprobs = np.array([[-0.1, -0.1, -0.1, -0.1, -2.5, -0.1, -0.1, -0.1]])
        return values, rewards, dones, response_masks, logprobs

    def test_standard_gae_runs(self):
        v, r, d, m, _ = self._inputs()
        adv, returns = calculate_advantages_packed(v, r, gamma=1.0, lam=0.95, dones=d, response_masks=m)
        self.assertEqual(adv.shape, v.shape)
        self.assertEqual(returns.shape, v.shape)
        # Terminal step should have a positive advantage (reward minus baseline value).
        self.assertGreater(adv[0, 7], 0)

    def test_vapo_has_two_outputs(self):
        v, r, d, m, _ = self._inputs()
        pa, cr, avg_lam = calculate_advantages_packed_vapo(v, r, gamma=1.0, dones=d, response_masks=m)
        self.assertEqual(pa.shape, v.shape)
        self.assertEqual(cr.shape, v.shape)
        self.assertEqual(avg_lam, 0.95)

    def test_sae_marks_boundary(self):
        v, r, d, m, logp = self._inputs()
        adv, returns, bf = calculate_advantages_packed_sae(
            v, r, gamma=1.0, lam=0.2, dones=d, response_masks=m, logprobs=logp, sae_threshold=0.2
        )
        # t=4 logprob < log(0.2) ≈ -1.609, so exactly one boundary among the response tokens.
        expected_frac = 1 / 5
        self.assertAlmostEqual(bf, expected_frac, places=6)
        self.assertEqual(adv.shape, v.shape)

    def test_sae_lambda_one_propagates_terminal_outcome_across_packed_sequences(self):
        # Two episodes share one packed row. With gamma=lambda=1 and no intermediate
        # rewards, every response action must receive exactly outcome - V(s),
        # regardless of SAE boundaries or intervening prompt tokens.
        values = np.array([[0.0, 0.0, 0.2, 0.4, 0.7, 0.0, 0.0, 0.3, 0.5, 0.6, 0.8]])
        rewards = np.zeros_like(values)
        rewards[0, 4] = 1.0
        dones = np.zeros_like(values)
        dones[0, [4, 10]] = 1.0
        response_masks = np.zeros_like(values)
        response_masks[0, [2, 3, 4, 7, 8, 9, 10]] = 1.0
        logprobs = np.full_like(values, -0.1)
        logprobs[0, [3, 8, 10]] = -2.5

        advantages, _, boundary_fraction = calculate_advantages_packed_sae(
            values,
            rewards,
            gamma=1.0,
            lam=1.0,
            dones=dones,
            response_masks=response_masks,
            logprobs=logprobs,
            sae_threshold=0.2,
        )

        np.testing.assert_allclose(advantages[0, [2, 3, 4]], 1.0 - values[0, [2, 3, 4]])
        np.testing.assert_allclose(advantages[0, [7, 8, 9, 10]], -values[0, [7, 8, 9, 10]])
        self.assertGreater(boundary_fraction, 0.0)

    def test_sae_vapo_combines_variants(self):
        v, r, d, m, logp = self._inputs()
        pa, cr, metrics = calculate_advantages_packed_sae_vapo(
            v, r, gamma=1.0, dones=d, response_masks=m, logprobs=logp, sae_threshold=0.2, lam_policy=0.5
        )
        self.assertEqual(pa.shape, v.shape)
        self.assertEqual(cr.shape, v.shape)
        self.assertGreater(metrics["value/sae_boundary_frac"], 0)
        self.assertEqual(metrics["value/sae_segments_mean"], 2.0)
        self.assertEqual(metrics["value/sae_boundary_lambda_mean"], 0.5)

    def test_length_adaptive_lambda(self):
        # alpha*length = 1 -> lambda = 0
        self.assertEqual(calculate_length_adaptive_lambda(1, alpha=1.0), 0.0)
        # alpha*length = 100 -> lambda close to 1
        self.assertGreater(calculate_length_adaptive_lambda(100, alpha=1.0), 0.98)

    def test_estimated_sae_terminal_credit_exposes_short_trace(self):
        self.assertEqual(estimate_sae_terminal_credit_retention(0.48, 1.0, 1024), 1.0)
        self.assertAlmostEqual(estimate_sae_terminal_credit_retention(0.48, 0.95, 128), 0.0446, places=3)
        self.assertLess(estimate_sae_terminal_credit_retention(0.48, 0.95, 1024), 1e-10)

    def test_skip_tool_outputs_bootstraps_across_observation_gap(self):
        # Prompt [0], action0 [1,2], tool [3,4], action1 [5,6] with terminal reward at t=6.
        # Values differ so skipping the tool gap changes the bootstrap at t=2.
        values = np.array([[0.0, 1.0, 2.0, 10.0, 10.0, 3.0, 4.0]], dtype=np.float64)
        rewards = np.zeros((1, 7), dtype=np.float64)
        rewards[0, 6] = 1.0
        dones = np.zeros((1, 7), dtype=np.float64)
        dones[0, 6] = 1.0
        response_masks = np.array([[0, 1, 1, 0, 0, 1, 1]], dtype=np.float64)

        adv_skip, _ = calculate_advantages_packed(
            values, rewards, gamma=1.0, lam=1.0, dones=dones, response_masks=response_masks
        )
        adv_std, _ = calculate_advantages_packed(
            values, rewards, gamma=1.0, lam=1.0, dones=dones, response_masks=response_masks, skip_tool_outputs=False
        )

        # Tool / prompt tokens get zero advantage when skipping.
        np.testing.assert_allclose(adv_skip[0, [0, 3, 4]], 0.0)
        # Last action token before the gap bootstraps to V(a_{i+1,0})=values[5]=3, not V(tool)=10.
        # With lam=1, gamma=1 and no intermediate rewards: A[2] = (0 + 3 - 2) + A[5].
        self.assertAlmostEqual(adv_skip[0, 2], (3.0 - 2.0) + adv_skip[0, 5], places=6)
        # Standard GAE would bootstrap onto the tool token value instead.
        self.assertNotAlmostEqual(adv_skip[0, 2], adv_std[0, 2], places=6)

    def test_skip_tool_outputs_matches_sequence_without_observations(self):
        values = np.array([[0.0, 0.2, 0.4, 9.0, 8.0, 0.6, 0.8]], dtype=np.float64)
        rewards = np.zeros((1, 7), dtype=np.float64)
        rewards[0, 6] = 1.0
        dones = np.zeros((1, 7), dtype=np.float64)
        dones[0, 6] = 1.0
        response_masks = np.array([[0, 1, 1, 0, 0, 1, 1]], dtype=np.float64)

        advantages, _ = calculate_advantages_packed(
            values, rewards, gamma=0.9, lam=0.5, dones=dones, response_masks=response_masks, skip_tool_outputs=True
        )

        action_indices = [1, 2, 5, 6]
        compact_values = values[:, action_indices]
        compact_rewards = rewards[:, action_indices]
        compact_dones = dones[:, action_indices]
        compact_mask = np.ones_like(compact_values)
        compact_advantages, _ = calculate_advantages_packed(
            compact_values,
            compact_rewards,
            gamma=0.9,
            lam=0.5,
            dones=compact_dones,
            response_masks=compact_mask,
            skip_tool_outputs=True,
        )

        np.testing.assert_allclose(advantages[:, action_indices], compact_advantages)
        np.testing.assert_allclose(advantages[:, [0, 3, 4]], 0.0)

    def test_sae_skip_tool_outputs_matches_sequence_without_observations(self):
        values = np.array([[0.0, 0.2, 0.4, 9.0, 8.0, 0.6, 0.8]], dtype=np.float64)
        rewards = np.zeros((1, 7), dtype=np.float64)
        rewards[0, 6] = 1.0
        dones = np.zeros((1, 7), dtype=np.float64)
        dones[0, 6] = 1.0
        response_masks = np.array([[0, 1, 1, 0, 0, 1, 1]], dtype=np.float64)
        logprobs = np.array([[-0.1, -0.1, -0.1, -0.1, -0.1, -2.5, -0.1]], dtype=np.float64)

        policy_advantages, critic_returns, metrics = calculate_advantages_packed_sae_vapo(
            values,
            rewards,
            gamma=0.9,
            dones=dones,
            response_masks=response_masks,
            logprobs=logprobs,
            sae_threshold=0.2,
            lam_policy=0.5,
            skip_tool_outputs=True,
        )

        action_indices = [1, 2, 5, 6]
        compact_policy_advantages, compact_critic_returns, compact_metrics = calculate_advantages_packed_sae_vapo(
            values[:, action_indices],
            rewards[:, action_indices],
            gamma=0.9,
            dones=dones[:, action_indices],
            response_masks=np.ones((1, len(action_indices)), dtype=np.float64),
            logprobs=logprobs[:, action_indices],
            sae_threshold=0.2,
            lam_policy=0.5,
            skip_tool_outputs=True,
        )

        np.testing.assert_allclose(policy_advantages[:, action_indices], compact_policy_advantages)
        np.testing.assert_allclose(critic_returns[:, action_indices], compact_critic_returns)
        np.testing.assert_allclose(policy_advantages[:, [0, 3, 4]], 0.0)
        self.assertEqual(metrics, compact_metrics)


class TestTISMask(unittest.TestCase):
    def test_cap_and_mask_match_policy_token_weighting(self):
        ratios = torch.tensor([[0.25, 1.0, 3.0]])
        trainer_logprobs = torch.log(ratios)
        rollout_logprobs = torch.zeros_like(trainer_logprobs)
        response_mask = torch.ones_like(trainer_logprobs, dtype=torch.bool)

        clamped, _ = grpo_utils.compute_tis_weights(
            trainer_logprobs.detach(), rollout_logprobs, response_mask, cap=2.0
        )
        mask = grpo_utils.compute_tis_mask(
            trainer_logprobs, rollout_logprobs, response_mask, lower_bound=0.5, upper_bound=2.0
        )
        combined = grpo_utils.combine_tis_terms(clamped, mask)

        torch.testing.assert_close(combined, torch.tensor([[0.0, 1.0, 0.0]]))

    def test_upper_and_lower_bounds(self):
        ratios = torch.tensor([[0.49, 0.5, 1.0, 1.99, 2.0, 3.0]])
        new_logprobs = torch.log(ratios)
        vllm_logprobs = torch.zeros_like(new_logprobs)
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)

        mask = grpo_utils.compute_tis_mask(
            new_logprobs, vllm_logprobs, response_mask, lower_bound=0.5, upper_bound=2.0
        )
        expected = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
        torch.testing.assert_close(mask, expected)

    def test_disabled_returns_none(self):
        self.assertIsNone(
            grpo_utils.compute_tis_mask(
                torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool), 0.0, 0.0
            )
        )


def _data_loader_available() -> bool:
    try:
        import vllm  # noqa: F401, PLC0415
    except Exception:
        return False
    return True


@unittest.skipUnless(_data_loader_available(), "data_loader requires vllm")
class TestSiblingAssembly(unittest.TestCase):
    def test_extract_subseq_indices_per_pack(self):
        from open_instruct.data_loader import _extract_subseq_indices_per_pack  # noqa: PLC0415

        # pack 1: [1,1,1,0,2,2,2], pack 2: [3,3,0,0]
        rm = [torch.tensor([[1, 1, 1, 0, 2, 2, 2]]), torch.tensor([[3, 3, 0, 0]])]
        got = _extract_subseq_indices_per_pack(rm)
        self.assertEqual(got, [[1, 2], [3]])

    def test_populate_value_model_fields_minimal(self):
        from open_instruct.data_loader import populate_value_model_fields  # noqa: PLC0415

        # Fake packed sequences with one sub-seq per pack.
        ps = PackedSequences(
            query_responses=[torch.tensor([[0, 1, 2, 3]])],
            attention_masks=[torch.tensor([[1, 1, 1, 1]])],
            response_masks=[torch.tensor([[0, 1, 1, 1]])],
            original_responses=[[1, 2, 3]],
            advantages=None,
            num_actions=[torch.tensor([[3]])],
            position_ids=[torch.tensor([[0, 1, 2, 3]])],
            packed_seq_lens=[torch.tensor([[4]])],
            vllm_logprobs=[torch.tensor([[0.0, -0.1, -3.0, -0.1]])],
            dones=[torch.tensor([[0, 0, 0, 1]])],
        )
        populate_value_model_fields(
            packed_sequences=ps,
            scores=np.array([0.5]),
            batch_ground_truths=["42"],
            decoded_responses=["hello"],
            num_samples_per_prompt=1,
            max_possible_score=1.0,
            use_sae=True,
            sae_threshold=0.2,
            need_ground_truths=True,
            need_siblings=False,
            num_siblings_to_sample=0,
            rng=np.random.default_rng(0),
        )
        self.assertIsNotNone(ps.rewards)
        self.assertEqual(ps.rewards[0].tolist(), [[0.0, 0.0, 0.0, 0.5]])
        self.assertIsNotNone(ps.segment_boundaries)
        # t=2 has logprob -3.0 < log(0.2) -> boundary.
        self.assertTrue(bool(ps.segment_boundaries[0][0, 2].item()))
        self.assertEqual(ps.ground_truths, [["42"]])


class TestConditioningBuilders(unittest.TestCase):
    def test_every_template_builds(self):
        siblings = [{"text": "abc", "is_correct": True}, {"text": "def", "is_correct": False}]
        for t in ["answer_prefix", "expected_accuracy", "rollout_context", "correct_demo"]:
            txt = build_conditioning_text(t, ground_truth="42", siblings=siblings)
            self.assertIsInstance(txt, str)
            self.assertGreater(len(txt), 0)

    def test_unknown_template_raises(self):
        with self.assertRaises(ValueError):
            build_conditioning_text("bogus", "42", [])

    def test_is_postfix_template(self):
        self.assertFalse(is_postfix_template("answer_prefix"))
        self.assertTrue(is_postfix_template("expected_accuracy"))
        self.assertFalse(is_postfix_template("rubrics"))

    def test_correct_demo_auto_samples_all_other_rollouts(self):
        self.assertEqual(resolve_num_siblings_to_sample("correct_demo", -1, num_samples_per_prompt=8), 7)
        self.assertEqual(resolve_num_siblings_to_sample("rollout_context", -1, num_samples_per_prompt=8), 4)
        self.assertEqual(resolve_num_siblings_to_sample("correct_demo", 2, num_samples_per_prompt=8), 2)


class TestRubricConditioning(unittest.TestCase):
    def _gt(self, rubrics: list[dict]) -> str:
        return json.dumps({"query": "What is 2+2?", "rubrics": rubrics})

    def test_renders_positive_and_negative_rubrics(self):
        gt = self._gt(
            [
                {"title": "Correct answer", "description": "Answer mentions 4.", "weight": 1.0},
                {"title": "Cites step", "description": "Shows arithmetic.", "weight": 1.0},
                {"title": "Hallucination", "description": "Invents calculation.", "weight": -1.0},
            ]
        )
        txt = build_conditioning_text("rubrics", ground_truth=gt)
        self.assertIn("Positive criteria", txt)
        self.assertIn("Correct answer: Answer mentions 4.", txt)
        self.assertIn("Cites step: Shows arithmetic.", txt)
        self.assertIn("Negative criteria", txt)
        self.assertIn("Hallucination: Invents calculation.", txt)

    def test_handles_missing_or_empty_rubrics(self):
        # Empty rubrics list -> empty conditioning string.
        self.assertEqual(build_conditioning_text("rubrics", ground_truth=self._gt([])), "")
        # No rubrics field -> empty.
        self.assertEqual(build_conditioning_text("rubrics", ground_truth=json.dumps({"query": "q"})), "")

    def test_handles_invalid_json_gracefully(self):
        self.assertEqual(build_conditioning_text("rubrics", ground_truth="not json"), "")
        self.assertEqual(build_conditioning_text("rubrics", ground_truth=""), "")

    def test_rubric_order_is_stable_within_polarity(self):
        gt = self._gt(
            [
                {"title": "A", "description": "first", "weight": 1.0},
                {"title": "B", "description": "second", "weight": -1.0},
                {"title": "C", "description": "third", "weight": 1.0},
            ]
        )
        txt = build_conditioning_text("rubrics", ground_truth=gt)
        self.assertLess(txt.index("first"), txt.index("third"))
        # Negative section appears after the positive section.
        self.assertLess(txt.index("third"), txt.index("second"))


class TestScoreParsing(unittest.TestCase):
    def test_direct_parsing(self):
        self.assertEqual(parse_generative_value_score("<answer>7</answer>"), 7.0)
        self.assertEqual(parse_generative_value_score("some reasoning... <answer>10</answer>"), 10.0)
        self.assertEqual(parse_generative_value_score("<answer>5.5</answer>"), 5.5)
        self.assertIsNone(parse_generative_value_score("no digits here"))

    def test_cot_parsing(self):
        self.assertEqual(parse_generative_value_score("The approach is good... <answer>7.5</answer>"), 7.5)
        self.assertIsNone(parse_generative_value_score("no answer tags"))

    def test_clamping(self):
        self.assertEqual(parse_generative_value_score("<answer>42</answer>", score_min=0, score_max=10), 10.0)
        self.assertEqual(parse_generative_value_score("<answer>-5</answer>", score_min=0, score_max=10), 0.0)

    def test_prompt_has_conditioning(self):
        p = build_generative_value_prompt("partial", conditioning="gt", ground_truth="42")
        self.assertIn("The correct answer is 42", p)
        self.assertIn("<rollout>", p)
        self.assertIn("Answer:", p)

    def test_prompt_has_actor_and_remaining_budget_context(self):
        p = build_generative_value_prompt(
            "partial",
            conditioning="none",
            actor_model_name="Qwen/Qwen3-4B-Base",
            actor_success_rate=0.125,
            response_tokens_used=7000,
            response_token_limit=8192,
        )

        self.assertIn("The active actor is Qwen/Qwen3-4B-Base", p)
        self.assertIn("success rate on this task distribution is 12.5%", p)
        self.assertIn("used 7000 of its 8192 token budget; 1192 tokens remain", p)

    def test_prompt_rejects_invalid_remaining_budget_context(self):
        with self.assertRaisesRegex(ValueError, "response_tokens_used must be in"):
            build_generative_value_prompt(
                "partial", conditioning="none", response_tokens_used=8193, response_token_limit=8192
            )


class TestValueLoss(unittest.TestCase):
    def test_defaults(self):
        config = grpo_utils.GRPOExperimentConfig()
        self.assertEqual(config.value_loss, "mse")
        self.assertEqual(config.value_loss_coef, 1.0)
        self.assertTrue(config.bound_value_predictions)
        self.assertTrue(config.skip_tool_outputs)

    def test_bounded_value_default_is_inert_without_value_model(self):
        config = grpo_utils.GRPOExperimentConfig(use_value_model=False)

        self.assertTrue(config.bound_value_predictions)

    def test_dppo_is_a_configured_loss_option(self):
        config = grpo_utils.GRPOExperimentConfig(loss_fn=grpo_utils.GRPOLossType.dppo, dppo_clip=0.02)
        self.assertEqual(config.loss_fn, grpo_utils.GRPOLossType.dppo)

    def test_dppo_requires_an_explicit_clip(self):
        with self.assertRaisesRegex(ValueError, "requires an explicit --dppo_clip"):
            grpo_utils.GRPOExperimentConfig(loss_fn=grpo_utils.GRPOLossType.dppo)

    def test_dppo_rejects_out_of_range_clip(self):
        with self.assertRaisesRegex(ValueError, "dppo_clip must be in"):
            grpo_utils.GRPOExperimentConfig(loss_fn=grpo_utils.GRPOLossType.dppo, dppo_clip=0.0)

    def test_bounded_value_prediction_uses_reward_range(self):
        logits = torch.tensor([-10.0, 0.0, 10.0])
        values = bounded_value_prediction(logits, value_min=-2.0, value_max=4.0)

        self.assertTrue(bool((values > -2.0).all()))
        self.assertTrue(bool((values < 4.0).all()))
        torch.testing.assert_close(values[1], torch.tensor(1.0))
        self.assertLess(float(values[0]), 0.0)
        self.assertGreater(float(values[2]), 2.0)

    def test_predict_values_can_bound_mse_head(self):
        logits = torch.tensor([[[-100.0], [0.0], [100.0]]])
        values = predict_values(logits, "mse", bound_predictions=True, value_min=0.0, value_max=10.0)

        self.assertTrue(bool((values > 0.0).all()))
        self.assertTrue(bool((values < 10.0).all()))
        torch.testing.assert_close(values[0, 1], torch.tensor(5.0))

    def test_gen_value_unit_score_maps_to_reward_range(self):
        self.assertEqual(unit_value_to_reward(0.25, value_min=-2.0, value_max=6.0), 0.0)
        self.assertEqual(reward_to_unit_value(0.0, value_min=-2.0, value_max=6.0), 0.25)

    def test_classification_supports_ground_truth_conditioning(self):
        config = grpo_utils.GRPOExperimentConfig(
            use_value_model=True,
            value_loss="classification",
            value_model_ground_truth_conditioning=True,
            gt_conditioning_template="answer_prefix",
        )
        self.assertEqual(config.value_loss, "classification")
        self.assertTrue(config.value_model_ground_truth_conditioning)

    def test_causal_value_mask_uses_shifted_action_coordinates(self):
        response_mask = torch.tensor([[False, False, True, False, True]])
        expected = torch.tensor([[False, True, False, True]])
        torch.testing.assert_close(causal_value_mask(response_mask), expected)

    def test_accumulation_group_token_counts_handles_variable_pack_lengths(self):
        masks = [
            torch.tensor([[True, True, False]]),
            torch.tensor([[True, True, True, False]]),
            torch.tensor([[False, True]]),
        ]

        counts = accumulation_group_token_counts(masks, accumulation_steps=2)

        torch.testing.assert_close(counts, torch.tensor([5.0, 1.0]))

    def test_balanced_accumulation_groups_use_exact_requested_step_count(self):
        self.assertEqual(balanced_accumulation_group_ids(num_samples=5, num_groups=2), [0, 0, 0, 1, 1])

    def test_balanced_accumulation_groups_reject_more_steps_than_packs(self):
        with self.assertRaisesRegex(ValueError, "num_groups cannot exceed num_samples"):
            balanced_accumulation_group_ids(num_samples=2, num_groups=5)

    def test_grouped_token_counts_follow_balanced_accumulation_groups(self):
        masks = [torch.ones(1, token_count, dtype=torch.bool) for token_count in (1, 2, 3, 4, 5)]
        group_ids = balanced_accumulation_group_ids(num_samples=len(masks), num_groups=2)

        counts = grouped_token_counts(masks, group_ids)

        torch.testing.assert_close(counts, torch.tensor([6.0, 9.0]))

    def test_value_loss_contributions_form_one_token_weighted_mean(self):
        short_pack = torch.tensor([1.0, 3.0])
        long_pack = torch.tensor([5.0, 7.0, 9.0])
        global_token_count = short_pack.numel() + long_pack.numel()

        loss = normalize_value_loss(short_pack, global_token_count, loss_coef=0.5, data_parallel_world_size=1)
        loss += normalize_value_loss(long_pack, global_token_count, loss_coef=0.5, data_parallel_world_size=1)

        expected = torch.cat((short_pack, long_pack)).mean() * 0.5
        torch.testing.assert_close(loss, expected)

    def test_value_loss_normalization_compensates_for_dp_averaging(self):
        rank_0 = normalize_value_loss(
            torch.tensor([1.0, 3.0]), global_token_count=5, loss_coef=1.0, data_parallel_world_size=2
        )
        rank_1 = normalize_value_loss(
            torch.tensor([5.0, 7.0, 9.0]), global_token_count=5, loss_coef=1.0, data_parallel_world_size=2
        )

        deepspeed_averaged_loss = (rank_0 + rank_1) / 2
        torch.testing.assert_close(deepspeed_averaged_loss, torch.tensor(5.0))

    def test_value_metrics_are_token_weighted_across_packs(self):
        short_pack_stats = value_metric_sums(torch.tensor([1.0, 3.0]), torch.tensor(0.5), torch.tensor([True, True]))
        long_pack_stats = value_metric_sums(
            torch.tensor([5.0, 7.0, 9.0]), torch.tensor(1 / 3), torch.tensor([True, True, True])
        )

        value_loss, clipfrac = value_metrics_from_sums(short_pack_stats + long_pack_stats, loss_coef=0.5)

        torch.testing.assert_close(value_loss, torch.tensor(2.5, dtype=torch.float64))
        torch.testing.assert_close(clipfrac, torch.tensor(0.4, dtype=torch.float64))

    def test_regression_diagnostics_are_exact_with_unequal_rank_token_counts(self):
        rank_0 = regression_metric_sums(torch.tensor([0.0]), torch.tensor([0.2]))
        rank_1 = regression_metric_sums(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0.5, 0.75, 1.0]))

        metrics = regression_metrics_from_sums(rank_0 + rank_1)
        returns = np.array([0.0, 1.0, 1.0, 1.0])
        predictions = np.array([0.2, 0.5, 0.75, 1.0])
        residuals = returns - predictions

        self.assertAlmostEqual(metrics["value/returns_mean"], float(returns.mean()))
        self.assertAlmostEqual(metrics["value/returns_std"], float(returns.std()))
        self.assertAlmostEqual(metrics["value/predictions_mean"], float(predictions.mean()))
        self.assertAlmostEqual(metrics["value/predictions_std"], float(predictions.std()))
        self.assertAlmostEqual(
            metrics["value/explained_variance"], 1.0 - float(residuals.var()) / (float(returns.var()) + 1e-8)
        )

    def test_gen_value_parse_failure_has_zero_reinforce_reward(self):
        reward, squared_error = generative_value_reinforce_reward(outcome=0.0, prediction=None)

        self.assertEqual(reward, 0.0)
        self.assertIsNone(squared_error)

    def test_gen_value_parsed_prediction_uses_mse_shaped_reward(self):
        reward, squared_error = generative_value_reinforce_reward(outcome=1.0, prediction=0.25)

        self.assertEqual(squared_error, 0.75**2)
        self.assertEqual(reward, 1.0 - 0.75**2)

    def test_gen_value_validation_holdout_averages_initial_siblings_and_excludes_exact_pairs(self):
        def pair(prompt_ids, outcome, used):
            return {
                "request_output": SimpleNamespace(prompt_token_ids=prompt_ids),
                "outcome": outcome,
                "response_tokens_used": used,
                "response_token_limit": 100,
                "state_kind": "final_action" if used >= 90 else "segment_start",
            }

        a0, a1 = pair([1, 2], 0.0, 0), pair([1, 3], 0.0, 90)
        b0, b1 = pair([1, 2], 1.0, 0), pair([1, 4], 1.0, 95)
        middle = pair([9], 0.0, 50)
        examples, training_pairs = build_gen_value_validation_holdout(
            [{"pairs": [a0, middle, a1]}, {"pairs": [b0, b1]}], max_examples=3, seed=1
        )

        initial = [example for example in examples if example["kind"] == "initial"]
        self.assertEqual(len(initial), 1)
        self.assertEqual(initial[0]["target"], 0.5)
        self.assertEqual(len(examples), 3)
        self.assertEqual(training_pairs, [middle])

    def test_gen_value_validation_metrics_separate_terminal_failures(self):
        examples = [
            {"kind": "initial", "target": 0.25, "response_tokens_used": 0, "response_token_limit": 1000},
            {"kind": "final_action", "target": 0.0, "response_tokens_used": 950, "response_token_limit": 1000},
            {"kind": "final_action", "target": 1.0, "response_tokens_used": 900, "response_token_limit": 1000},
        ]

        metrics = gen_value_validation_metrics(examples, [0.5, 0.4, None])

        self.assertAlmostEqual(metrics["gen_value/validation_parse_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["gen_value/validation_mse"], (0.25**2 + 0.4**2) / 2)
        self.assertAlmostEqual(metrics["gen_value/validation_penalized_mse"], (0.25**2 + 0.4**2 + 1.0) / 3)
        self.assertEqual(metrics["gen_value/validation_final_incorrect_v_hat_mean"], 0.4)
        self.assertEqual(metrics["gen_value/validation_final_action_incorrect_v_hat_mean"], 0.4)
        self.assertEqual(metrics["gen_value/validation_near_horizon_incorrect_v_hat_mean"], 0.4)

    def test_gen_value_validation_snapshot_preserves_inspectable_outputs(self):
        examples = [
            {
                "prompt_token_ids": [1, 2, 3],
                "kind": "final_action",
                "target": 0.0,
                "response_tokens_used": 99,
                "response_token_limit": 100,
            }
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            snapshot_path = write_gen_value_validation_snapshot(
                output_dir,
                version=25,
                examples=examples,
                predictions=[0.2],
                prompts=["critic prompt"],
                generations=["reasoning <answer>2</answer>"],
            )

            self.assertEqual(snapshot_path, Path(output_dir) / "gen_value_validation/version_000025.jsonl")
            row = json.loads(snapshot_path.read_text())
            self.assertNotIn("prompt_token_ids", row)
            self.assertEqual(row["version"], 25)
            self.assertEqual(row["prediction"], 0.2)
            self.assertEqual(row["prompt"], "critic prompt")
            self.assertEqual(row["generation"], "reasoning <answer>2</answer>")

    def test_gen_value_training_trace_reservoir_is_atomic_and_manifested(self):
        examples = [
            {
                "outcome": 0.0,
                "prediction": 0.2,
                "prompt": "critic prompt",
                "generation": "reasoning <answer>2</answer>",
            }
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            trace_path = write_gen_value_training_trace_reservoir(
                output_dir, version=25, examples=examples, seen_by_outcome={"correct": 11, "incorrect": 23}
            )

            self.assertEqual(trace_path, Path(output_dir) / "gen_value_training_traces/reservoir.jsonl")
            self.assertEqual(json.loads(trace_path.read_text()), examples[0])
            manifest = json.loads(trace_path.with_name("manifest.json").read_text())
            self.assertEqual(manifest["critic_version"], 25)
            self.assertEqual(manifest["retained_examples"], 1)
            self.assertEqual(manifest["seen_by_outcome"], {"correct": 11, "incorrect": 23})

    def test_gen_value_sft_trace_selection_is_accurate_balanced_and_deduplicated(self):
        examples = [
            {
                "source_critic_version": 20,
                "outcome": 1.0,
                "prediction": 0.8,
                "squared_error": 0.04,
                "prompt": "correct-a",
                "generation": "reasoning <answer>0.8</answer>",
            },
            {
                "source_critic_version": 25,
                "outcome": 1.0,
                "prediction": 0.9,
                "squared_error": 0.01,
                "prompt": "correct-a",
                "generation": "better reasoning <answer>0.9</answer>",
            },
            {
                "source_critic_version": 25,
                "outcome": 1.0,
                "prediction": 0.95,
                "squared_error": 0.0025,
                "prompt": "correct-b",
                "generation": "reasoning <answer>0.95</answer>",
            },
            {
                "source_critic_version": 25,
                "outcome": 0.0,
                "prediction": 0.1,
                "squared_error": 0.01,
                "prompt": "incorrect-a",
                "generation": "reasoning <answer>0.1</answer>",
            },
            {
                "source_critic_version": 25,
                "outcome": 0.0,
                "prediction": None,
                "squared_error": None,
                "prompt": "parse-failure",
                "generation": "unparseable",
            },
            {
                "source_critic_version": 25,
                "outcome": 0.0,
                "prediction": 0.8,
                "squared_error": 0.64,
                "prompt": "inaccurate",
                "generation": "reasoning <answer>0.8</answer>",
            },
        ]

        selected = select_gen_value_sft_traces(examples, max_squared_error=0.04, min_critic_version=25, seed=7)

        self.assertEqual(len(selected), 2)
        self.assertEqual({example["outcome"] for example in selected}, {0.0, 1.0})
        selected_by_prompt = {example["prompt"]: example for example in selected}
        self.assertNotIn("parse-failure", selected_by_prompt)
        self.assertNotIn("inaccurate", selected_by_prompt)
        if "correct-a" in selected_by_prompt:
            self.assertEqual(selected_by_prompt["correct-a"]["prediction"], 0.9)

    def test_final_action_prefix_retains_observations_and_is_causal(self):
        prefix, response_tokens_used = causal_final_action_prefix_token_ids(
            [10, 11, 20, 21, 30, 31, 22], [False, False, True, True, False, False, True]
        )

        self.assertEqual(prefix, [20, 21, 30, 31])
        self.assertEqual(response_tokens_used, 2)

    def test_classification_loss_preserves_continuous_targets(self):
        probabilities = torch.tensor([[[0.75, 0.25], [0.25, 0.75]]])
        logits = probabilities.log()
        returns = torch.tensor([[0.25, 0.75]])
        mask = torch.tensor([[True, True]])

        per_token, clipfrac = compute_value_loss(
            logits, returns, old_values=None, mask=mask, loss_type="classification", clip_range=0.2
        )

        expected = -(probabilities * probabilities.log()).sum(dim=-1)
        torch.testing.assert_close(per_token, expected)
        torch.testing.assert_close(clipfrac, torch.tensor(0.0))
        torch.testing.assert_close(predict_values(logits, "classification"), returns)

    def test_classification_loss_rejects_out_of_range_targets(self):
        with self.assertRaisesRegex(ValueError, "outside \\[0.0, 1.0\\]"):
            compute_value_loss(
                torch.zeros(1, 1, 2),
                torch.tensor([[1.1]]),
                old_values=None,
                mask=torch.tensor([[True]]),
                loss_type="classification",
                clip_range=0.2,
            )

    def test_classification_supports_custom_reward_range(self):
        probabilities = torch.tensor([[[0.75, 0.25], [0.25, 0.75]]])
        logits = probabilities.log()
        returns = torch.tensor([[0.0, 4.0]])
        mask = torch.tensor([[True, True]])

        per_token, _ = compute_value_loss(
            logits,
            returns,
            old_values=None,
            mask=mask,
            loss_type="classification",
            clip_range=0.2,
            value_min=-2.0,
            value_max=6.0,
        )

        expected = -(probabilities * probabilities.log()).sum(dim=-1)
        torch.testing.assert_close(per_token, expected)
        torch.testing.assert_close(predict_values(logits, "classification", value_min=-2.0, value_max=6.0), returns)

    def test_mse_loss_no_clip(self):
        new_v = torch.tensor([[1.0, 2.0, 3.0]])
        ret = torch.tensor([[1.0, 1.0, 1.0]])
        mask = torch.tensor([[True, True, True]])
        per_tok, clipfrac = value_clipped_mse_loss(new_v, ret, None, mask, clip_range=0.0)
        self.assertEqual(per_tok.shape, new_v.shape)
        self.assertEqual(float(clipfrac), 0.0)

    def test_mse_loss_with_clip(self):
        new_v = torch.tensor([[10.0, 2.0]])
        old_v = torch.tensor([[0.0, 0.0]])
        ret = torch.tensor([[0.0, 0.0]])
        mask = torch.tensor([[True, True]])
        per_tok, clipfrac = value_clipped_mse_loss(new_v, ret, old_v, mask, clip_range=0.1)
        # PPO2 clipping is pessimistic: it uses the MAX of clipped and unclipped losses, so the
        # final per-token loss is dominated by (new - ret)^2 here (= 100). The configurable
        # coefficient is applied by the trainer, not this helper.
        self.assertAlmostEqual(float(per_tok[0, 0]), 100.0, places=5)
        self.assertEqual(per_tok.shape, new_v.shape)
        self.assertGreaterEqual(float(clipfrac), 0.0)


class TestGenValueSegmentation(unittest.TestCase):
    def test_fixed_segmentation(self):
        boundaries = segment_rollout(list(range(1500)), None, mode="fixed", fixed_chunk_size=500)
        # Inclusive ends for three exactly 500-token chunks.
        self.assertEqual(boundaries, [499, 999, 1499])

    def test_sae_segmentation(self):
        logps = [-0.1] * 10 + [-3.0] + [-0.1] * 10  # one boundary at t=10
        boundaries = segment_rollout([0] * 21, logps, mode="sae", sae_threshold=0.2)
        self.assertIn(10, boundaries)
        self.assertEqual(boundaries[-1], 20)

    def test_max_segments_cap(self):
        # 100 SAE boundaries (every token is low-prob), cap to 4.
        logps = [-3.0] * 100
        boundaries = segment_rollout([0] * 100, logps, mode="sae", sae_threshold=0.2, max_segments=4)
        self.assertEqual(len(boundaries), 4)
        self.assertEqual(boundaries[-1], 99)

    def test_fixed_with_max_segments(self):
        # Fixed chunks every 100 tokens over 1000 tokens = 10 boundaries, cap to 5.
        boundaries = segment_rollout(list(range(1000)), None, mode="fixed", fixed_chunk_size=100, max_segments=5)
        self.assertEqual(len(boundaries), 5)
        self.assertEqual(boundaries[-1], 999)


if __name__ == "__main__":
    unittest.main()
