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
import unittest

import numpy as np
import torch

from open_instruct import grpo_utils
from open_instruct.rl_utils import (
    PackedSequences,
    calculate_advantages_packed,
    calculate_advantages_packed_sae,
    calculate_advantages_packed_sae_vapo,
    calculate_advantages_packed_vapo,
    calculate_length_adaptive_lambda,
)
from open_instruct.value_model_utils import (
    build_conditioning_text,
    build_generative_value_prompt,
    causal_value_mask,
    compute_value_loss,
    is_postfix_template,
    parse_generative_value_score,
    predict_values,
    resolve_num_siblings_to_sample,
    segment_rollout,
    value_clipped_mse_loss,
)


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

    def test_segment_adaptive_sae_preserves_whole_response_retention(self):
        v, r, d, m, logp = self._inputs()
        alpha = 1.5
        _, _, metrics = calculate_advantages_packed_sae_vapo(
            v,
            r,
            gamma=1.0,
            dones=d,
            response_masks=m,
            logprobs=logp,
            sae_threshold=0.2,
            segment_adaptive=True,
            segment_adaptive_alpha=alpha,
        )
        num_boundaries = 1
        expected_lambda = np.exp(-1.0 / (alpha * num_boundaries))
        self.assertAlmostEqual(metrics["value/sae_boundary_lambda_mean"], expected_lambda)
        self.assertAlmostEqual(expected_lambda**num_boundaries, np.exp(-1.0 / alpha))

    def test_length_adaptive_lambda(self):
        # alpha*length = 1 -> lambda = 0
        self.assertEqual(calculate_length_adaptive_lambda(1, alpha=1.0), 0.0)
        # alpha*length = 100 -> lambda close to 1
        self.assertGreater(calculate_length_adaptive_lambda(100, alpha=1.0), 0.98)

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


class TestTISMask(unittest.TestCase):
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


class TestValueLoss(unittest.TestCase):
    def test_defaults(self):
        config = grpo_utils.GRPOExperimentConfig()
        self.assertEqual(config.value_loss, "mse")
        self.assertEqual(config.value_loss_coef, 1.0)
        self.assertTrue(config.skip_tool_outputs)

    def test_causal_value_mask_uses_shifted_action_coordinates(self):
        response_mask = torch.tensor([[False, False, True, False, True]])
        expected = torch.tensor([[False, True, False, True]])
        torch.testing.assert_close(causal_value_mask(response_mask), expected)

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
        with self.assertRaisesRegex(ValueError, "outside \\[0, 1\\]"):
            compute_value_loss(
                torch.zeros(1, 1, 2),
                torch.tensor([[1.1]]),
                old_values=None,
                mask=torch.tensor([[True]]),
                loss_type="classification",
                clip_range=0.2,
            )

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
        # Boundaries at 500 and 1000, plus a final boundary at L-1.
        self.assertEqual(boundaries, [500, 1000, 1499])

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
