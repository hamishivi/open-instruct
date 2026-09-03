import math
from unittest import mock

import pytest
import torch

from open_instruct import grpo_utils


def test_whiten_advantages_normalizes_valid_policy_tokens():
    advantages = [torch.tensor([[9.0, -1.0, 0.0, 1.0]]), torch.tensor([[8.0, 3.0, 5.0, 6.0]])]
    response_masks = [torch.tensor([[0, 1, 1, 1]]), torch.tensor([[0, 1, 0, 0]])]

    with mock.patch.object(grpo_utils.dist, "all_reduce") as all_reduce:
        whitened = grpo_utils.whiten_advantages(advantages, response_masks)

    all_reduce.assert_called_once()
    valid = torch.cat(
        [advantage[:, 1:][response_mask[:, 1:].bool()] for advantage, response_mask in zip(whitened, response_masks)]
    )
    torch.testing.assert_close(valid.mean(), torch.tensor(0.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(valid.std(unbiased=False), torch.tensor(1.0), atol=1e-6, rtol=0.0)
    assert whitened[0][0, 0] == 9.0
    assert whitened[1][0, 0] == 8.0
    assert torch.equal(whitened[1][0, 2:], torch.zeros(2))


def test_whiten_advantages_requires_matching_masks():
    with pytest.raises(ValueError, match="one response mask per advantage tensor"):
        grpo_utils.whiten_advantages([torch.zeros(1, 2)], [])


def test_dppo_policy_ratio_clamps_log_ratio_for_numerical_stability():
    new = torch.tensor([[0.0, -100.0]])
    old = torch.tensor([[-100.0, 0.0]])

    ratio = grpo_utils.compute_policy_ratio(new, old, grpo_utils.GRPOLossType.dppo)

    torch.testing.assert_close(ratio, torch.tensor([[math.exp(20.0), math.exp(-20.0)]]))


def test_dppo_old_logprobs_are_always_anchored_to_rollout_policy():
    rollout_logprobs = torch.tensor([[-2.0, -3.0]])
    recomputed_logprobs = torch.tensor([[-1.0, -1.5]])

    result = grpo_utils.resolve_old_logprob(
        [recomputed_logprobs],
        sample_idx=0,
        epoch_idx=1,
        num_mini_batches=2,
        use_vllm_logprobs=False,
        vllm_logprobs=rollout_logprobs,
        new_logprobs=torch.zeros_like(rollout_logprobs),
        loss_fn=grpo_utils.GRPOLossType.dppo,
    )

    torch.testing.assert_close(result, rollout_logprobs)


def test_dppo_reports_bounded_surrogate_while_clipped_gradient_is_zero():
    old_logprobs = torch.tensor([[math.log(0.5)]])
    new_logprobs = torch.tensor([[math.log(0.8)]], requires_grad=True)
    ratio = grpo_utils.compute_policy_ratio(new_logprobs, old_logprobs, grpo_utils.GRPOLossType.dppo)
    config = grpo_utils.GRPOExperimentConfig(loss_fn=grpo_utils.GRPOLossType.dppo, dppo_clip=0.1)

    pg_losses, bounded_losses, policy_loss, _ = grpo_utils.compute_grpo_loss(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        ratio=ratio,
        advantages=torch.ones_like(new_logprobs),
        ref_logprobs=None,
        config=config,
    )

    assert policy_loss.item() == pytest.approx(bounded_losses.item())
    assert policy_loss.item() > pg_losses.item()
    policy_loss.sum().backward()
    torch.testing.assert_close(new_logprobs.grad, torch.zeros_like(new_logprobs))


def test_mask_logprobs_rejects_nonfinite_response_values():
    with pytest.raises(ValueError, match="non-finite logprob"):
        grpo_utils.mask_logprobs(torch.tensor([[float("nan"), -1.0]]), torch.tensor([[True, True]]))


def test_mask_logprobs_allows_nonfinite_masked_values():
    result = grpo_utils.mask_logprobs(torch.tensor([[float("nan"), -1.0]]), torch.tensor([[False, True]]))

    torch.testing.assert_close(result, torch.tensor([[grpo_utils.INVALID_LOGPROB, -1.0]]))
