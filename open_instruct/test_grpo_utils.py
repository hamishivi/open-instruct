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
