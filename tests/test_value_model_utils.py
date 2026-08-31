"""CPU-only tests for generative-value target and replay helpers."""

from types import SimpleNamespace

import pytest

from open_instruct import value_model_utils


def _pair(
    prompt_token_ids: list[int],
    outcome: float,
    *,
    state_kind: str,
    response_tokens_used: int,
    response_token_limit: int = 8192,
) -> dict:
    return {
        "request_output": SimpleNamespace(prompt_token_ids=prompt_token_ids),
        "outcome": outcome,
        "state_kind": state_kind,
        "response_tokens_used": response_tokens_used,
        "response_token_limit": response_token_limit,
    }


def test_gen_value_validation_targets_use_empirical_initial_and_sampled_prefix_returns():
    correct = {
        "pairs": [
            _pair([1, 2], 1.0, state_kind="segment_start", response_tokens_used=0),
            _pair([1, 2, 3], 1.0, state_kind="segment_start", response_tokens_used=3000),
            _pair([1, 2, 4], 1.0, state_kind="final_action", response_tokens_used=4000),
        ]
    }
    incorrect = {
        "pairs": [
            _pair([1, 2], 0.0, state_kind="segment_start", response_tokens_used=0),
            _pair([1, 2, 5], 0.0, state_kind="segment_start", response_tokens_used=3000),
            _pair([1, 2, 6], 0.0, state_kind="final_action", response_tokens_used=4000),
        ]
    }

    examples, training_pairs = value_model_utils.build_gen_value_validation_holdout(
        [correct, incorrect], max_examples=8, prompt_holdout_fraction=1.0
    )

    initial = [example for example in examples if example["kind"] == "initial"]
    assert len(initial) == 1
    assert initial[0]["target"] == pytest.approx(0.5)
    assert initial[0]["target_source"] == "sibling_empirical_return"

    sampled = [example for example in examples if example.get("target_source") == "single_sample_return"]
    final_actions = [example for example in sampled if example["kind"] == "final_action"]
    assert {example["target"] for example in final_actions} == {0.0, 1.0}
    assert all(example["target"] in {0.0, 1.0} for example in sampled)
    assert training_pairs == []


def test_gen_value_validation_removes_every_state_from_heldout_prompt_group():
    rollouts = []
    for prompt_ids, outcome in (([1], 0.0), ([2], 1.0)):
        rollouts.append(
            {
                "pairs": [
                    _pair(prompt_ids, outcome, state_kind="segment_start", response_tokens_used=0),
                    _pair(prompt_ids + [9], outcome, state_kind="final_action", response_tokens_used=7000),
                ]
            }
        )

    examples, training_pairs = value_model_utils.build_gen_value_validation_holdout(
        rollouts, max_examples=1, seed=7, prompt_holdout_fraction=0.5
    )

    heldout_prompt = tuple(examples[0]["prompt_token_ids"])
    assert training_pairs
    assert all(tuple(pair["request_output"].prompt_token_ids) != heldout_prompt for pair in training_pairs)


def test_final_action_replay_does_not_distort_leave_one_out_baseline():
    first = {"state_kind": "final_action"}
    second = {"state_kind": "segment_start"}
    replayed = value_model_utils.replay_gen_value_final_actions([first, second], replay_weight=4)
    # Ray may deserialize repeated dictionary aliases as distinct objects. The
    # explicit sample ID must still collapse them to their original completion.
    serialized_replayed = [dict(pair) for pair in replayed]
    sample_ids = [value_model_utils.gen_value_pair_sample_id(pair) for pair in serialized_replayed]

    weights = value_model_utils.generative_value_reinforce_weights_with_replay(
        rewards=[1.0, 1.0, 1.0, 1.0, 0.75],
        baseline="leave_one_out_by_outcome",
        sample_ids=sample_ids,
        outcomes=[1.0] * 5,
    )

    assert sample_ids == [0, 0, 0, 0, 1]
    assert len(value_model_utils.unique_replayed_gen_value_pairs(serialized_replayed)) == 2
    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25, -0.25])


def test_shared_state_returns_pool_unique_continuations_without_dropping_replays():
    correct = _pair([1, 2], 1.0, state_kind="final_action", response_tokens_used=0)
    incorrect = _pair([1, 2], 0.0, state_kind="segment_start", response_tokens_used=0)
    distinct = _pair([1, 3], 0.0, state_kind="segment_start", response_tokens_used=0)
    replayed = value_model_utils.replay_gen_value_final_actions([correct, incorrect, distinct], replay_weight=4)
    serialized_replayed = [dict(pair) for pair in replayed]

    targets, metrics = value_model_utils.pool_gen_value_shared_state_returns(serialized_replayed)

    assert targets == {0: 0.5, 1: 0.5, 2: 0.0}
    assert metrics == {
        "gen_value/shared_state_unique_examples": 3.0,
        "gen_value/shared_state_groups": 2.0,
        "gen_value/shared_state_pooled_groups": 1.0,
        "gen_value/shared_state_pooled_examples": 2.0,
        "gen_value/shared_state_changed_examples": 2.0,
    }


@pytest.mark.parametrize("outcome", [float("nan"), -0.1, 1.1])
def test_shared_state_return_pooling_rejects_invalid_outcomes(outcome: float):
    pair = _pair([1, 2], outcome, state_kind="segment_start", response_tokens_used=0)

    with pytest.raises(ValueError, match="finite and in \\[0, 1\\]"):
        value_model_utils.pool_gen_value_shared_state_returns([pair])


def test_near_horizon_detection_requires_incorrect_outcome_and_little_budget():
    near_horizon = {"outcome": 0.0, "response_tokens_used": 7500, "response_token_limit": 8192}
    assert value_model_utils.is_gen_value_near_horizon_incorrect(near_horizon)
    assert not value_model_utils.is_gen_value_near_horizon_incorrect({**near_horizon, "outcome": 1.0})
    assert not value_model_utils.is_gen_value_near_horizon_incorrect({**near_horizon, "response_tokens_used": 6000})


def test_gen_value_validation_reports_early_to_late_value_deltas():
    examples = [
        {"kind": "segment_start", "target": 1.0, "trajectory_fraction": 0.25},
        {"kind": "segment_start", "target": 1.0, "trajectory_fraction": 0.75},
        {"kind": "segment_start", "target": 0.0, "trajectory_fraction": 0.25},
        {"kind": "segment_start", "target": 0.0, "trajectory_fraction": 0.75},
    ]

    metrics = value_model_utils.gen_value_validation_metrics(examples, [0.2, 0.8, 0.5, 0.1])

    assert metrics["gen_value/validation_prefix_correct_early_to_late_delta"] == pytest.approx(0.6)
    assert metrics["gen_value/validation_prefix_incorrect_early_to_late_delta"] == pytest.approx(-0.4)
    assert metrics["gen_value/validation_prefix_value_gap_early_to_late_delta"] == pytest.approx(1.0)
    assert metrics["gen_value/validation_prefix_auc"] == pytest.approx(0.75)
    assert metrics["gen_value/validation_prefix_early_auc"] == pytest.approx(0.0)
    assert metrics["gen_value/validation_prefix_late_auc"] == pytest.approx(1.0)


def test_gen_value_validation_auc_assigns_half_credit_to_ties():
    examples = [
        {"kind": "final_action", "target": 1.0},
        {"kind": "final_action", "target": 1.0},
        {"kind": "final_action", "target": 0.0},
        {"kind": "final_action", "target": 0.0},
    ]

    metrics = value_model_utils.gen_value_validation_metrics(examples, [0.8, 0.4, 0.4, 0.2])

    assert metrics["gen_value/validation_final_auc"] == pytest.approx(0.875)
    assert metrics["gen_value/validation_final_action_auc"] == pytest.approx(0.875)


@pytest.mark.parametrize(
    ("threshold", "observed", "expected"),
    [
        (None, 0.0, False),
        (0.2, None, False),
        (0.2, 0.3, False),
        (0.2, 0.2, False),
        (0.2, 0.19, True),
        (0.2, float("nan"), True),
    ],
)
def test_gen_value_policy_guard(threshold: float | None, observed: float | None, expected: bool):
    assert value_model_utils.gen_value_policy_guard_active(threshold, observed) is expected


def test_gen_value_policy_guard_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="min_advantage_gap"):
        value_model_utils.gen_value_policy_guard_active(-0.1, 0.3)


@pytest.mark.parametrize(("world_size", "max_async_steps", "expected"), [(1, 1, 1), (4, 1, 4), (2, 3, 6)])
def test_gen_value_training_queue_capacity(world_size: int, max_async_steps: int, expected: int):
    assert value_model_utils.gen_value_training_queue_capacity(world_size, max_async_steps) == expected


@pytest.mark.parametrize(("world_size", "max_async_steps"), [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_gen_value_training_queue_capacity_rejects_invalid_values(world_size: int, max_async_steps: int):
    with pytest.raises(ValueError):
        value_model_utils.gen_value_training_queue_capacity(world_size, max_async_steps)


@pytest.mark.parametrize(
    ("policy_step", "latest_trained_step", "max_async_steps", "expected"),
    [
        (1, 0, 1, True),
        (2, 1, 1, True),
        (2, 0, 1, False),
        (5, 3, 2, True),
        (6, 3, 2, False),
    ],
)
def test_gen_value_source_step_admission_window(
    policy_step: int, latest_trained_step: int, max_async_steps: int, expected: bool
):
    assert (
        value_model_utils.gen_value_source_step_is_admissible(
            policy_step, latest_trained_step, max_async_steps
        )
        is expected
    )


@pytest.mark.parametrize(("policy_step", "latest_trained_step", "max_async_steps"), [(-1, 0, 1), (1, -1, 1), (1, 0, 0)])
def test_gen_value_source_step_admission_rejects_invalid_values(
    policy_step: int, latest_trained_step: int, max_async_steps: int
):
    with pytest.raises(ValueError):
        value_model_utils.gen_value_source_step_is_admissible(
            policy_step, latest_trained_step, max_async_steps
        )


def test_gen_value_training_progress_waits_for_all_admitted_rollouts():
    progress = value_model_utils.GenValueTrainingProgressState(latest_trained_policy_step=3, policy_world_size=2)

    progress.register_admitted_policy_step(policy_training_step=4, policy_rank=0, num_rollouts=2)
    progress.register_admitted_policy_step(policy_training_step=4, policy_rank=1, num_rollouts=1)
    progress.record_trained_policy_steps([4, 4])
    assert progress.get_latest_trained_policy_step() == 3

    progress.record_trained_policy_steps([4])
    assert progress.get_latest_trained_policy_step() == 4


def test_gen_value_training_progress_handles_training_before_registration():
    progress = value_model_utils.GenValueTrainingProgressState(latest_trained_policy_step=4, policy_world_size=2)

    progress.record_trained_policy_steps([5, 5])
    progress.register_admitted_policy_step(policy_training_step=5, policy_rank=0, num_rollouts=1)
    assert progress.get_latest_trained_policy_step() == 4

    progress.register_admitted_policy_step(policy_training_step=5, policy_rank=1, num_rollouts=1)
    assert progress.get_latest_trained_policy_step() == 5


def test_gen_value_training_progress_rejects_duplicate_rank_registration():
    progress = value_model_utils.GenValueTrainingProgressState(latest_trained_policy_step=0, policy_world_size=1)
    progress.register_admitted_policy_step(policy_training_step=1, policy_rank=0, num_rollouts=1)

    with pytest.raises(ValueError, match="more than once"):
        progress.register_admitted_policy_step(policy_training_step=1, policy_rank=0, num_rollouts=1)
