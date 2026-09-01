"""CPU-only tests for generative-value target and replay helpers."""

import json
import random
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


def test_mark_gen_value_training_pairs_for_optimizer_is_uniform_and_reproducible():
    pairs = [
        {"identifier": index, "request_output": SimpleNamespace(prompt_token_ids=[index])} for index in range(20)
    ]

    first, first_probability, first_count = value_model_utils.mark_gen_value_training_pairs_for_optimizer(
        pairs, 6, random.Random(17)
    )
    second, second_probability, second_count = value_model_utils.mark_gen_value_training_pairs_for_optimizer(
        pairs, 6, random.Random(17)
    )

    assert first == second
    assert first_probability == second_probability == pytest.approx(0.3 + 0.7**20 / 20)
    assert first_count == second_count
    assert len(first) == len(pairs)
    selected = [pair for pair in first if value_model_utils.gen_value_optimizer_selected(pair)]
    assert 1 <= len(selected) < len(pairs)
    assert len({pair["identifier"] for pair in selected}) == len(selected)


def test_mark_gen_value_training_pairs_for_optimizer_preserves_small_batches():
    pairs = [
        {"identifier": index, "request_output": SimpleNamespace(prompt_token_ids=[index])} for index in range(3)
    ]

    for target in (3, 4):
        marked, inclusion_probability, selected_count = (
            value_model_utils.mark_gen_value_training_pairs_for_optimizer(pairs, target, random.Random(0))
        )
        assert [pair["identifier"] for pair in marked] == [0, 1, 2]
        assert all(value_model_utils.gen_value_optimizer_selected(pair) for pair in marked)
        assert inclusion_probability == 1.0
        assert selected_count == 3


def test_mark_gen_value_training_pairs_for_optimizer_rejects_nonpositive_target():
    with pytest.raises(ValueError, match="must be positive"):
        value_model_utils.mark_gen_value_training_pairs_for_optimizer([], 0, random.Random(0))


def test_mark_gen_value_training_pairs_for_optimizer_keeps_shared_states_together():
    pairs = [
        {"identifier": index, "request_output": SimpleNamespace(prompt_token_ids=[index // 3])}
        for index in range(30)
    ]

    marked, _, selected_count = value_model_utils.mark_gen_value_training_pairs_for_optimizer(
        pairs, 8, random.Random(5)
    )
    sampled = [pair for pair in marked if value_model_utils.gen_value_optimizer_selected(pair)]

    selected_prompts = {tuple(pair["request_output"].prompt_token_ids) for pair in sampled}
    assert sampled
    assert selected_count == len(sampled)
    for prompt_ids in selected_prompts:
        expected_group = [pair for pair in pairs if tuple(pair["request_output"].prompt_token_ids) == prompt_ids]
        actual_group = [pair for pair in sampled if tuple(pair["request_output"].prompt_token_ids) == prompt_ids]
        assert [pair["identifier"] for pair in actual_group] == [pair["identifier"] for pair in expected_group]


def test_mark_gen_value_training_pairs_for_optimizer_fallback_probability_is_exact():
    class EmptyDrawRng(random.Random):
        def random(self):
            return 1.0

        def choice(self, values):
            return values[0]

    pairs = [
        {"identifier": index, "request_output": SimpleNamespace(prompt_token_ids=[index])} for index in range(4)
    ]
    marked, inclusion_probability, selected_count = value_model_utils.mark_gen_value_training_pairs_for_optimizer(
        pairs, 1, EmptyDrawRng()
    )

    assert inclusion_probability == pytest.approx(0.25 + 0.75**4 / 4)
    assert selected_count == 1
    assert value_model_utils.gen_value_optimizer_selected(marked[0])
    assert not any(value_model_utils.gen_value_optimizer_selected(pair) for pair in marked[1:])


def test_mark_gen_value_training_pairs_for_optimizer_is_unbiased_for_unequal_groups():
    class ScriptedRng:
        def __init__(self, retained: tuple[bool, ...], fallback_index: int):
            self.retained = iter(retained)
            self.fallback_index = fallback_index

        def random(self):
            return 0.0 if next(self.retained) else 1.0

        def choice(self, values):
            return values[self.fallback_index]

    group_sizes = (1, 2, 4)
    pairs = [
        {
            "contribution": float(10 * group_index + pair_index + 1),
            "request_output": SimpleNamespace(prompt_token_ids=[group_index]),
        }
        for group_index, group_size in enumerate(group_sizes)
        for pair_index in range(group_size)
    ]
    target_examples = 2
    base_probability = target_examples / len(pairs)
    expected_estimator = 0.0

    for mask in range(1 << len(group_sizes)):
        retained = tuple(bool(mask & (1 << group_index)) for group_index in range(len(group_sizes)))
        draw_probability = base_probability ** sum(retained) * (1.0 - base_probability) ** (
            len(group_sizes) - sum(retained)
        )
        fallback_indices = range(len(group_sizes)) if not any(retained) else range(1)
        for fallback_index in fallback_indices:
            marked, inclusion_probability, _ = value_model_utils.mark_gen_value_training_pairs_for_optimizer(
                pairs, target_examples, ScriptedRng(retained, fallback_index)
            )
            estimator = sum(
                pair["contribution"] / inclusion_probability
                for pair in marked
                if value_model_utils.gen_value_optimizer_selected(pair)
            )
            fallback_probability = 1.0 / len(group_sizes) if not any(retained) else 1.0
            expected_estimator += draw_probability * fallback_probability * estimator

    assert expected_estimator == pytest.approx(sum(pair["contribution"] for pair in pairs))


def test_expected_gen_value_score_from_logprobs_is_continuous_and_normalized():
    expected_score, probabilities = value_model_utils.expected_gen_value_score_from_logprobs(
        list(range(11)), [0.0] * 11
    )

    assert expected_score == pytest.approx(5.0)
    assert probabilities == pytest.approx([1.0 / 11.0] * 11)
    assert sum(probabilities) == pytest.approx(1.0)


def test_expected_gen_value_score_from_logprobs_is_numerically_stable():
    expected_score, probabilities = value_model_utils.expected_gen_value_score_from_logprobs(
        [0.0, 10.0], [-10_000.0, -10_001.0]
    )

    assert probabilities == pytest.approx([0.7310585786300049, 0.2689414213699951])
    assert expected_score == pytest.approx(2.6894142136999513)


@pytest.mark.parametrize(
    ("scores", "logprobs", "message"),
    [
        ([], [], "At least one"),
        ([0.0], [0.0, 1.0], "same length"),
        ([float("inf")], [0.0], "scores must be finite"),
        ([0.0], [float("nan")], "log probabilities must be finite"),
    ],
)
def test_expected_gen_value_score_from_logprobs_rejects_invalid_inputs(scores, logprobs, message):
    with pytest.raises(ValueError, match=message):
        value_model_utils.expected_gen_value_score_from_logprobs(scores, logprobs)


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


def test_gen_value_validation_preserves_near_horizon_and_both_outcomes():
    near_horizon_incorrect = {
        "pairs": [
            _pair(
                [1],
                0.0,
                state_kind="final_action",
                response_tokens_used=8000,
                response_token_limit=8192,
            )
        ]
    }
    mixed_incorrect = {
        "pairs": [_pair([2], 0.0, state_kind="final_action", response_tokens_used=4000)]
    }
    mixed_correct = {
        "pairs": [_pair([2], 1.0, state_kind="final_action", response_tokens_used=4000)]
    }
    ordinary_incorrect = {
        "pairs": [_pair([3], 0.0, state_kind="final_action", response_tokens_used=4000)]
    }

    examples, _ = value_model_utils.build_gen_value_validation_holdout(
        [near_horizon_incorrect, mixed_incorrect, mixed_correct, ordinary_incorrect],
        max_examples=16,
        seed=7,
        prompt_holdout_fraction=0.5,
    )

    sampled = [example for example in examples if example["target_source"] == "single_sample_return"]
    assert {example["target"] for example in sampled} == {0.0, 1.0}
    assert any(value_model_utils.is_gen_value_near_horizon_incorrect(example) for example in sampled)


def test_gen_value_validation_requires_both_sampled_outcome_classes_before_capture():
    one_class = [
        {"target": 0.0, "target_source": "sibling_empirical_return"},
        {"target": 0.0, "target_source": "single_sample_return"},
        {"target": 0.0, "target_source": "single_sample_return"},
    ]
    mixed = one_class + [{"target": 1.0, "target_source": "single_sample_return"}]

    assert not value_model_utils.gen_value_validation_has_both_sampled_outcomes(one_class)
    assert value_model_utils.gen_value_validation_has_both_sampled_outcomes(mixed)


def test_gen_value_checkpoint_detects_legacy_weights_only_layout(tmp_path):
    checkpoint_dir = tmp_path / "global_step75"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "mp_rank_00_model_states.pt").touch()

    assert not value_model_utils.gen_value_checkpoint_has_optimizer_state(str(tmp_path), "global_step75")


def test_gen_value_checkpoint_detects_zero_optimizer_shard(tmp_path):
    checkpoint_dir = tmp_path / "global_step100"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "mp_rank_00_model_states.pt").touch()
    (checkpoint_dir / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").touch()

    assert value_model_utils.gen_value_checkpoint_has_optimizer_state(str(tmp_path), "global_step100")


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


def test_final_action_replay_collapse_preserves_exact_token_loss_numerator():
    replay = {
        "source_pair_id": 4,
        "sequence_ids": [1, 2, 3, 4],
        "generated_ids": [3, 4],
        "rollout_logprobs": [-0.1, -0.2],
        "outcome": 1.0,
        "reward": 0.25,
        "optimizer_selected": True,
        "parsed": True,
        "prediction": 0.8,
        "squared_error": 0.04,
    }
    distinct = dict(replay, source_pair_id=5, reward=-0.5)

    collapsed = value_model_utils.collapse_replayed_gen_value_optimizer_examples(
        [dict(replay), dict(replay), dict(replay), dict(replay), distinct]
    )

    assert len(collapsed) == 2
    assert value_model_utils.gen_value_replay_multiplicity(collapsed[0]) == 4
    assert collapsed[0]["reward"] == 1.0
    assert value_model_utils.gen_value_replay_multiplicity(collapsed[1]) == 1
    assert collapsed[1]["reward"] == -0.5
    logical_token_loss_numerator = sum(
        len(example["generated_ids"]) * float(example["reward"])
        for example in [replay, replay, replay, replay, distinct]
    )
    collapsed_token_loss_numerator = sum(
        len(example["generated_ids"]) * float(example["reward"]) for example in collapsed
    )
    assert collapsed_token_loss_numerator == logical_token_loss_numerator
    packs = value_model_utils.pack_gen_value_examples(collapsed, target_tokens=8)
    flattened_rewards = [
        reward
        for pack in packs
        for reward in value_model_utils.flatten_gen_value_pack(pack)[-1]
    ]
    assert sum(flattened_rewards) == logical_token_loss_numerator
    assert sum(len(pack) for pack in packs) == 2


def test_final_action_replay_collapse_rejects_inconsistent_copies():
    first = {
        "source_pair_id": 2,
        "sequence_ids": [1, 2],
        "generated_ids": [2],
        "rollout_logprobs": [-0.1],
        "outcome": 0.0,
        "reward": 0.3,
        "optimizer_selected": True,
        "parsed": True,
        "prediction": 0.2,
        "squared_error": 0.04,
    }

    with pytest.raises(ValueError, match="identical generated_ids"):
        value_model_utils.collapse_replayed_gen_value_optimizer_examples(
            [first, dict(first, generated_ids=[3])]
        )


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


def test_generative_value_reinforce_outcome_mass_metrics_include_tokens_and_replays():
    metrics = value_model_utils.generative_value_reinforce_outcome_mass_metrics(
        weights=[0.5, -0.25, 0.5],
        outcomes=[1.0, 0.0, 1.0],
        generated_token_counts=[4, 8, 2],
    )

    assert metrics == {
        "gen_value/reinforce_correct_examples": 2.0,
        "gen_value/reinforce_correct_tokens": 6.0,
        "gen_value/reinforce_correct_abs_weight_sum": 1.0,
        "gen_value/reinforce_correct_abs_token_weight_mass": 3.0,
        "gen_value/reinforce_incorrect_examples": 1.0,
        "gen_value/reinforce_incorrect_tokens": 8.0,
        "gen_value/reinforce_incorrect_abs_weight_sum": 0.25,
        "gen_value/reinforce_incorrect_abs_token_weight_mass": 2.0,
        "gen_value/reinforce_correct_abs_token_weight_mass_frac": 0.6,
    }


@pytest.mark.parametrize("token_count", [-1, True, 1.5])
def test_generative_value_reinforce_outcome_mass_metrics_reject_invalid_token_counts(token_count):
    with pytest.raises(ValueError, match="nonnegative integers"):
        value_model_utils.generative_value_reinforce_outcome_mass_metrics([0.5], [1.0], [token_count])


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
    assert metrics["gen_value/validation_prefix_macro_mse"] == pytest.approx((0.34 + 0.13) / 2)
    assert metrics["gen_value/validation_prefix_early_macro_mse"] == pytest.approx((0.64 + 0.25) / 2)
    assert metrics["gen_value/validation_prefix_late_macro_mse"] == pytest.approx((0.04 + 0.01) / 2)


def test_gen_value_validation_macro_mse_is_not_dominated_by_majority_outcome():
    examples = [
        {"kind": "segment_start", "target": 1.0},
        *[{"kind": "segment_start", "target": 0.0} for _ in range(9)],
    ]

    metrics = value_model_utils.gen_value_validation_metrics(examples, [0.0, *([0.0] * 9)])

    assert metrics["gen_value/validation_mse"] == pytest.approx(0.1)
    assert metrics["gen_value/validation_prefix_macro_mse"] == pytest.approx(0.5)


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


def test_read_gen_value_validation_snapshot_validates_required_fields(tmp_path):
    snapshot_path = tmp_path / "snapshot.jsonl"
    snapshot_path.write_text(json.dumps({"prompt": "Score this.", "target": 1.0}) + "\n", encoding="utf-8")

    assert value_model_utils.read_gen_value_validation_snapshot(snapshot_path) == [
        {"prompt": "Score this.", "target": 1.0}
    ]


@pytest.mark.parametrize(
    "row, message", [({"target": 1.0}, "non-empty prompt"), ({"prompt": "Score this."}, "numeric target")]
)
def test_read_gen_value_validation_snapshot_rejects_invalid_rows(tmp_path, row, message):
    snapshot_path = tmp_path / "snapshot.jsonl"
    snapshot_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        value_model_utils.read_gen_value_validation_snapshot(snapshot_path)


def test_gen_value_validation_panel_round_trips_exact_examples(tmp_path):
    examples = [
        {
            "prompt_token_ids": [1, 2, 3],
            "kind": "segment_start",
            "target": 0.75,
            "target_source": "empirical_sibling_return",
        },
        {
            "prompt_token_ids": [4, 5],
            "kind": "final_action",
            "target": 0.0,
            "target_source": "single_sample_return",
        },
    ]

    panel_path = value_model_utils.write_gen_value_validation_panel(str(tmp_path), examples)

    assert panel_path == tmp_path / "gen_value_validation/panel.jsonl"
    assert value_model_utils.read_gen_value_validation_panel(panel_path) == examples


@pytest.mark.parametrize(
    "row, message",
    [
        ({"prompt_token_ids": [], "target": 1.0}, "integer prompt_token_ids"),
        ({"prompt_token_ids": [1, True], "target": 1.0}, "integer prompt_token_ids"),
        ({"prompt_token_ids": [1, 2]}, "numeric target"),
    ],
)
def test_read_gen_value_validation_panel_rejects_invalid_rows(tmp_path, row, message):
    panel_path = tmp_path / "panel.jsonl"
    panel_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        value_model_utils.read_gen_value_validation_panel(panel_path)


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


@pytest.mark.parametrize(("world_size", "max_async_steps", "expected"), [(1, 1, 2), (4, 1, 8), (2, 3, 8)])
def test_gen_value_training_queue_capacity(world_size: int, max_async_steps: int, expected: int):
    assert value_model_utils.gen_value_training_queue_capacity(world_size, max_async_steps) == expected


@pytest.mark.parametrize(("world_size", "max_async_steps"), [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_gen_value_training_queue_capacity_rejects_invalid_values(world_size: int, max_async_steps: int):
    with pytest.raises(ValueError):
        value_model_utils.gen_value_training_queue_capacity(world_size, max_async_steps)


@pytest.mark.parametrize(
    ("policy_step", "latest_trained_step", "max_async_steps", "expected"),
    [(1, 0, 1, True), (2, 1, 1, True), (2, 0, 1, False), (5, 3, 2, True), (6, 3, 2, False)],
)
def test_gen_value_source_step_admission_window(
    policy_step: int, latest_trained_step: int, max_async_steps: int, expected: bool
):
    assert (
        value_model_utils.gen_value_source_step_is_admissible(policy_step, latest_trained_step, max_async_steps)
        is expected
    )


@pytest.mark.parametrize(
    ("policy_step", "latest_trained_step", "max_async_steps"), [(-1, 0, 1), (1, -1, 1), (1, 0, 0)]
)
def test_gen_value_source_step_admission_rejects_invalid_values(
    policy_step: int, latest_trained_step: int, max_async_steps: int
):
    with pytest.raises(ValueError):
        value_model_utils.gen_value_source_step_is_admissible(policy_step, latest_trained_step, max_async_steps)


def test_wait_for_gen_value_source_window_blocks_until_critic_catches_up():
    observed_steps = iter([3, 3, 4])

    latest_trained_step, waited_seconds = value_model_utils.wait_for_gen_value_source_window(
        lambda: next(observed_steps),
        policy_training_step=5,
        max_async_steps=1,
        timeout_s=1.0,
        poll_interval_s=0.0,
    )

    assert latest_trained_step == 4
    assert waited_seconds >= 0.0


def test_wait_for_gen_value_source_window_times_out_explicitly():
    with pytest.raises(TimeoutError, match="latest_trained_policy_step=3"):
        value_model_utils.wait_for_gen_value_source_window(
            lambda: 3,
            policy_training_step=5,
            max_async_steps=1,
            timeout_s=1e-6,
            poll_interval_s=0.0,
        )


def _rollouts_for_steps(*source_steps: int) -> list[dict]:
    return [
        {
            "policy_training_step": source_step,
            "policy_model_version": source_step,
            "identifier": index,
        }
        for index, source_step in enumerate(source_steps)
    ]


def test_select_fresh_gen_value_rollouts_prefers_newest_and_discards_outside_window():
    pending = _rollouts_for_steps(8, 9, 10, 10, 9)

    selected, retained, stale = value_model_utils.select_fresh_gen_value_rollouts(
        pending, batch_size=2, max_async_steps=1
    )

    assert [rollout["identifier"] for rollout in selected] == [2, 3]
    assert [rollout["identifier"] for rollout in retained] == [1, 4]
    assert [rollout["identifier"] for rollout in stale] == [0]


def test_select_fresh_gen_value_rollouts_keeps_inclusive_async_boundary():
    pending = _rollouts_for_steps(8, 9, 10)

    selected, retained, stale = value_model_utils.select_fresh_gen_value_rollouts(
        pending, batch_size=4, max_async_steps=2
    )

    assert selected == []
    assert retained == pending
    assert stale == []


def test_select_fresh_gen_value_rollouts_discards_stale_without_partial_batch():
    pending = _rollouts_for_steps(7, 8, 10)

    selected, retained, stale = value_model_utils.select_fresh_gen_value_rollouts(
        pending, batch_size=3, max_async_steps=1
    )

    assert selected == []
    assert [rollout["policy_training_step"] for rollout in retained] == [10]
    assert [rollout["policy_training_step"] for rollout in stale] == [7, 8]


def test_select_fresh_gen_value_rollouts_reuses_frozen_policy_batches():
    pending = [
        {"policy_training_step": step, "policy_model_version": 0, "identifier": step}
        for step in (1, 12, 25)
    ]

    selected, retained, stale = value_model_utils.select_fresh_gen_value_rollouts(
        pending, batch_size=2, max_async_steps=1
    )

    assert [rollout["identifier"] for rollout in selected] == [12, 25]
    assert [rollout["identifier"] for rollout in retained] == [1]
    assert stale == []


def test_select_fresh_gen_value_rollouts_uses_model_version_before_batch_recency():
    pending = [
        {"policy_training_step": 20, "policy_model_version": 3, "identifier": "old-weights-new-batch"},
        {"policy_training_step": 18, "policy_model_version": 4, "identifier": "new-weights-old-batch"},
    ]

    selected, retained, stale = value_model_utils.select_fresh_gen_value_rollouts(
        pending, batch_size=1, max_async_steps=1
    )

    assert [rollout["identifier"] for rollout in selected] == ["new-weights-old-batch"]
    assert [rollout["identifier"] for rollout in retained] == ["old-weights-new-batch"]
    assert stale == []


def test_select_fresh_gen_value_rollouts_does_not_discard_for_critic_version_age():
    pending = [
        {
            "policy_training_step": 20,
            "policy_model_version": 10,
            "critic_version": 1,
            "identifier": "old-critic-fresh-policy",
        },
        {
            "policy_training_step": 19,
            "policy_model_version": 10,
            "critic_version": 9,
            "identifier": "newer-critic-fresh-policy",
        },
    ]

    selected, retained, stale = value_model_utils.select_fresh_gen_value_rollouts(
        pending, batch_size=2, max_async_steps=1
    )

    assert [rollout["identifier"] for rollout in selected] == [
        "old-critic-fresh-policy",
        "newer-critic-fresh-policy",
    ]
    assert retained == []
    assert stale == []


def test_select_fresh_gen_value_rollouts_requires_policy_model_version():
    with pytest.raises(ValueError, match="policy_model_version"):
        value_model_utils.select_fresh_gen_value_rollouts(
            [{"policy_training_step": 1}], batch_size=1, max_async_steps=1
        )


def test_select_fresh_gen_value_rollouts_does_not_regress_newest_seen_version():
    pending = [{"policy_training_step": 20, "policy_model_version": 8, "identifier": "old-remainder"}]

    selected, retained, stale = value_model_utils.select_fresh_gen_value_rollouts(
        pending,
        batch_size=1,
        max_async_steps=1,
        newest_policy_model_version=10,
    )

    assert selected == []
    assert retained == []
    assert stale == pending


def test_gen_value_training_progress_waits_for_all_admitted_rollouts():
    progress = value_model_utils.GenValueTrainingProgressState(latest_trained_policy_step=3, policy_world_size=2)

    progress.register_admitted_policy_step(policy_training_step=4, policy_rank=0, num_rollouts=2)
    progress.register_admitted_policy_step(policy_training_step=4, policy_rank=1, num_rollouts=1)
    progress.record_trained_policy_steps([4, 4])
    assert progress.get_latest_trained_policy_step() == 4
    assert progress.get_latest_processed_policy_step() == 3

    progress.record_trained_policy_steps([4])
    assert progress.get_latest_trained_policy_step() == 4


def test_gen_value_training_progress_counts_trained_and_discarded_rollouts():
    progress = value_model_utils.GenValueTrainingProgressState(latest_trained_policy_step=3, policy_world_size=2)

    progress.register_admitted_policy_step(policy_training_step=4, policy_rank=0, num_rollouts=2)
    progress.register_admitted_policy_step(policy_training_step=4, policy_rank=1, num_rollouts=1)
    progress.record_trained_policy_steps([4])
    progress.record_discarded_policy_steps([4])
    assert progress.get_latest_processed_policy_step() == 3

    progress.record_discarded_policy_steps([4])
    assert progress.get_latest_processed_policy_step() == 4
    assert progress.get_rollout_accounting() == {
        "latest_processed_policy_step": 4,
        "latest_trained_policy_step": 4,
        "admitted_rollouts": 3,
        "trained_rollouts": 1,
        "discarded_rollouts": 2,
    }


def test_gen_value_training_progress_handles_training_before_registration():
    progress = value_model_utils.GenValueTrainingProgressState(latest_trained_policy_step=4, policy_world_size=2)

    progress.record_trained_policy_steps([5, 5])
    assert progress.get_latest_trained_policy_step() == 5
    progress.register_admitted_policy_step(policy_training_step=5, policy_rank=0, num_rollouts=1)
    assert progress.get_latest_processed_policy_step() == 4

    progress.register_admitted_policy_step(policy_training_step=5, policy_rank=1, num_rollouts=1)
    assert progress.get_latest_processed_policy_step() == 5


def test_gen_value_training_progress_rejects_duplicate_rank_registration():
    progress = value_model_utils.GenValueTrainingProgressState(latest_trained_policy_step=0, policy_world_size=1)
    progress.register_admitted_policy_step(policy_training_step=1, policy_rank=0, num_rollouts=1)

    with pytest.raises(ValueError, match="more than once"):
        progress.register_admitted_policy_step(policy_training_step=1, policy_rank=0, num_rollouts=1)
