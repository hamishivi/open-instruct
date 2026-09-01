# ruff: noqa: E402, I001
"""Unit tests for grpo_fast_genvalue helpers (no GPU required)."""

from queue import Queue
import threading
import time
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("vllm")

from open_instruct import grpo_fast
from open_instruct.dataset_transformation import INPUT_IDS_PROMPT_KEY, TokenizerConfig
from open_instruct.grpo_fast_genvalue import (
    GenValueTrainingProgress,
    GenValueExperimentConfig,
    _build_sample_scoring_prompts,
    _drain_gen_value_metrics,
    _gen_value_scoring_loop,
    _hold_gen_value_training_for_checkpoint,
    _put_gen_value_metrics,
    _resolve_gen_value_model,
    _resolve_gen_value_train_pack_length,
    _resolve_gen_value_tokenizer,
    _sync_gen_value_weights,
    _wait_for_gen_value_publish_barrier,
)
from open_instruct.value_model_utils import segment_rollout


# ── segment_rollout ────────────────────────────────────────────────────────────


def test_segment_rollout_fixed_basic():
    tokens = list(range(10))
    result = segment_rollout(tokens, None, mode="fixed", fixed_chunk_size=3)
    # Inclusive ends for 3-token chunks, plus the one-token final chunk.
    assert result == [2, 5, 8, 9]


def test_segment_rollout_fixed_exact_multiple():
    tokens = list(range(6))
    result = segment_rollout(tokens, None, mode="fixed", fixed_chunk_size=3)
    assert result == [2, 5]


def test_segment_rollout_fixed_terminal_appended():
    tokens = list(range(5))
    result = segment_rollout(tokens, None, mode="fixed", fixed_chunk_size=10)
    assert result == [4], "should append terminal index when no boundary falls before it"


def test_segment_rollout_empty():
    assert segment_rollout([], None, mode="fixed") == []
    assert segment_rollout([], None, mode="sae", sae_threshold=0.5) == []


def test_segment_rollout_sae_basic():
    tokens = [0, 1, 2, 3, 4]
    # logprob < log(0.5) ≈ -0.693 → tokens 1 and 3 are boundaries
    logprobs = [0.0, -1.0, -0.1, -1.5, -0.2]
    result = segment_rollout(tokens, logprobs, mode="sae", sae_threshold=0.5)
    assert 1 in result
    assert 3 in result
    assert result[-1] == 4, "terminal should always be included"


def test_segment_rollout_sae_no_low_prob():
    tokens = [0, 1, 2]
    logprobs = [0.0, -0.1, -0.2]  # all above threshold
    result = segment_rollout(tokens, logprobs, mode="sae", sae_threshold=0.01)
    # all probs > 0.01, so only terminal boundary
    assert result == [2]


def test_segment_rollout_sae_missing_logprobs():
    with pytest.raises(ValueError, match="SAE segmentation requires response_logprobs"):
        segment_rollout([1, 2, 3], None, mode="sae")


def test_segment_rollout_terminal_not_duplicated():
    tokens = list(range(3))
    logprobs = [0.0, -1.0, -1.0]  # last two are boundaries
    result = segment_rollout(tokens, logprobs, mode="sae", sae_threshold=0.5)
    # should not have 2 appearing twice
    assert result.count(2) == 1


def test_segment_rollout_fixed_sorted():
    tokens = list(range(20))
    result = segment_rollout(tokens, None, mode="fixed", fixed_chunk_size=4)
    assert result == sorted(result)


# ── GenValueExperimentConfig validation ───────────────────────────────────────


def _base_kwargs():
    """Minimal valid kwargs for GenValueExperimentConfig (all fields have defaults)."""
    return dict(
        use_generative_value_model=True,
        use_value_model=True,
        gen_value_segmentation="fixed",
        gen_value_chunk_size=256,
        gen_value_score_min=0.0,
        gen_value_score_max=10.0,
        gen_value_conditioning="none",
    )


def test_genvalue_config_valid():
    kwargs = _base_kwargs()
    cfg = GenValueExperimentConfig(**kwargs)
    assert cfg.use_generative_value_model is True
    assert cfg.gen_value_segmentation == "fixed"


def test_genvalue_tokenizer_defaults_to_policy_and_allows_independent_override():
    policy_tokenizer = TokenizerConfig(tokenizer_name_or_path="policy-tokenizer", tokenizer_revision="policy-rev")
    default_cfg = GenValueExperimentConfig(**_base_kwargs())
    assert _resolve_gen_value_tokenizer(default_cfg, policy_tokenizer) == ("policy-tokenizer", "policy-rev")

    override_cfg = GenValueExperimentConfig(
        **_base_kwargs(),
        gen_value_tokenizer_name_or_path="critic-tokenizer",
        gen_value_tokenizer_revision="critic-rev",
    )
    assert _resolve_gen_value_tokenizer(override_cfg, policy_tokenizer) == ("critic-tokenizer", "critic-rev")

    override_without_revision = GenValueExperimentConfig(
        **_base_kwargs(), gen_value_tokenizer_name_or_path="critic-tokenizer"
    )
    assert _resolve_gen_value_tokenizer(override_without_revision, policy_tokenizer) == ("critic-tokenizer", None)


def test_genvalue_model_defaults_to_policy_and_allows_independent_revision():
    policy_model = MagicMock(model_name_or_path="policy-model", model_revision="policy-rev")
    policy_model.model_name_or_path = "policy-model"
    policy_model.model_revision = "policy-rev"

    default_cfg = GenValueExperimentConfig(**_base_kwargs())
    assert _resolve_gen_value_model(default_cfg, policy_model) == ("policy-model", "policy-rev")

    independent_cfg = GenValueExperimentConfig(
        **_base_kwargs(), gen_value_model_name_or_path="critic-model", gen_value_model_revision="critic-rev"
    )
    assert _resolve_gen_value_model(independent_cfg, policy_model) == ("critic-model", "critic-rev")

    independent_default_revision_cfg = GenValueExperimentConfig(
        **_base_kwargs(), gen_value_model_name_or_path="critic-model"
    )
    assert _resolve_gen_value_model(independent_default_revision_cfg, policy_model) == ("critic-model", None)

    policy_repo_other_revision_cfg = GenValueExperimentConfig(**_base_kwargs(), gen_value_model_revision="critic-rev")
    assert _resolve_gen_value_model(policy_repo_other_revision_cfg, policy_model) == ("policy-model", "critic-rev")


def test_genvalue_config_requires_flag():
    kwargs = _base_kwargs()
    kwargs["use_generative_value_model"] = False
    with pytest.raises(ValueError, match="requires --use_generative_value_model"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_requires_value_model_path():
    kwargs = _base_kwargs()
    kwargs["use_value_model"] = False
    with pytest.raises(ValueError, match="requires --use_value_model"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_requires_serving_engine():
    kwargs = _base_kwargs()
    kwargs["gen_value_vllm_num_engines"] = 0
    with pytest.raises(ValueError, match="gen_value_vllm_num_engines must be > 0"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_rejects_negative_reinforce_coef():
    kwargs = _base_kwargs()
    kwargs["gen_value_reinforce_coef"] = -0.1
    with pytest.raises(ValueError, match="gen_value_reinforce_coef must be >= 0"):
        GenValueExperimentConfig(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gen_value_batch_size", 0),
        ("gen_value_train_target_examples_per_update", 0),
        ("gen_value_vllm_tensor_parallel_size", 0),
        ("gen_value_max_segments", 0),
        ("gen_value_max_new_tokens", 0),
        ("gen_value_train_pack_length", 0),
        ("gen_value_learning_rate", 0.0),
    ],
)
def test_genvalue_config_rejects_nonpositive_settings(field: str, value: int):
    kwargs = _base_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_bad_segmentation():
    kwargs = _base_kwargs()
    kwargs["gen_value_segmentation"] = "wavelet"
    with pytest.raises(ValueError, match="must be 'sae' or 'fixed'"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_sae_requires_use_sae():
    kwargs = _base_kwargs()
    kwargs["gen_value_segmentation"] = "sae"
    kwargs["use_sae"] = False
    with pytest.raises(ValueError, match="requires --use_sae"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_sae_with_use_sae():
    kwargs = _base_kwargs()
    kwargs["gen_value_segmentation"] = "sae"
    kwargs["use_sae"] = True
    kwargs["use_value_model"] = True  # use_sae requires use_value_model
    cfg = GenValueExperimentConfig(**kwargs)
    assert cfg.gen_value_segmentation == "sae"


def test_genvalue_config_bad_chunk_size():
    kwargs = _base_kwargs()
    kwargs["gen_value_chunk_size"] = 0
    with pytest.raises(ValueError, match="must be > 0"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_requires_positive_temperature():
    kwargs = _base_kwargs()
    kwargs["gen_value_temperature"] = 0.0
    with pytest.raises(ValueError, match="gen_value_temperature must be > 0"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_allows_greedy_inference_temperature():
    kwargs = _base_kwargs()
    kwargs["gen_value_inference_temperature"] = 0.0
    cfg = GenValueExperimentConfig(**kwargs)
    assert cfg.gen_value_inference_temperature == 0.0


def test_genvalue_config_rejects_negative_inference_temperature():
    kwargs = _base_kwargs()
    kwargs["gen_value_inference_temperature"] = -0.1
    with pytest.raises(ValueError, match="gen_value_inference_temperature must be >= 0"):
        GenValueExperimentConfig(**kwargs)


def test_actor_values_and_reinforce_samples_use_independent_temperatures(monkeypatch):
    trainer_cls = grpo_fast.PolicyTrainerRayProcess.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer._gen_value_version = 7
    trainer.args = SimpleNamespace(
        gen_value_max_new_tokens=16,
        gen_value_temperature=1.0,
        gen_value_inference_temperature=0.0,
    )

    engine = MagicMock()
    engine.generate_request_outputs.remote.side_effect = lambda prompts, **kwargs: [
        {"temperature": kwargs["temperature"], "prompt": prompt} for prompt in prompts
    ]
    trainer._gen_value_engines = [engine]
    monkeypatch.setattr(grpo_fast, "ray_get_with_progress", lambda refs, **_: (refs, None))

    def finish(_, request, outputs):
        temperature = float(outputs[0]["temperature"])
        return torch.tensor([temperature]), [{"request_output": outputs[0], "request": request}]

    trainer._finish_gen_value_scoring_request = MethodType(finish, trainer)
    results = trainer._score_gen_value_requests([{"prompts": ["a", "b"]}])

    assert [call.kwargs["temperature"] for call in engine.generate_request_outputs.remote.call_args_list] == [0.0, 1.0]
    values, pairs = results[0]
    assert values.tolist() == [0.0]
    assert pairs[0]["request_output"]["temperature"] == 1.0
    assert pairs[0]["critic_version"] == 7


def test_matching_critic_temperatures_reuse_one_completion(monkeypatch):
    trainer_cls = grpo_fast.PolicyTrainerRayProcess.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer._gen_value_version = 3
    trainer.args = SimpleNamespace(
        gen_value_max_new_tokens=16,
        gen_value_temperature=0.7,
        gen_value_inference_temperature=None,
    )

    engine = MagicMock()
    engine.generate_request_outputs.remote.side_effect = lambda prompts, **kwargs: [
        {"temperature": kwargs["temperature"], "prompt": prompt} for prompt in prompts
    ]
    trainer._gen_value_engines = [engine]
    monkeypatch.setattr(grpo_fast, "ray_get_with_progress", lambda refs, **_: (refs, None))

    def finish(_, request, outputs):
        temperature = float(outputs[0]["temperature"])
        return torch.tensor([temperature]), [{"request_output": outputs[0], "request": request}]

    trainer._finish_gen_value_scoring_request = MethodType(finish, trainer)
    values, pairs = trainer._score_gen_value_requests([{"prompts": ["a"]}])[0]

    assert engine.generate_request_outputs.remote.call_count == 1
    assert values.tolist() == pytest.approx([0.7])
    assert pairs[0]["request_output"]["temperature"] == pytest.approx(0.7)
    assert pairs[0]["critic_version"] == 3


def test_critic_pool_shards_round_robin_and_restores_request_order(monkeypatch):
    trainer_cls = grpo_fast.PolicyTrainerRayProcess.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer._gen_value_version = 11
    trainer.args = SimpleNamespace(
        gen_value_max_new_tokens=16,
        gen_value_temperature=0.0,
        gen_value_inference_temperature=0.0,
    )

    engines = []
    for _ in range(5):
        engine = MagicMock()
        engine.generate_request_outputs.remote.side_effect = lambda prompts, **_: list(prompts)
        engines.append(engine)
    trainer._gen_value_engines = engines
    monkeypatch.setattr(grpo_fast, "ray_get_with_progress", lambda refs, **_: (refs, None))

    def finish(_, request, outputs):
        return torch.zeros(len(outputs)), [{"prompt": output, "request": request} for output in outputs]

    trainer._finish_gen_value_scoring_request = MethodType(finish, trainer)
    requests = [
        {"prompts": [f"p{i}" for i in range(7)]},
        {"prompts": [f"p{i}" for i in range(7, 12)]},
    ]

    results = trainer._score_gen_value_requests(requests)

    expected_buckets = [
        ["p0", "p5", "p10"],
        ["p1", "p6", "p11"],
        ["p2", "p7"],
        ["p3", "p8"],
        ["p4", "p9"],
    ]
    for engine, expected_prompts in zip(engines, expected_buckets, strict=True):
        engine.generate_request_outputs.remote.assert_called_once()
        assert engine.generate_request_outputs.remote.call_args.args[0] == expected_prompts
    assert [[pair["prompt"] for pair in pairs] for _, pairs in results] == [
        [f"p{i}" for i in range(7)],
        [f"p{i}" for i in range(7, 12)],
    ]
    assert {pair["critic_version"] for _, pairs in results for pair in pairs} == {11}


def test_gen_value_generation_progress_snapshot_does_not_wait(monkeypatch):
    trainer_cls = grpo_fast.PolicyTrainerRayProcess.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)

    class RemoteMethod:
        def remote(self):
            return 4

    trainer._gen_value_training_progress = SimpleNamespace(
        get_latest_processed_policy_step=RemoteMethod()
    )
    monkeypatch.setattr(grpo_fast.ray, "get", lambda value: value)

    assert trainer._get_gen_value_processed_policy_step() == 4


def test_gen_value_latest_enqueue_evicts_oldest_shard_and_records_discard(monkeypatch):
    trainer_cls = grpo_fast.PolicyTrainerRayProcess.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer._gen_value_training_queue = Queue(maxsize=1)
    old_rollouts = [{"policy_training_step": 7, "policy_model_version": 5, "identifier": "old"}]
    new_rollouts = [{"policy_training_step": 9, "policy_model_version": 6, "identifier": "new"}]
    trainer._gen_value_training_queue.put(old_rollouts)
    discarded_steps: list[int] = []

    class RemoteMethod:
        def remote(self, policy_steps):
            discarded_steps.extend(policy_steps)

    trainer._gen_value_training_progress = SimpleNamespace(record_discarded_policy_steps=RemoteMethod())
    monkeypatch.setattr(grpo_fast.ray, "get", lambda value: value)

    evicted = trainer._enqueue_gen_value_rollouts_latest(new_rollouts)

    assert evicted == old_rollouts
    assert trainer._gen_value_training_queue.get(block=False) == new_rollouts
    assert discarded_steps == [7]


def test_gen_value_publish_barrier_releases_after_serving_version_advances():
    stop_event = threading.Event()
    publish_complete_event = threading.Event()
    progress_lock = threading.Lock()
    progress_state = {"synced_version": 1, "pending_publish_version": None}
    result: list[float] = []

    waiter = threading.Thread(
        target=lambda: result.append(
            _wait_for_gen_value_publish_barrier(
                2, 1, stop_event, publish_complete_event, progress_state, progress_lock
            )
        )
    )
    waiter.start()
    deadline = time.monotonic() + 1.0
    while progress_state["pending_publish_version"] != 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert progress_state["pending_publish_version"] == 2
    assert waiter.is_alive()

    with progress_lock:
        progress_state["synced_version"] = 2
    publish_complete_event.set()
    waiter.join(timeout=1.0)

    assert not waiter.is_alive()
    assert result and result[0] >= 0.0
    assert progress_state["pending_publish_version"] is None


def test_gen_value_publish_barrier_skips_non_publish_versions():
    progress_state = {"synced_version": 0, "pending_publish_version": None}
    wait_seconds = _wait_for_gen_value_publish_barrier(
        1, 2, threading.Event(), threading.Event(), progress_state, threading.Lock()
    )

    assert wait_seconds == 0.0
    assert progress_state["pending_publish_version"] is None


def test_gen_value_publish_barrier_shutdown_does_not_deadlock():
    stop_event = threading.Event()
    stop_event.set()
    progress_state = {"synced_version": 1, "pending_publish_version": None}

    wait_seconds = _wait_for_gen_value_publish_barrier(
        2, 1, stop_event, threading.Event(), progress_state, threading.Lock()
    )

    assert wait_seconds >= 0.0
    assert progress_state["pending_publish_version"] is None


def test_gen_value_checkpoint_hold_acknowledges_idle_trainer_until_release():
    pause_event = threading.Event()
    paused_event = threading.Event()
    stop_event = threading.Event()
    pause_event.set()
    result: list[float] = []

    waiter = threading.Thread(
        target=lambda: result.append(
            _hold_gen_value_training_for_checkpoint(pause_event, paused_event, stop_event)
        )
    )
    waiter.start()

    assert paused_event.wait(timeout=1.0)
    assert waiter.is_alive()
    pause_event.clear()
    waiter.join(timeout=1.0)

    assert not waiter.is_alive()
    assert not paused_event.is_set()
    assert result and result[0] >= 0.0


def test_gen_value_checkpoint_hold_shutdown_releases_waiter():
    pause_event = threading.Event()
    paused_event = threading.Event()
    stop_event = threading.Event()
    pause_event.set()

    waiter = threading.Thread(
        target=_hold_gen_value_training_for_checkpoint,
        args=(pause_event, paused_event, stop_event),
    )
    waiter.start()

    assert paused_event.wait(timeout=1.0)
    stop_event.set()
    waiter.join(timeout=1.0)

    assert not waiter.is_alive()
    assert not paused_event.is_set()


def test_gen_value_checkpoint_hold_is_noop_without_pause():
    paused_event = threading.Event()

    wait_seconds = _hold_gen_value_training_for_checkpoint(
        threading.Event(), paused_event, threading.Event()
    )

    assert wait_seconds == 0.0
    assert not paused_event.is_set()


def test_gen_value_training_progress_is_monotonic():
    progress_cls = GenValueTrainingProgress.__ray_metadata__.modified_class
    progress = progress_cls(latest_trained_policy_step=3, policy_world_size=2)

    assert progress.get_latest_trained_policy_step() == 3
    progress.register_admitted_policy_step(policy_training_step=4, policy_rank=0, num_rollouts=2)
    progress.register_admitted_policy_step(policy_training_step=4, policy_rank=1, num_rollouts=1)
    progress.record_trained_policy_steps([4, 4])
    assert progress.get_latest_trained_policy_step() == 3
    progress.record_trained_policy_steps([4])
    assert progress.get_latest_trained_policy_step() == 4

    # Mixed critic batches may begin training the next source step before all
    # ranks finish registering it. Do not advance until admission is complete.
    progress.record_trained_policy_steps([5, 5])
    progress.register_admitted_policy_step(policy_training_step=5, policy_rank=0, num_rollouts=1)
    assert progress.get_latest_trained_policy_step() == 4
    progress.register_admitted_policy_step(policy_training_step=5, policy_rank=1, num_rollouts=1)
    assert progress.get_latest_trained_policy_step() == 5

    with pytest.raises(ValueError, match="already complete"):
        progress.record_trained_policy_steps([4])


def test_gen_value_training_progress_rejects_duplicate_rank_registration():
    progress_cls = GenValueTrainingProgress.__ray_metadata__.modified_class
    progress = progress_cls(latest_trained_policy_step=0, policy_world_size=1)
    progress.register_admitted_policy_step(policy_training_step=1, policy_rank=0, num_rollouts=1)
    with pytest.raises(ValueError, match="more than once"):
        progress.register_admitted_policy_step(policy_training_step=1, policy_rank=0, num_rollouts=1)


def test_gen_value_training_progress_rejects_negative_initial_step():
    progress_cls = GenValueTrainingProgress.__ray_metadata__.modified_class
    with pytest.raises(ValueError, match="must be nonnegative"):
        progress_cls(latest_trained_policy_step=-1, policy_world_size=1)


def test_genvalue_config_bad_score_range():
    kwargs = _base_kwargs()
    kwargs["gen_value_score_max"] = kwargs["gen_value_score_min"]
    with pytest.raises(ValueError, match="score_max must be greater"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_bad_conditioning():
    kwargs = _base_kwargs()
    kwargs["gen_value_conditioning"] = "oracle"
    with pytest.raises(ValueError, match="must be one of"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_valid_conditionings():
    for cond in ("none", "gt", "correct_demo", "rollout_context"):
        kwargs = _base_kwargs()
        kwargs["gen_value_conditioning"] = cond
        cfg = GenValueExperimentConfig(**kwargs)
        assert cfg.gen_value_conditioning == cond


def test_genvalue_config_rejects_context_without_prompt_room():
    kwargs = _base_kwargs()
    kwargs["gen_value_max_new_tokens"] = 1024
    kwargs["gen_value_max_model_len"] = 1024
    with pytest.raises(ValueError, match="must be greater than --gen_value_max_new_tokens"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_rejects_training_pack_larger_than_declared_context():
    kwargs = _base_kwargs()
    kwargs["gen_value_max_model_len"] = 32768
    kwargs["gen_value_train_pack_length"] = 32769
    with pytest.raises(ValueError, match="cannot exceed --gen_value_max_model_len"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_train_pack_length_defaults_to_policy_and_allows_critic_override():
    default_cfg = GenValueExperimentConfig(**_base_kwargs())
    assert _resolve_gen_value_train_pack_length(default_cfg, 10240, 32768) == 10240

    override_cfg = GenValueExperimentConfig(**_base_kwargs(), gen_value_train_pack_length=32768)
    assert _resolve_gen_value_train_pack_length(override_cfg, 10240, 32768) == 32768


def test_genvalue_train_pack_length_rejects_effective_context_overflow():
    cfg = GenValueExperimentConfig(**_base_kwargs(), gen_value_train_pack_length=32768)
    with pytest.raises(ValueError, match="effective context limit"):
        _resolve_gen_value_train_pack_length(cfg, 10240, 16384)


def test_diagnostic_scoring_exception_propagates(monkeypatch):
    trigger = threading.Event()
    trigger.set()
    stop = threading.Event()
    cfg = GenValueExperimentConfig(**_base_kwargs())

    def fail_prompt_build(*_args, **_kwargs):
        raise RuntimeError("diagnostic failed")

    monkeypatch.setattr("open_instruct.grpo_fast_genvalue._build_sample_scoring_prompts", fail_prompt_build)
    with pytest.raises(RuntimeError, match="diagnostic failed"):
        _gen_value_scoring_loop(
            cfg,
            MagicMock(),
            [],
            [MagicMock()],
            trigger,
            stop,
            threading.Lock(),
            Queue(),
            {"synced_version": 0},
            threading.Lock(),
        )


def test_multiple_critic_updates_are_token_and_example_weighted():
    metrics_q = Queue()
    _put_gen_value_metrics(
        metrics_q,
        {
            "gen_value/reinforce_loss": 1.0,
            "gen_value/reward_mean": 0.0,
            "gen_value/optimization_reward_mean": 0.0,
            "gen_value/mse": 0.25,
            "gen_value/optimization_mse": 0.25,
            "gen_value/train_tokens": 2,
            "gen_value/train_examples": 1,
            "gen_value/reinforce_correct_examples": 1,
            "gen_value/reinforce_incorrect_examples": 0,
            "gen_value/reinforce_correct_tokens": 2,
            "gen_value/reinforce_incorrect_tokens": 0,
            "gen_value/reinforce_correct_abs_weight_sum": 0.5,
            "gen_value/reinforce_incorrect_abs_weight_sum": 0.0,
            "gen_value/reinforce_correct_abs_token_weight_mass": 1.0,
            "gen_value/reinforce_incorrect_abs_token_weight_mass": 0.0,
            "gen_value/reinforce_correct_abs_token_weight_mass_frac": 1.0,
            "gen_value/parsed_examples": 1,
            "gen_value/unique_examples": 1,
            "gen_value/unique_parsed_examples": 1,
            "gen_value/shared_state_unique_examples": 1,
            "gen_value/shared_state_groups": 1,
            "gen_value/shared_state_pooled_groups": 0,
            "gen_value/shared_state_pooled_examples": 0,
            "gen_value/shared_state_changed_examples": 0,
            "gen_value/train_packs": 1,
            "gen_value/train_pack_tokens": 100,
            "gen_value/train_examples_per_pack": 1.0,
            "gen_value/train_mean_pack_tokens": 100.0,
            "gen_value/train_max_pack_tokens": 100,
            "gen_value/batch_rollouts": 1,
            "gen_value/source_value_version_min": 4,
            "gen_value/source_value_version_max": 4,
            "gen_value/newest_available_source_policy_step": 11,
            "gen_value/source_policy_lag_min": 0,
            "gen_value/source_policy_lag_max": 1,
            "gen_value/discarded_stale_rollouts_since_update": 1,
        },
        "REINFORCE",
    )
    _put_gen_value_metrics(
        metrics_q,
        {
            "gen_value/reinforce_loss": 3.0,
            "gen_value/reward_mean": 1.0,
            "gen_value/optimization_reward_mean": 1.0,
            "gen_value/mse": 0.5,
            "gen_value/optimization_mse": 0.5,
            "gen_value/train_tokens": 6,
            "gen_value/train_examples": 3,
            "gen_value/reinforce_correct_examples": 1,
            "gen_value/reinforce_incorrect_examples": 2,
            "gen_value/reinforce_correct_tokens": 2,
            "gen_value/reinforce_incorrect_tokens": 4,
            "gen_value/reinforce_correct_abs_weight_sum": 1.0,
            "gen_value/reinforce_incorrect_abs_weight_sum": 1.5,
            "gen_value/reinforce_correct_abs_token_weight_mass": 2.0,
            "gen_value/reinforce_incorrect_abs_token_weight_mass": 3.0,
            "gen_value/reinforce_correct_abs_token_weight_mass_frac": 0.4,
            "gen_value/parsed_examples": 2,
            "gen_value/unique_examples": 2,
            "gen_value/unique_parsed_examples": 1,
            "gen_value/shared_state_unique_examples": 2,
            "gen_value/shared_state_groups": 1,
            "gen_value/shared_state_pooled_groups": 1,
            "gen_value/shared_state_pooled_examples": 2,
            "gen_value/shared_state_changed_examples": 1,
            "gen_value/train_packs": 2,
            "gen_value/train_pack_tokens": 300,
            "gen_value/train_examples_per_pack": 1.5,
            "gen_value/train_mean_pack_tokens": 150.0,
            "gen_value/train_max_pack_tokens": 180,
            "gen_value/batch_rollouts": 2,
            "gen_value/source_value_version_min": 5,
            "gen_value/source_value_version_max": 7,
            "gen_value/newest_available_source_policy_step": 13,
            "gen_value/source_policy_lag_min": 1,
            "gen_value/source_policy_lag_max": 3,
            "gen_value/discarded_stale_rollouts_since_update": 2,
        },
        "REINFORCE",
    )

    metrics = _drain_gen_value_metrics(metrics_q)

    assert metrics["gen_value/reinforce_loss"] == pytest.approx(2.5)
    assert metrics["gen_value/reward_mean"] == pytest.approx(2 / 3)
    assert metrics["gen_value/optimization_reward_mean"] == pytest.approx(0.75)
    assert metrics["gen_value/mse"] == pytest.approx((0.25 + 0.5) / 2)
    assert metrics["gen_value/optimization_mse"] == pytest.approx((0.25 + 2 * 0.5) / 3)
    assert metrics["gen_value/train_tokens"] == 8
    assert metrics["gen_value/train_examples"] == 4
    assert metrics["gen_value/reinforce_correct_examples"] == 2
    assert metrics["gen_value/reinforce_incorrect_examples"] == 2
    assert metrics["gen_value/reinforce_correct_tokens"] == 4
    assert metrics["gen_value/reinforce_incorrect_tokens"] == 4
    assert metrics["gen_value/reinforce_correct_abs_weight_sum"] == pytest.approx(1.5)
    assert metrics["gen_value/reinforce_incorrect_abs_weight_sum"] == pytest.approx(1.5)
    assert metrics["gen_value/reinforce_correct_abs_token_weight_mass"] == pytest.approx(3.0)
    assert metrics["gen_value/reinforce_incorrect_abs_token_weight_mass"] == pytest.approx(3.0)
    assert metrics["gen_value/reinforce_correct_abs_token_weight_mass_frac"] == pytest.approx(0.5)
    assert metrics["gen_value/parsed_examples"] == 3
    assert metrics["gen_value/unique_examples"] == 3
    assert metrics["gen_value/unique_parsed_examples"] == 2
    assert metrics["gen_value/shared_state_unique_examples"] == 3
    assert metrics["gen_value/shared_state_groups"] == 2
    assert metrics["gen_value/shared_state_pooled_groups"] == 1
    assert metrics["gen_value/shared_state_pooled_examples"] == 2
    assert metrics["gen_value/shared_state_changed_examples"] == 1
    assert metrics["gen_value/train_packs"] == 3
    assert metrics["gen_value/train_pack_tokens"] == 400
    assert metrics["gen_value/train_examples_per_pack"] == pytest.approx(4 / 3)
    assert metrics["gen_value/train_mean_pack_tokens"] == pytest.approx(400 / 3)
    assert metrics["gen_value/train_max_pack_tokens"] == 180
    assert metrics["gen_value/batch_rollouts"] == 3
    assert metrics["gen_value/source_value_version_min"] == 4
    assert metrics["gen_value/source_value_version_max"] == 7
    assert metrics["gen_value/source_value_version_spread"] == 3
    assert metrics["gen_value/newest_available_source_policy_step"] == 13
    assert metrics["gen_value/source_policy_lag_min"] == 0
    assert metrics["gen_value/source_policy_lag_max"] == 3
    assert metrics["gen_value/discarded_stale_rollouts_since_update"] == 3


def test_worker_async_queue_metrics_use_global_reductions():
    metrics = grpo_fast.aggregate_worker_scalar_metrics(
        (
            {
                "_token_count": 1.0,
                "loss/policy_avg": 1.0,
                "gen_value/enqueued_rollouts": 3.0,
                "gen_value/queue_evicted_rollouts": 2.0,
                "gen_value/queue_evicted_source_step_min": 4.0,
                "gen_value/queue_evicted_source_step_max": 5.0,
                "gen_value/training_queue_size": 4.0,
                "gen_value/training_queue_backpressure_seconds": 0.1,
            },
            {
                "_token_count": 3.0,
                "loss/policy_avg": 3.0,
                "gen_value/enqueued_rollouts": 5.0,
                "gen_value/queue_evicted_rollouts": 1.0,
                "gen_value/queue_evicted_source_step_min": 7.0,
                "gen_value/queue_evicted_source_step_max": 9.0,
                "gen_value/training_queue_size": 6.0,
                "gen_value/training_queue_backpressure_seconds": 0.3,
            },
        )
    )

    assert metrics["loss/policy_avg"] == pytest.approx(2.5)
    assert metrics["gen_value/enqueued_rollouts"] == 8
    assert metrics["gen_value/queue_evicted_rollouts"] == 3
    assert metrics["gen_value/queue_evicted_source_step_min"] == 4
    assert metrics["gen_value/queue_evicted_source_step_max"] == 9
    assert metrics["gen_value/training_queue_size"] == 6
    assert metrics["gen_value/training_queue_backpressure_seconds"] == pytest.approx(0.3)


def test_failed_weight_transfer_still_wakes_critic_engines(monkeypatch):
    trainer = MagicMock()
    engine = MagicMock()
    wait_calls = 0

    def fake_wait(*_args, **_kwargs):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            raise RuntimeError("transfer failed")
        return [None], [0.0]

    monkeypatch.setattr("open_instruct.grpo_fast_genvalue.utils.ray_get_with_progress", fake_wait)

    with pytest.raises(RuntimeError, match="transfer failed"):
        _sync_gen_value_weights(trainer, [engine], threading.Lock())

    engine.wake_up.remote.assert_called_once_with()


def test_weight_sync_health_checks_while_waiting_for_scoring(monkeypatch):
    trainer = MagicMock()
    engine = MagicMock()
    engines_lock = threading.Lock()
    engines_lock.acquire()
    health_checks = 0

    def health_check():
        nonlocal health_checks
        health_checks += 1
        engines_lock.release()

    wait_calls = 0

    def fake_wait(*_args, **_kwargs):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            return [{"engine_refs": [], "version": 3}], [0.0]
        return [None], [0.0]

    monkeypatch.setattr("open_instruct.grpo_fast_genvalue._check_gen_value_engines", lambda _engines: None)
    monkeypatch.setattr("open_instruct.grpo_fast_genvalue.utils.ray_get_with_progress", fake_wait)

    metrics = _sync_gen_value_weights(trainer, [engine], engines_lock, health_check_fn=health_check)

    assert health_checks == 1
    assert metrics["gen_value/synced_version"] == 3
    assert not engines_lock.locked()


# ── _build_sample_scoring_prompts (pure-Python, no GPU) ───────────────────────


def test_build_sample_scoring_prompts_length():
    # Build a tiny fake dataset
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "What is 2+2?"
    dataset = [
        {INPUT_IDS_PROMPT_KEY: [1, 2, 3], "ground_truth": "4"},
        {INPUT_IDS_PROMPT_KEY: [4, 5, 6], "ground_truth": "5"},
        {INPUT_IDS_PROMPT_KEY: [7, 8, 9], "ground_truth": "6"},
    ]

    kwargs = _base_kwargs()
    cfg = GenValueExperimentConfig(**kwargs)

    prompts = _build_sample_scoring_prompts(cfg, tokenizer, dataset, n=2, ground_truths_key="ground_truth")
    assert len(prompts) == 2
    for p in prompts:
        assert isinstance(p, str)
        assert len(p) > 0
