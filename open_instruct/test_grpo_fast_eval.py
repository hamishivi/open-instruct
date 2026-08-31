import math
import pathlib
import tempfile
import unittest
from queue import Empty
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from datasets import Dataset

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_utils
from open_instruct.data_types import EnvConfig
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
)
from open_instruct.environments.tools.utils import EnvsConfig
from open_instruct.grpo_fast import (
    CHECKPOINT_COMPLETE_MARKER,
    PolicyTrainerRayProcess,
    _build_data_prep_actor_resume_state,
    _is_in_warmup_window,
    create_generation_configs,
    maybe_evaluate,
    maybe_save_checkpoint,
    setup_runtime_variables,
)


class _QueueWithSize:
    def __init__(self, size: int):
        self._size = size

    def qsize(self) -> int:
        return self._size


class TestCreateGenerationConfigs(unittest.TestCase):
    def test_eval_response_length_defaults_to_response_length(self):
        streaming_config = data_loader_lib.StreamingDataLoaderConfig(
            max_prompt_token_length=128, response_length=128, pack_length=512
        )
        self.assertEqual(streaming_config.eval_response_length, streaming_config.response_length)

    def test_eval_uses_pass_at_k_and_eval_response_length(self):
        args = grpo_utils.GRPOExperimentConfig(eval_pass_at_k=8)
        streaming_config = data_loader_lib.StreamingDataLoaderConfig(response_length=256, eval_response_length=512)
        vllm_config = data_loader_lib.VLLMConfig()

        configs = create_generation_configs(args, streaming_config, vllm_config)

        self.assertEqual(configs["train"].n, streaming_config.num_samples_per_prompt_rollout)
        self.assertEqual(configs["train"].max_tokens, 256)
        self.assertEqual(configs["eval"].n, 8)
        self.assertEqual(configs["eval"].max_tokens, 512)

    def test_vllm_max_model_len_uses_longest_response_length(self):
        streaming_config = data_loader_lib.StreamingDataLoaderConfig(
            max_prompt_token_length=1024, response_length=256, eval_response_length=512, pack_length=1536
        )
        max_model_len = streaming_config.max_prompt_token_length + max(
            streaming_config.response_length, streaming_config.eval_response_length
        )
        self.assertEqual(max_model_len, 1536)


class TestWarmupWindows(unittest.TestCase):
    def test_value_warmup_freezes_policy_for_generative_value_model(self):
        args = SimpleNamespace(
            value_warmup_steps=100,
            policy_warmup_steps=0,
            value_rewarmup_start=0,
            value_rewarmup_steps=0,
            use_generative_value_model=True,
            gen_value_vllm_num_engines=5,
        )

        self.assertTrue(_is_in_warmup_window(args, 100))
        self.assertFalse(_is_in_warmup_window(args, 101))

    def test_value_rewarmup_freezes_policy_for_generative_value_model(self):
        args = SimpleNamespace(
            value_warmup_steps=0,
            policy_warmup_steps=0,
            value_rewarmup_start=50,
            value_rewarmup_steps=10,
            use_generative_value_model=True,
            gen_value_vllm_num_engines=5,
        )

        self.assertTrue(_is_in_warmup_window(args, 55))
        self.assertFalse(_is_in_warmup_window(args, 61))


class TestDataPreparationResumeState(unittest.TestCase):
    def test_resume_uses_last_consumed_step_without_mutating_checkpoint(self):
        checkpoint_state = {
            "training_step": 50,
            "data_prep_actor_state": {
                "training_step": 0,
                "last_consumed_step": 49,
                "iter_dataloader_state": {"batches_processed": 1696},
            },
        }

        resume_state = _build_data_prep_actor_resume_state(checkpoint_state)

        self.assertEqual(resume_state["last_consumed_step"], 49)
        self.assertEqual(resume_state["training_step"], 50)
        self.assertEqual(resume_state["iter_dataloader_state"], {"batches_processed": 1696})
        self.assertEqual(checkpoint_state["data_prep_actor_state"]["training_step"], 0)


class TestGenerativeValueBoundaryState(unittest.TestCase):
    @staticmethod
    def _trainer():
        trainer_class = PolicyTrainerRayProcess.__ray_metadata__.modified_class
        trainer = object.__new__(trainer_class)
        trainer.args = SimpleNamespace(
            gen_value_segmentation="fixed",
            gen_value_chunk_size=512,
            gen_value_max_segments=16,
            sae_threshold=0.2,
            gen_value_score_min=0.0,
            gen_value_score_max=10.0,
            gen_value_conditioning="none",
            gen_value_use_icc=False,
            value_reward_min=0.0,
            value_reward_max=1.0,
        )
        trainer.streaming_config = SimpleNamespace(response_length=8192)
        trainer.tokenizer = Mock()
        trainer.tokenizer.decode.side_effect = lambda token_ids, **_: " ".join(map(str, token_ids))
        return trainer

    def test_final_action_probe_baselines_only_the_final_action(self):
        trainer = self._trainer()

        query_responses = torch.tensor([[10, 11, 20, 21, 22, 23]])
        position_ids = torch.arange(6).unsqueeze(0)
        response_mask = torch.tensor([[False, False, True, True, True, True]])
        request = trainer._build_gen_value_scoring_request(
            query_responses, position_ids, response_mask, ground_truths_pack=["42"]
        )

        self.assertEqual(request["prompt_state_kinds"], ["segment_start", "final_action"])
        self.assertEqual(request["prompt_response_tokens_used"], [0, 3])
        self.assertEqual(request["prompt_trajectory_fractions"], [0.0, 1.0])
        self.assertIn("Partial response:\n<rollout>20 21 22</rollout>", request["prompts"][1])

        request_outputs = [
            SimpleNamespace(outputs=[SimpleNamespace(text="<answer>5</answer>")]),
            SimpleNamespace(outputs=[SimpleNamespace(text="<answer>0</answer>")]),
        ]
        values, training_pairs = trainer._finish_gen_value_scoring_request(request, request_outputs)

        torch.testing.assert_close(values, torch.tensor([[0.0, 0.5, 0.5, 0.5, 0.0]]))
        self.assertEqual([pair["state_kind"] for pair in training_pairs], ["segment_start", "final_action"])
        self.assertEqual([pair["trajectory_fraction"] for pair in training_pairs], [0.0, 1.0])

    def test_packed_final_action_probes_do_not_cross_subsequences(self):
        trainer = self._trainer()
        query_responses = torch.tensor([[10, 11, 20, 21, 30, 31, 40, 41, 42]])
        position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4]])
        response_mask = torch.tensor([[False, False, True, True, False, False, True, True, True]])
        request = trainer._build_gen_value_scoring_request(
            query_responses, position_ids, response_mask, ground_truths_pack=["2", "3"]
        )

        self.assertEqual(
            request["prompt_state_kinds"], ["segment_start", "final_action", "segment_start", "final_action"]
        )
        self.assertEqual(request["prompt_subseq_idx"], [0, 0, 1, 1])
        self.assertEqual(request["prompt_response_tokens_used"], [0, 1, 0, 2])
        self.assertEqual(request["prompt_trajectory_fractions"], [0.0, 1.0, 0.0, 1.0])

        request_outputs = [
            SimpleNamespace(outputs=[SimpleNamespace(text=f"<answer>{score}</answer>")]) for score in (5, 1, 6, 2)
        ]
        values, training_pairs = trainer._finish_gen_value_scoring_request(request, request_outputs)

        torch.testing.assert_close(values, torch.tensor([[0.0, 0.5, 0.1, 0.0, 0.0, 0.6, 0.6, 0.2]]))
        self.assertEqual([pair["subseq_idx"] for pair in training_pairs], [0, 0, 1, 1])


class TestValueRewardRangeSetup(unittest.TestCase):
    def test_requires_explicit_bounds_for_summed_rewards(self):
        args = grpo_utils.GRPOExperimentConfig(use_value_model=True)
        streaming_config = data_loader_lib.StreamingDataLoaderConfig(reward_aggregator="sum")

        with self.assertRaisesRegex(ValueError, "summed multi-turn rewards"):
            setup_runtime_variables(args, streaming_config, EnvsConfig())

    def test_requires_explicit_bounds_for_tool_rewards(self):
        args = grpo_utils.GRPOExperimentConfig(use_value_model=True)
        streaming_config = data_loader_lib.StreamingDataLoaderConfig()

        with self.assertRaisesRegex(ValueError, "tool/environment rewards"):
            setup_runtime_variables(args, streaming_config, EnvsConfig(tools=["python"]))

    @patch("open_instruct.grpo_fast.maybe_use_ai2_hf_entity", return_value=None)
    def test_accepts_explicit_bounds_for_tool_rewards(self, _maybe_use_ai2_hf_entity):
        args = grpo_utils.GRPOExperimentConfig(use_value_model=True, value_reward_min=-5.0, value_reward_max=5.0)
        streaming_config = data_loader_lib.StreamingDataLoaderConfig()

        resolved = setup_runtime_variables(args, streaming_config, EnvsConfig(tools=["python"]))

        self.assertEqual(resolved.value_reward_min, -5.0)
        self.assertEqual(resolved.value_reward_max, 5.0)


class TestValueCheckpointState(unittest.TestCase):
    def test_value_engine_checkpoint_is_tagged_with_policy_step(self):
        trainer_cls = PolicyTrainerRayProcess.__ray_metadata__.modified_class
        trainer = object.__new__(trainer_cls)
        value_mpu = object()
        trainer.value_model = SimpleNamespace(mpu=value_mpu, save_checkpoint=Mock(return_value=True))
        trainer._save_value_model = Mock()

        trainer._save_value_checkpoint_state("/tmp/value-checkpoint", training_step=17)

        trainer._save_value_model.assert_called_once_with("/tmp/value-checkpoint")
        trainer.value_model.save_checkpoint.assert_called_once_with(
            "/tmp/value-checkpoint/deepspeed", tag="global_step17", client_state={"training_step": 17}
        )
        self.assertIs(trainer.value_model.mpu, value_mpu)


class TestModelCompletionMarker(unittest.TestCase):
    @patch("open_instruct.grpo_fast.ray_get_with_progress")
    def test_periodic_marker_is_published_after_external_model(self, wait_for_policy):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SimpleNamespace(
                save_freq=1,
                eval_on_step_0=False,
                output_dir=str(pathlib.Path(tmp_dir) / "model"),
                world_size=1,
                try_launch_beaker_eval_jobs_on_weka=False,
            )
            policy_model = SimpleNamespace(save_model=SimpleNamespace(remote=Mock(return_value="policy-save")))
            policy_group = SimpleNamespace(models=[policy_model])
            step_dir = pathlib.Path(f"{args.output_dir}_checkpoints") / "step_2"
            events = []

            def finish_policy(*_args, **_kwargs):
                step_dir.mkdir(parents=True)
                events.append("policy")

            def save_critic(output_dir, training_step):
                self.assertEqual(pathlib.Path(output_dir), step_dir)
                self.assertEqual(training_step, 2)
                self.assertFalse((step_dir / CHECKPOINT_COMPLETE_MARKER).exists())
                events.append("critic")

            wait_for_policy.side_effect = finish_policy
            maybe_save_checkpoint(args, 2, policy_group, "chat", Mock(), "wandb", save_critic)

            self.assertEqual(events, ["policy", "critic"])
            self.assertTrue((step_dir / CHECKPOINT_COMPLETE_MARKER).exists())

    @patch("open_instruct.grpo_fast.ray_get_with_progress")
    def test_periodic_marker_is_not_published_when_external_model_fails(self, wait_for_policy):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SimpleNamespace(
                save_freq=1,
                eval_on_step_0=False,
                output_dir=str(pathlib.Path(tmp_dir) / "model"),
                world_size=1,
                try_launch_beaker_eval_jobs_on_weka=False,
            )
            policy_model = SimpleNamespace(save_model=SimpleNamespace(remote=Mock(return_value="policy-save")))
            policy_group = SimpleNamespace(models=[policy_model])
            step_dir = pathlib.Path(f"{args.output_dir}_checkpoints") / "step_2"

            def finish_policy(*_args, **_kwargs):
                step_dir.mkdir(parents=True, exist_ok=True)

            def fail_critic(*_args, **_kwargs):
                raise RuntimeError("critic save failed")

            wait_for_policy.side_effect = finish_policy
            undecorated_save = maybe_save_checkpoint.__wrapped__
            with self.assertRaisesRegex(RuntimeError, "critic save failed"):
                undecorated_save(args, 2, policy_group, "chat", Mock(), "wandb", fail_critic)

            self.assertFalse((step_dir / CHECKPOINT_COMPLETE_MARKER).exists())


class TestValueTokenAlignment(unittest.TestCase):
    def test_rejects_non_sequence_parallel_shape_mismatch(self):
        trainer_cls = PolicyTrainerRayProcess.__ray_metadata__.modified_class
        trainer = object.__new__(trainer_cls)
        trainer._sp_world_size = 1

        with self.assertRaisesRegex(RuntimeError, "only a masked sequence-parallel padding suffix"):
            trainer._align_value_predictions(
                torch.tensor([[1.0]]), torch.tensor([[True, False]]), "test value forward"
            )

    def test_allows_only_masked_sequence_parallel_padding_suffix(self):
        trainer_cls = PolicyTrainerRayProcess.__ray_metadata__.modified_class
        trainer = object.__new__(trainer_cls)
        trainer._sp_world_size = 2

        aligned = trainer._align_value_predictions(
            torch.tensor([[1.0]]), torch.tensor([[True, False, False]]), "test value forward"
        )

        torch.testing.assert_close(aligned, torch.tensor([[1.0, 0.0, 0.0]]))

    def test_rejects_sequence_parallel_padding_over_real_tokens(self):
        trainer_cls = PolicyTrainerRayProcess.__ray_metadata__.modified_class
        trainer = object.__new__(trainer_cls)
        trainer._sp_world_size = 2

        with self.assertRaisesRegex(RuntimeError, "only a masked sequence-parallel padding suffix"):
            trainer._align_value_predictions(
                torch.tensor([[1.0]]), torch.tensor([[True, True, False]]), "test value forward"
            )

    def test_dummy_value_forward_uses_one_attended_token(self):
        trainer_cls = PolicyTrainerRayProcess.__ray_metadata__.modified_class
        trainer = object.__new__(trainer_cls)
        trainer.tokenizer = SimpleNamespace(pad_token_id=7)
        trainer.value_model = Mock(return_value=SimpleNamespace(logits=torch.ones(1, 1, 1, requires_grad=True)))
        dummy_outputs = []

        trainer._dummy_value_forward(torch.long, torch.device("cpu"), dummy_outputs)

        kwargs = trainer.value_model.call_args.kwargs
        torch.testing.assert_close(kwargs["input_ids"], torch.tensor([[7]]))
        torch.testing.assert_close(kwargs["attention_mask"], torch.ones(1, 1, dtype=torch.long))
        torch.testing.assert_close(kwargs["position_ids"], torch.zeros(1, 1, dtype=torch.long))
        self.assertEqual(len(dummy_outputs), 1)


class TestConditionedClassificationValueForward(unittest.TestCase):
    def test_preserves_two_logits_and_converts_to_scalar_values(self):
        trainer_cls = PolicyTrainerRayProcess.__ray_metadata__.modified_class
        trainer = object.__new__(trainer_cls)
        trainer.args = SimpleNamespace(
            value_loss="classification",
            bound_value_predictions=False,
            value_reward_min=0.0,
            value_reward_max=1.0,
            sequence_parallel_size=1,
        )
        trainer._sp_world_size = 1

        query_responses = torch.tensor([[10, 11, 12, 13]])
        position_ids = torch.arange(4).unsqueeze(0)
        response_mask = torch.tensor([[False, True, True, False]])
        subseq = {"offset_in_pack": 0, "response_is_resp": response_mask[0]}
        trainer._unpack_subseqs = Mock(return_value=[subseq])
        trainer._build_conditioned_value_entries = Mock(
            return_value=[{"input_ids": query_responses[0], "orig_mask": response_mask[0], "subseq": subseq}]
        )
        model_logits = torch.tensor([[[0.0, 1.0], [1.0, 3.0], [4.0, 1.0], [2.0, 2.0]]], requires_grad=True)
        trainer.value_model = Mock(return_value=SimpleNamespace(logits=model_logits))

        logits = trainer._forward_value_with_conditioning(
            query_responses, position_ids, response_mask, ["42"], None, return_logits=True
        )
        values = trainer._forward_value_with_conditioning(query_responses, position_ids, response_mask, ["42"], None)

        self.assertEqual(logits.shape, (1, 3, 2))
        torch.testing.assert_close(logits[0, :2], model_logits[0, :2])
        torch.testing.assert_close(values[0, :2], model_logits[0, :2].softmax(dim=-1)[:, 1])
        self.assertTrue(logits.requires_grad)


class TestMaybeEvaluate(unittest.TestCase):
    def _build_eval_dataset(self, num_prompts: int) -> Dataset:
        return Dataset.from_dict(
            {
                INPUT_IDS_PROMPT_KEY: [[1, 2, 3] for _ in range(num_prompts)],
                GROUND_TRUTHS_KEY: ["42" for _ in range(num_prompts)],
                VERIFIER_SOURCE_KEY: ["unit_test" for _ in range(num_prompts)],
                RAW_PROMPT_KEY: ["prompt" for _ in range(num_prompts)],
                "index": list(range(num_prompts)),
            }
        )

    def test_non_final_step_defers_when_eval_results_incomplete(self):
        args = SimpleNamespace(num_training_steps=10, with_tracking=False)
        eval_dataset = self._build_eval_dataset(num_prompts=3)
        eval_queue = _QueueWithSize(size=2)
        eval_generation_config = SimpleNamespace(n=32)

        with patch("open_instruct.grpo_fast.accumulate_inference_batches") as mock_accumulate:
            maybe_evaluate(
                args=args,
                training_step=5,
                evaluation_inference_results_Q=eval_queue,
                tokenizer=Mock(),
                episode=0,
                eval_dataset=eval_dataset,
                eval_generation_config=eval_generation_config,
                model_dims=Mock(),
                base_env_config=EnvConfig(),
                max_possible_score=1.0,
            )

        mock_accumulate.assert_not_called()

    def test_final_step_calls_accumulate_even_when_queue_is_incomplete(self):
        args = SimpleNamespace(num_training_steps=10, with_tracking=False)
        eval_dataset = self._build_eval_dataset(num_prompts=3)
        eval_queue = _QueueWithSize(size=0)
        eval_generation_config = SimpleNamespace(n=32)

        with patch("open_instruct.grpo_fast.accumulate_inference_batches", side_effect=Empty) as mock_accumulate:
            maybe_evaluate(
                args=args,
                training_step=10,
                evaluation_inference_results_Q=eval_queue,
                tokenizer=Mock(),
                episode=0,
                eval_dataset=eval_dataset,
                eval_generation_config=eval_generation_config,
                model_dims=Mock(),
                base_env_config=EnvConfig(),
                max_possible_score=1.0,
            )

        mock_accumulate.assert_called_once()

    def test_records_eval_model_step_summary(self):
        args = SimpleNamespace(num_training_steps=200, with_tracking=False)
        eval_dataset = self._build_eval_dataset(num_prompts=1)
        eval_queue = _QueueWithSize(size=1)
        eval_generation_config = SimpleNamespace(n=2)
        tokenizer = Mock()
        tokenizer.batch_decode.return_value = ["prompt", "prompt"]
        tokenizer.pad_token = "<pad>"

        eval_result = SimpleNamespace(
            responses=[[1], [2]],
            finish_reasons=["stop", "stop"],
            token_statistics=SimpleNamespace(num_prompt_tokens=10, num_response_tokens=4, generation_time=2.0),
        )
        eval_batch = SimpleNamespace(
            scores=[1.0, 0.0],
            queries=[[1, 2, 3], [1, 2, 3]],
            decoded_responses=["resp_a", "resp_b"],
            ground_truths=["42", "42"],
            active_tools=None,
        )
        reward_metrics = {"model_step_min": 102.0, "model_step_max": 104.0, "model_step_mean": 103.0}

        with (
            patch(
                "open_instruct.grpo_fast.accumulate_inference_batches",
                return_value=(eval_result, eval_batch, reward_metrics, None),
            ),
            patch("open_instruct.grpo_fast.print_rich_single_line_metrics") as mock_print_metrics,
            patch("open_instruct.grpo_fast.print_rich_table"),
        ):
            maybe_evaluate(
                args=args,
                training_step=100,
                evaluation_inference_results_Q=eval_queue,
                tokenizer=tokenizer,
                episode=0,
                eval_dataset=eval_dataset,
                eval_generation_config=eval_generation_config,
                model_dims=Mock(),
                base_env_config=EnvConfig(),
                max_possible_score=1.0,
            )

        logged = mock_print_metrics.call_args.args[0]
        self.assertEqual(logged["eval/model_step_min"], 102.0)
        self.assertEqual(logged["eval/model_step_max"], 104.0)
        self.assertEqual(logged["eval/model_step_mean"], 103.0)

    def test_records_pass_at_k_metrics(self):
        args = SimpleNamespace(num_training_steps=200, with_tracking=False)
        eval_dataset = self._build_eval_dataset(num_prompts=2)
        eval_queue = _QueueWithSize(size=2)
        eval_generation_config = SimpleNamespace(n=2)
        tokenizer = Mock()
        tokenizer.batch_decode.return_value = ["prompt"] * 4
        tokenizer.pad_token = "<pad>"

        eval_result = SimpleNamespace(
            responses=[[1], [2], [3], [4]],
            finish_reasons=["stop", "stop", "stop", "stop"],
            token_statistics=SimpleNamespace(num_prompt_tokens=10, num_response_tokens=4, generation_time=2.0),
        )
        eval_batch = SimpleNamespace(
            scores=[1.0, 0.0, 0.0, 1.0],
            queries=[[1, 2, 3]] * 4,
            decoded_responses=["resp_a", "resp_b", "resp_c", "resp_d"],
            ground_truths=["42"] * 4,
            active_tools=None,
        )

        with (
            patch(
                "open_instruct.grpo_fast.accumulate_inference_batches",
                return_value=(eval_result, eval_batch, {}, None),
            ),
            patch("open_instruct.grpo_fast.print_rich_single_line_metrics") as mock_print_metrics,
            patch("open_instruct.grpo_fast.print_rich_table"),
        ):
            maybe_evaluate(
                args=args,
                training_step=100,
                evaluation_inference_results_Q=eval_queue,
                tokenizer=tokenizer,
                episode=0,
                eval_dataset=eval_dataset,
                eval_generation_config=eval_generation_config,
                model_dims=Mock(),
                base_env_config=EnvConfig(),
                max_possible_score=1.0,
            )

        logged = mock_print_metrics.call_args.args[0]
        self.assertEqual(logged["eval/pass_at_1"], 0.5)
        self.assertEqual(logged["eval/pass_at_2"], 1.0)
        self.assertEqual(logged["eval/pass_at_1_unbiased"], 0.5)
        self.assertEqual(logged["eval/pass_at_2_unbiased"], 1.0)


class TestComputePassAtKMetrics(unittest.TestCase):
    def test_formula_matches_one_minus_comb_ratio_single_prompt(self):
        n, c, k = 8, 3, 4
        wrong = n - c
        expected = 1.0 - math.comb(wrong, k) / math.comb(n, k)
        correct = np.zeros((1, n), dtype=bool)
        correct[0, :c] = True
        m = grpo_utils.compute_pass_at_k_metrics(correct)
        self.assertAlmostEqual(m["eval/pass_at_4_unbiased"], expected)
        self.assertAlmostEqual(m["eval/pass_at_1"], c / n)

    def test_two_prompts_n2_matches_maybe_evaluate_mock(self):
        correct = np.array([[True, False], [False, True]])
        m = grpo_utils.compute_pass_at_k_metrics(correct)
        self.assertAlmostEqual(m["eval/pass_at_1"], 0.5)
        self.assertAlmostEqual(m["eval/pass_at_2"], 1.0)
        self.assertAlmostEqual(m["eval/pass_at_1_unbiased"], 0.5)
        self.assertAlmostEqual(m["eval/pass_at_2_unbiased"], 1.0)

    def test_all_correct(self):
        m = grpo_utils.compute_pass_at_k_metrics(np.ones((1, 4), dtype=bool))
        self.assertEqual(m["eval/pass_at_1"], 1.0)
        self.assertEqual(m["eval/pass_at_2_unbiased"], 1.0)

    def test_all_wrong_when_k_fits(self):
        m = grpo_utils.compute_pass_at_k_metrics(np.zeros((1, 4), dtype=bool))
        self.assertEqual(m["eval/pass_at_1"], 0.0)
        self.assertEqual(m["eval/pass_at_2_unbiased"], 0.0)

    def test_fewer_than_k_wrong_returns_one(self):
        """Any k-subset must include a correct completion (here k=2, only one wrong)."""
        m = grpo_utils.compute_pass_at_k_metrics(np.array([[True, True, True, False]]))
        self.assertEqual(m["eval/pass_at_1"], 0.75)
        self.assertEqual(m["eval/pass_at_2_unbiased"], 1.0)


if __name__ == "__main__":
    unittest.main()
