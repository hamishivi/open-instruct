import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from scripts.data.synthesize_gen_value_sft import (
    audit,
    collect,
    extract_response_text,
    make_batch_request,
    prepare,
    select_teacher_consensus,
    select_teacher_states,
)


class TestGenValueSFTSynthesis(unittest.TestCase):
    @staticmethod
    def _state(outcome: float, position: str) -> dict:
        metadata = {
            "early": {"state_kind": "segment_start", "trajectory_fraction": 0.25},
            "middle": {"state_kind": "segment_start", "trajectory_fraction": 0.5},
            "late": {"state_kind": "segment_start", "trajectory_fraction": 0.75},
            "final_action": {"state_kind": "final_action", "trajectory_fraction": 1.0},
        }[position]
        return {
            "source_critic_version": 25,
            "outcome": outcome,
            # Source output quality must not affect teacher-state selection.
            "prediction": None,
            "squared_error": None,
            "prompt": f"prompt-{outcome}-{position}",
            "generation": "source parse failure",
            **metadata,
        }

    def test_teacher_state_selection_ignores_source_output_and_balances_positions(self):
        examples = [
            self._state(outcome, position)
            for outcome in (0.0, 1.0)
            for position in ("early", "middle", "late", "final_action")
        ]

        selected = select_teacher_states(examples, min_critic_version=25, max_examples_per_outcome=4, seed=1)

        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(example["outcome"] == 0.0 for example in selected), 4)
        self.assertEqual(sum(example["outcome"] == 1.0 for example in selected), 4)

    def test_teacher_state_selection_accepts_fixed_validation_snapshot(self):
        snapshot = {
            "target": 0.0,
            "kind": "final_action",
            "version": 25,
            "trajectory_fraction": 1.0,
            "prediction": 0.9,
            "prompt": "held-out critic prompt",
            "generation": "source critic output",
        }

        selected = select_teacher_states([snapshot], min_critic_version=25, max_examples_per_outcome=1, seed=0)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["outcome"], 0.0)
        self.assertEqual(selected[0]["state_kind"], "final_action")
        self.assertEqual(selected[0]["source_critic_version"], 25)

    def test_batch_request_uses_paper_teacher_and_responses_endpoint(self):
        request = make_batch_request(
            "trace-1", "critic prompt", model="gpt-5", reasoning_effort="medium", max_output_tokens=1024
        )

        self.assertEqual(request["url"], "/v1/responses")
        self.assertEqual(request["body"]["model"], "gpt-5")
        self.assertEqual(request["body"]["input"], "critic prompt")
        self.assertFalse(request["body"]["store"])
        self.assertIn("<answer>N</answer>", request["body"]["instructions"])

    def test_batch_request_supports_local_chat_completions(self):
        request = make_batch_request(
            "trace-1",
            "critic prompt",
            model="Qwen/Qwen3-8B",
            reasoning_effort="medium",
            max_output_tokens=1024,
            request_format="chat_completions",
        )

        self.assertEqual(request["url"], "/v1/chat/completions")
        self.assertEqual(request["body"]["model"], "Qwen/Qwen3-8B")
        self.assertEqual(request["body"]["messages"][-1], {"role": "user", "content": "critic prompt"})
        self.assertIn("<answer>N</answer>", request["body"]["messages"][0]["content"])
        self.assertEqual(request["body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertFalse(request["body"]["stream"])

    def test_batch_request_can_enable_local_thinking(self):
        request = make_batch_request(
            "trace-1",
            "critic prompt",
            model="Qwen/Qwen3-32B",
            reasoning_effort="medium",
            max_output_tokens=4096,
            request_format="chat_completions",
            enable_thinking=True,
        )

        self.assertEqual(request["body"]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(request["body"]["max_tokens"], 4096)

    def test_batch_request_accepts_custom_teacher_instructions(self):
        request = make_batch_request(
            "trace-1",
            "critic prompt",
            model="Qwen/Qwen3-32B",
            reasoning_effort="medium",
            max_output_tokens=1024,
            request_format="chat_completions",
            teacher_instructions="terminal calibration <answer>0</answer>",
        )

        self.assertEqual(request["body"]["messages"][0]["content"], "terminal calibration <answer>0</answer>")

    def test_extract_response_text_supports_local_reasoning_content(self):
        text = extract_response_text(
            {
                "custom_id": "trace-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "reasoning_content": "The algebra contains a decisive error. ",
                                    "content": " <answer>1</answer>",
                                }
                            }
                        ]
                    },
                },
                "error": None,
            }
        )

        self.assertEqual(text, "The algebra contains a decisive error.\n<answer>1</answer>")

    def test_extract_response_text_rejects_missing_output(self):
        with self.assertRaisesRegex(ValueError, "contains no output_text"):
            extract_response_text(
                {"custom_id": "trace-1", "response": {"status_code": 200, "body": {"output": []}}, "error": None}
            )

    def test_prepare_and_collect_round_trip(self):
        examples = [self._state(outcome, "final_action") for outcome in (0.0, 1.0)]
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            reservoir = root / "reservoir.jsonl"
            reservoir.write_text("".join(json.dumps(example) + "\n" for example in examples))
            batch_path = root / "batch.jsonl"
            metadata_path = root / "metadata.jsonl"
            prepare(
                argparse.Namespace(
                    inputs=[reservoir],
                    batch_output=batch_path,
                    metadata_output=metadata_path,
                    model="gpt-5",
                    reasoning_effort="medium",
                    max_output_tokens=1024,
                    min_critic_version=25,
                    max_examples_per_outcome=1,
                    seed=0,
                    allow_ground_truth_conditioning=False,
                )
            )

            metadata = [json.loads(line) for line in metadata_path.read_text().splitlines()]
            result_path = root / "results.jsonl"
            results = []
            for row in metadata:
                outcome = float(row["source"]["outcome"])
                answer = 9 if outcome > 0.5 else 1
                results.append(
                    {
                        "custom_id": row["custom_id"],
                        "response": {
                            "status_code": 200,
                            "body": {
                                "output": [
                                    {
                                        "type": "message",
                                        "content": [
                                            {
                                                "type": "output_text",
                                                "text": f"Value reasoning. <answer>{answer}</answer>",
                                            }
                                        ],
                                    }
                                ]
                            },
                        },
                        "error": None,
                    }
                )
            result_path.write_text("".join(json.dumps(result) + "\n" for result in results))
            sft_path = root / "sft.jsonl"
            collect(
                argparse.Namespace(
                    metadata=metadata_path,
                    results=[result_path],
                    output=sft_path,
                    teacher_model="gpt-5",
                    max_teacher_squared_error=None,
                )
            )

            sft_rows = [json.loads(line) for line in sft_path.read_text().splitlines()]
            self.assertEqual(len(sft_rows), 2)
            self.assertEqual({row["teacher_prediction"] for row in sft_rows}, {0.1, 0.9})
            self.assertTrue(all(row["teacher_model"] == "gpt-5" for row in sft_rows))

    def test_prepare_conditions_only_terminal_teacher_request_on_outcome(self):
        examples = [
            self._state(0.0, "early"),
            self._state(0.0, "final_action"),
            self._state(1.0, "early"),
            self._state(1.0, "final_action"),
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            reservoir = root / "reservoir.jsonl"
            reservoir.write_text("".join(json.dumps(example) + "\n" for example in examples))
            batch_path = root / "batch.jsonl"
            metadata_path = root / "metadata.jsonl"
            prepare(
                argparse.Namespace(
                    inputs=[reservoir],
                    batch_output=batch_path,
                    metadata_output=metadata_path,
                    model="Qwen/Qwen3-32B",
                    request_format="chat_completions",
                    reasoning_effort="medium",
                    max_output_tokens=1024,
                    min_critic_version=25,
                    max_examples_per_outcome=2,
                    seed=0,
                    allow_ground_truth_conditioning=False,
                    condition_terminal_teacher_on_outcome=True,
                )
            )

            metadata = [json.loads(line) for line in metadata_path.read_text().splitlines()]
            batches = [json.loads(line) for line in batch_path.read_text().splitlines()]
            for source_row, batch_row in zip(metadata, batches, strict=True):
                source = source_row["source"]
                instructions = batch_row["body"]["messages"][0]["content"]
                self.assertEqual(batch_row["body"]["messages"][1]["content"], source["prompt"])
                if source["state_kind"] == "final_action":
                    expected_score = 10 if source["outcome"] > 0.5 else 0
                    self.assertIn(f"<answer>{expected_score}</answer>", instructions)
                    self.assertIn("outcome verifier", instructions)
                else:
                    self.assertNotIn("outcome verifier", instructions)
                    self.assertIn("Do not assume knowledge", instructions)

    def test_collect_can_skip_an_invalid_score_explicitly(self):
        good = self._state(1.0, "final_action")
        bad = self._state(0.0, "final_action")
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            metadata_path = root / "metadata.jsonl"
            result_path = root / "results.jsonl"
            output_path = root / "sft.jsonl"
            metadata_path.write_text(
                json.dumps({"custom_id": "good", "source": good})
                + "\n"
                + json.dumps({"custom_id": "bad", "source": bad})
                + "\n"
            )
            result_path.write_text(
                json.dumps(
                    {
                        "custom_id": "good",
                        "response": {
                            "status_code": 200,
                            "body": {"choices": [{"message": {"content": "Sound. <answer>9</answer>"}}]},
                        },
                        "error": None,
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "custom_id": "bad",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {
                                        "message": {"content": "Reasoning exhausted its budget."},
                                        "finish_reason": "length",
                                    }
                                ]
                            },
                        },
                        "error": None,
                    }
                )
                + "\n"
            )

            collect(
                argparse.Namespace(
                    metadata=metadata_path,
                    results=[result_path],
                    output=output_path,
                    teacher_model="local-thinking",
                    max_teacher_squared_error=None,
                    skip_invalid_scores=True,
                )
            )

            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prompt"], good["prompt"])

    def test_consensus_never_treats_sampled_final_action_as_terminal(self):
        primary = []
        judge_one = []
        judge_two = []
        for outcome in (0.0, 1.0):
            for position in ("early", "final_action"):
                row = self._state(outcome, position)
                # An incorrect sampled trajectory can still begin in a valuable
                # state.  Independent teachers agree it has high continuation
                # value, so this must survive at an early prefix.
                prediction = 0.8 if outcome == 0.0 else 0.9
                primary.append({**row, "generation": "primary reasoning", "teacher_prediction": prediction})
                judge_one.append({**row, "generation": "judge one", "teacher_prediction": prediction - 0.1})
                judge_two.append({**row, "generation": "judge two", "teacher_prediction": prediction})

        selected, stats = select_teacher_consensus(
            primary, [judge_one, judge_two], max_teacher_range=0.2, max_examples_per_outcome=2, seed=0
        )

        selected_keys = {
            (example["outcome"], example["state_kind"], example["trajectory_fraction"]) for example in selected
        }
        self.assertIn((0.0, "segment_start", 0.25), selected_keys)
        self.assertIn((0.0, "final_action", 1.0), selected_keys)
        self.assertEqual(stats["consensus_candidates"], 4)

    def test_consensus_requires_all_teachers_and_rejects_score_disagreement(self):
        agreed = self._state(1.0, "final_action")
        disagreed = self._state(0.0, "final_action")
        primary = [
            {**agreed, "generation": "primary", "teacher_prediction": 0.9},
            {**disagreed, "generation": "primary", "teacher_prediction": 0.1},
        ]
        judge_one = [
            {**agreed, "generation": "judge", "teacher_prediction": 0.8},
            {**disagreed, "generation": "judge", "teacher_prediction": 0.8},
        ]
        judge_two = [{**agreed, "generation": "judge", "teacher_prediction": 1.0}]

        selected, stats = select_teacher_consensus(
            primary, [judge_one, judge_two], max_teacher_range=0.2, max_examples_per_outcome=1, seed=0
        )

        # The incorrect prompt is absent from judge two, while the correct prompt
        # passes at an exact floating-point range of 0.2.
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["outcome"], 1.0)
        self.assertEqual(stats["missing_judge"], 1)

    def test_prepare_rejects_answer_conditioned_prompt_by_default(self):
        example = self._state(0.0, "final_action")
        example["prompt"] = "critic header\n\nThe correct answer is 42. \nPartial response: wrong"
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_path = root / "snapshot.jsonl"
            input_path.write_text(json.dumps(example) + "\n")

            with self.assertRaisesRegex(ValueError, "answer-conditioned critic prompts"):
                prepare(
                    argparse.Namespace(
                        inputs=[input_path],
                        batch_output=root / "batch.jsonl",
                        metadata_output=root / "metadata.jsonl",
                        model="gpt-5",
                        reasoning_effort="medium",
                        max_output_tokens=1024,
                        min_critic_version=25,
                        max_examples_per_outcome=1,
                        seed=0,
                        allow_ground_truth_conditioning=False,
                    )
                )

    def test_prepare_allows_answer_phrase_inside_actor_rollout(self):
        example = self._state(0.0, "final_action")
        example["prompt"] = (
            "critic header\n\nProblem:\nA problem\n\nPartial response:\n<rollout>"
            "The correct answer is 7, but I guessed 8.</rollout>\nAnswer:"
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_path = root / "snapshot.jsonl"
            input_path.write_text(json.dumps(example) + "\n")

            prepare(
                argparse.Namespace(
                    inputs=[input_path],
                    batch_output=root / "batch.jsonl",
                    metadata_output=root / "metadata.jsonl",
                    model="gpt-5",
                    reasoning_effort="medium",
                    max_output_tokens=1024,
                    exclude_problem_dataset=None,
                    min_critic_version=25,
                    max_examples_per_outcome=1,
                    seed=0,
                    allow_ground_truth_conditioning=False,
                )
            )

            self.assertEqual(len((root / "batch.jsonl").read_text().splitlines()), 1)

    def test_prepare_excludes_heldout_mc_problems_before_balancing(self):
        kept = self._state(0.0, "final_action")
        kept["prompt"] = "Problem:\ntraining problem\n\nPartial response:\n<rollout>x</rollout>"
        excluded = self._state(1.0, "final_action")
        excluded["prompt"] = "Problem:\nheldout problem\n\nPartial response:\n<rollout>y</rollout>"
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_path = root / "snapshot.jsonl"
            input_path.write_text(json.dumps(kept) + "\n" + json.dumps(excluded) + "\n")
            holdout_path = root / "holdout.parquet"
            pd.DataFrame({"problem": ["heldout problem"]}).to_parquet(holdout_path, index=False)

            prepare(
                argparse.Namespace(
                    inputs=[input_path],
                    batch_output=root / "batch.jsonl",
                    metadata_output=root / "metadata.jsonl",
                    model="gpt-5",
                    reasoning_effort="medium",
                    max_output_tokens=1024,
                    min_critic_version=0,
                    max_examples_per_outcome=1,
                    seed=0,
                    allow_ground_truth_conditioning=False,
                    exclude_problem_dataset=holdout_path,
                )
            )

            metadata = [json.loads(line) for line in (root / "metadata.jsonl").read_text().splitlines()]
            self.assertEqual(len(metadata), 1)
            self.assertEqual(metadata[0]["source"]["prompt"], kept["prompt"])

    def test_audit_requires_parseable_unique_unconditioned_traces(self):
        example = self._state(0.0, "final_action")
        example["prompt"] = (
            "critic header\n\nProblem:\nA problem\n\nPartial response:\n<rollout>"
            "The correct answer is not obvious.</rollout>\nAnswer:"
        )
        example["generation"] = "The state is weak. <answer>1</answer>"
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_path = Path(temporary_dir) / "sft.jsonl"
            input_path.write_text(json.dumps(example) + "\n")

            audit(argparse.Namespace(inputs=[input_path], min_examples=1, allow_ground_truth_conditioning=False))


if __name__ == "__main__":
    unittest.main()
