import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.data.synthesize_gen_value_sft import (
    collect,
    extract_response_text,
    make_batch_request,
    prepare,
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
        self.assertFalse(request["body"]["stream"])

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


if __name__ == "__main__":
    unittest.main()
