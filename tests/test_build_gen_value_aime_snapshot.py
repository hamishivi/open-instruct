from scripts.eval.value_estimation import build_gen_value_aime_snapshot


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def test_aime_snapshot_covers_every_problem_and_balances_available_outcomes():
    table = {
        "columns": ["prompt", "response", "scores", "ground_truth", "active_tools"],
        "data": [
            ["problem-a", "a" * 40, 1.0, ["1"], "all"],
            ["problem-a", "b" * 40, 0.0, ["1"], "all"],
            ["problem-a", "c" * 40, 0.0, ["1"], "all"],
            ["problem-b", "d" * 40, 0.0, ["2"], "all"],
            ["problem-b", "e" * 40, 0.0, ["2"], "all"],
        ],
    }

    examples = build_gen_value_aime_snapshot.build_aime_validation_examples(
        table,
        FakeTokenizer(),
        actor_model_name="actor",
        actor_success_rate=0.25,
        response_token_limit=100,
        max_trajectories_per_problem=2,
        seed=7,
    )

    initial = [example for example in examples if example["kind"] == "initial"]
    sampled = [example for example in examples if example["target_source"] == "single_sample_return"]
    assert len(initial) == 2
    assert sorted(example["target"] for example in initial) == [0.0, 1 / 3]
    assert len({example["problem_id"] for example in examples}) == 2
    assert len(sampled) == 12
    assert sum(example["target"] == 1.0 for example in sampled) == 4
    assert sum(example["target"] == 0.0 for example in sampled) == 8
    assert {example["trajectory_fraction"] for example in sampled} == {0.25, 0.5, 0.75, 1.0}
    assert all("The active actor is actor." in example["prompt"] for example in examples)
    assert all("25.0%" in example["prompt"] for example in examples)


def test_aime_snapshot_is_deterministic_for_a_fixed_seed():
    table = {
        "columns": ["prompt", "response", "scores", "ground_truth"],
        "data": [["problem", str(index) * 40, float(index % 2), ["0"]] for index in range(8)],
    }

    first = build_gen_value_aime_snapshot.build_aime_validation_examples(table, FakeTokenizer(), seed=11)
    second = build_gen_value_aime_snapshot.build_aime_validation_examples(table, FakeTokenizer(), seed=11)

    assert first == second
