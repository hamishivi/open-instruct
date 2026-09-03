import pytest

from open_instruct import value_estimation


def test_collate_generative_value_generations_preserves_text_and_parse_failures():
    generations = [
        "The state looks promising. <answer>8</answer>",
        "I cannot assign a score reliably.",
        "This branch is unlikely. <answer>2</answer>",
    ]
    positions = [(1, 0), (0, 1), (0, 0)]

    predictions, raw_generations = value_estimation._collate_generative_value_generations(
        generations, positions, [2, 1], score_min=0.0, score_max=10.0
    )

    assert predictions == [[0.2, None], [0.8]]
    assert raw_generations == [
        ["This branch is unlikely. <answer>2</answer>", "I cannot assign a score reliably."],
        ["The state looks promising. <answer>8</answer>"],
    ]


@pytest.mark.parametrize(
    ("generations", "positions", "num_probes_per_row", "error"),
    [
        (["<answer>1</answer>"], [], [1], ValueError),
        (["<answer>1</answer>"], [(1, 0)], [1], IndexError),
        (["<answer>1</answer>"], [(0, 1)], [1], IndexError),
    ],
)
def test_collate_generative_value_generations_rejects_invalid_layout(
    generations, positions, num_probes_per_row, error
):
    with pytest.raises(error):
        value_estimation._collate_generative_value_generations(
            generations, positions, num_probes_per_row, score_min=0.0, score_max=10.0
        )
