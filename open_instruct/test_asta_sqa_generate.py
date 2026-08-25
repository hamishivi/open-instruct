from scripts.eval import asta_sqa_generate
from scripts.eval.asta_sqa_generate import build_asta_record, extract_final_answer, format_search_results


def test_format_search_results_builds_dr_tulu_snippets() -> None:
    response = {
        "data": [{"snippet": {"text": "Evidence text.", "snippetKind": "body"}, "paper": {"title": "Evidence Paper"}}]
    }

    formatted, catalogue = format_search_results(response, "call/1")

    assert "<snippet id=call_1-0>" in formatted
    assert "Title: Evidence Paper" in formatted
    assert "Snippet: Evidence text." in formatted
    assert catalogue["call_1-0"] == {"title": "Evidence Paper", "snippets": ["Evidence text."]}


def test_extract_final_answer_removes_reasoning_wrapper() -> None:
    response = "<think>research</think><answer>Final <cite id=source-1>claim</cite>.</answer>"

    assert extract_final_answer(response) == "Final <cite id=source-1>claim</cite>."


def test_build_asta_record_preserves_sections_and_citations() -> None:
    answer = """# Finding
<cite id="call-1-0,call-2-0">A supported claim.</cite>

# Caveat
Uncited uncertainty."""
    catalogue = {
        "call-1-0": {"title": "Paper One", "snippets": ["First snippet"]},
        "call-2-0": {"title": "Paper Two", "snippets": ["Second snippet"]},
    }

    record = build_asta_record("Question?", answer, catalogue)

    assert record["question"] == "Question?"
    sections = record["response"]["sections"]
    assert sections[0]["title"] == "# Finding"
    assert sections[0]["text"] == "A supported claim [call-1-0][call-2-0]."
    assert sections[0]["citations"] == [
        {"id": "[call-1-0]", "title": "Paper One", "snippets": ["First snippet"]},
        {"id": "[call-2-0]", "title": "Paper Two", "snippets": ["Second snippet"]},
    ]
    assert sections[1] == {"title": "# Caveat", "text": "Uncited uncertainty.", "citations": []}


def test_generate_case_runs_native_tool_loop(monkeypatch) -> None:
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {"name": "snippet_search", "arguments": '{"query":"evidence"}'},
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": '<answer><cite id="call-1-0">Supported claim.</cite></answer>'}}]},
        ]
    )
    monkeypatch.setattr(asta_sqa_generate, "request_json", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(
        asta_sqa_generate,
        "semantic_scholar_search",
        lambda *args, **kwargs: {
            "data": [{"paper": {"title": "Paper"}, "snippet": {"text": "Evidence", "snippetKind": "body"}}]
        },
    )

    result = asta_sqa_generate.generate_case(
        {"case_id": "case", "question": "Question?"},
        api_base="http://localhost/v1",
        model="model",
        system_prompt="prompt",
        max_tool_calls=2,
        num_docs=10,
        temperature=0.6,
        top_p=0.95,
        max_tokens=100,
        request_timeout=1,
        s2_timeout=1,
    )

    assert result["tool_calls"] == 1
    assert result["final_response"] == '<cite id="call-1-0">Supported claim.</cite>'
    assert result["asta_record"]["response"]["sections"] == [
        {
            "title": None,
            "text": "Supported claim [call-1-0].",
            "citations": [{"id": "[call-1-0]", "title": "Paper", "snippets": ["Evidence"]}],
        }
    ]


def test_generate_case_returns_search_errors_to_model(monkeypatch) -> None:
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {"name": "snippet_search", "arguments": '{"query":"A* search"}'},
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "<answer>Recovered answer.</answer>"}}]},
        ]
    )
    requests = []

    def request_json(*args, **kwargs):
        requests.append(kwargs["payload"])
        return next(responses)

    def search_fails(*args, **kwargs):
        raise RuntimeError("HTTP 400: unsupported wildcard")

    monkeypatch.setattr(asta_sqa_generate, "request_json", request_json)
    monkeypatch.setattr(asta_sqa_generate, "semantic_scholar_search", search_fails)

    result = asta_sqa_generate.generate_case(
        {"case_id": "case", "question": "Question?"},
        api_base="http://localhost/v1",
        model="model",
        system_prompt="prompt",
        max_tool_calls=2,
        num_docs=10,
        temperature=0.6,
        top_p=0.95,
        max_tokens=100,
        request_timeout=1,
        s2_timeout=1,
    )

    tool_message = requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "unsupported wildcard" in tool_message["content"]
    assert "rephrase" in tool_message["content"]
    assert result["tool_calls"] == 1
    assert result["final_response"] == "Recovered answer."
