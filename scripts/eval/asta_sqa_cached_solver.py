"""Inspect cached solver for ASTA Bench ScholarQA responses."""

import json
from pathlib import Path

from astabench.types.sqa import SQAResponse, SQAResponseWithUsage
from astabench.util.model import record_model_usage_with_inspect
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_error
from inspect_ai.model import ChatMessageAssistant, ModelUsage
from inspect_ai.solver import Generate, Solver, TaskState, solver

set_verbosity_error()
_responses: dict[str, dict] | None = None


def _load_responses(path: str, split: str) -> dict[str, dict]:
    global _responses
    if _responses is None:
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"ASTA response file not found: {data_path}")
        dataset = load_dataset("json", data_files={split: str(data_path)})[split]
        _responses = {row["question"]: row["response"] for row in dataset}
    return _responses


@solver
def cache_solver(path: str, split: str = "test", model: str = "openai/gpt-4.1") -> Solver:
    """Return saved SQA responses by exact question text."""
    responses = _load_responses(path, split)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        question = state.metadata["initial_prompt"]
        if question not in responses:
            raise ValueError(f"No cached result found for question: {question}")
        response_dict = responses[question]
        response = SQAResponseWithUsage(**response_dict) if "tokens" in response_dict else SQAResponse(**response_dict)
        content = json.dumps({"response": response.model_dump(mode="json", exclude={"tokens"})}, indent=2)
        if hasattr(response, "tokens"):
            usage = {f"{key}_tokens": value for key, value in response.tokens.items()}
            record_model_usage_with_inspect(model_name=model, usage=ModelUsage(**usage))
        state.messages.append(ChatMessageAssistant(content=content))
        state.output.completion = content
        return state

    return solve
