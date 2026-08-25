#!/usr/bin/env python3
"""Generate ScholarQA responses with a local vLLM server and Semantic Scholar.

The output format matches the cached-solver input used by DR-Tulu's
self-contained ASTA Bench evaluation recipe.
"""

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

S2_ENDPOINT = "https://api.semanticscholar.org/graph/v1/snippet/search"
SQA_INSTRUCTION = (
    "Please write a well structured, data-driven report on the given research question, and add citations when needed."
)
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "snippet_search",
        "description": "Retrieve focused snippets from scientific papers through Semantic Scholar.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The scientific-literature search query."}},
            "required": ["query"],
        },
    },
}


def request_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 3600,
    attempts: int = 4,
) -> dict[str, Any]:
    """Make a JSON request with bounded retries for transient failures."""
    request_headers = {"User-Agent": "open-instruct-asta-eval/1.0", **(headers or {})}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code != 429 and error.code < 500:
                body = error.read().decode("utf-8", errors="replace")[:1000]
                raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error

        if attempt + 1 < attempts:
            time.sleep(min(2**attempt, 30))

    raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from last_error


def semantic_scholar_search(query: str, *, num_docs: int, timeout: float) -> dict[str, Any]:
    api_key = os.environ.get("S2_API_KEY")
    if not api_key:
        raise RuntimeError("S2_API_KEY is required")
    query_string = urllib.parse.urlencode({"query": query, "limit": num_docs})
    return request_json(f"{S2_ENDPOINT}?{query_string}", headers={"x-api-key": api_key}, timeout=timeout)


def safe_citation_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", value)


def format_search_results(response: dict[str, Any], call_id: str) -> tuple[str, dict[str, dict[str, Any]]]:
    """Format S2 results like DR-Tulu and return the citation catalogue."""
    formatted = []
    catalogue: dict[str, dict[str, Any]] = {}
    safe_call_id = safe_citation_id(call_id)
    for index, item in enumerate(response.get("data", [])):
        snippet = item.get("snippet") or {}
        paper = item.get("paper") or {}
        title = str(paper.get("title") or "").strip()
        text = str(snippet.get("text") or "").strip()
        if snippet.get("snippetKind") == "title":
            text = ""
        if not title and not text:
            continue

        citation_id = f"{safe_call_id}-{index}"
        body_parts = []
        if title:
            body_parts.append(f"Title: {title}")
        if text:
            body_parts.append(f"Snippet: {text}")
        body = "\n".join(body_parts)
        formatted.append(f"<snippet id={citation_id}>\n{body}\n</snippet>")
        catalogue[citation_id] = {"title": title, "snippets": [text or title]}

    if not formatted:
        return "Query returned no results.", catalogue
    return "\n".join(formatted), catalogue


def parse_tool_arguments(arguments: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    parsed = json.loads(arguments)
    if isinstance(parsed, str):
        return {"query": parsed}
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected object tool arguments, got {type(parsed).__name__}")
    return parsed


def extract_final_answer(response: str) -> str:
    """Remove reasoning/answer wrappers while retaining citation markup."""
    answer_matches = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    if answer_matches:
        return answer_matches[-1].strip()
    if "</think>" in response:
        response = response.rsplit("</think>", maxsplit=1)[-1]
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    return response.replace("<answer>", "").replace("</answer>", "").strip()


def _replace_citations(text: str) -> tuple[str, list[str]]:
    citation_ids: list[str] = []
    pattern = re.compile(r"<cite\s+id=[\"']?([^\"'>]+)[\"']?\s*>(.*?)</cite>", re.DOTALL | re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        ids = [item.strip() for item in match.group(1).split(",") if item.strip()]
        citation_ids.extend(ids)
        claim = match.group(2).strip()
        markers = "".join(f"[{citation_id}]" for citation_id in ids)
        if claim and claim[-1] in ".!?,;:":
            return f"{claim[:-1].rstrip()} {markers}{claim[-1]}"
        return f"{claim} {markers}".strip()

    return pattern.sub(replace, text).strip(), citation_ids


def _section_blocks(answer: str) -> list[tuple[str | None, str]]:
    blocks: list[tuple[str | None, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in answer.splitlines():
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading:
            if current_lines or current_title is not None:
                blocks.append((current_title, "\n".join(current_lines).strip()))
            current_title = f"# {heading.group(1).strip()}"
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines or current_title is not None:
        blocks.append((current_title, "\n".join(current_lines).strip()))
    if not blocks:
        blocks.append((None, answer.strip()))
    return [(title, text) for title, text in blocks if title is not None or text]


def build_asta_record(
    question: str, final_response: str, citation_catalogue: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Convert a DR-Tulu answer and its retrieved snippets to ASTA's SQA schema."""
    sections = []
    for title, section_text in _section_blocks(final_response):
        clean_text, section_ids = _replace_citations(section_text)
        unique_ids = list(dict.fromkeys(section_ids))
        citations = []
        for citation_id in unique_ids:
            citation = citation_catalogue.get(citation_id, {})
            snippets = citation.get("snippets") or [""]
            citations.append({"id": f"[{citation_id}]", "title": citation.get("title") or "", "snippets": snippets})
        sections.append({"title": title, "text": clean_text, "citations": citations})

    if not sections:
        sections.append({"title": None, "text": "", "citations": []})
    return {"question": question, "response": {"sections": sections}}


def _assistant_transcript(message: dict[str, Any]) -> str:
    parts = []
    reasoning = message.get("reasoning_content")
    content = message.get("content")
    if reasoning:
        parts.append(f"<think>{reasoning}</think>")
    if content:
        parts.append(str(content))
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        parts.append(
            "<tool_call>"
            + json.dumps({"name": function.get("name"), "arguments": function.get("arguments")}, ensure_ascii=False)
            + "</tool_call>"
        )
    return "\n".join(parts)


def generate_case(
    sample: dict[str, Any],
    *,
    api_base: str,
    model: str,
    system_prompt: str,
    max_tool_calls: int,
    num_docs: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    request_timeout: float,
    s2_timeout: float,
) -> dict[str, Any]:
    question = str(sample["question"])
    case_id = str(sample.get("case_id") or sample.get("id") or question)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{question}\n\n{SQA_INSTRUCTION}"},
    ]
    transcript: list[str] = []
    citation_catalogue: dict[str, dict[str, Any]] = {}
    tool_calls_used = 0

    while True:
        allow_tools = tool_calls_used < max_tool_calls
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if allow_tools:
            payload["tools"] = [TOOL_DEFINITION]
            payload["tool_choice"] = "auto"

        response = request_json(
            f"{api_base.rstrip('/')}/chat/completions", method="POST", payload=payload, timeout=request_timeout
        )
        message = response["choices"][0]["message"]
        transcript.append(_assistant_transcript(message))
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            final_response = extract_final_answer(str(message.get("content") or ""))
            return {
                "case_id": case_id,
                "question": question,
                "problem": question,
                "final_response": final_response,
                "full_trace": "\n".join(transcript),
                "citation_catalogue": citation_catalogue,
                "tool_calls": tool_calls_used,
                "asta_record": build_asta_record(question, final_response, citation_catalogue),
            }

        remaining_calls = max_tool_calls - tool_calls_used
        selected_tool_calls = tool_calls[:remaining_calls]
        for index, tool_call in enumerate(selected_tool_calls):
            if not tool_call.get("id"):
                tool_call["id"] = f"call-{safe_citation_id(case_id)}-{tool_calls_used + index}"
        assistant_message = {"role": "assistant", "content": message.get("content"), "tool_calls": selected_tool_calls}
        messages.append(assistant_message)
        for tool_call in selected_tool_calls:
            function = tool_call.get("function") or {}
            if function.get("name") != "snippet_search":
                tool_output = f"Unsupported tool: {function.get('name')}"
            else:
                arguments = parse_tool_arguments(function.get("arguments") or "{}")
                query = str(arguments.get("query") or "").strip()
                if not query:
                    tool_output = "Empty query. Please provide a search query."
                else:
                    try:
                        search_response = semantic_scholar_search(query, num_docs=num_docs, timeout=s2_timeout)
                    except RuntimeError as error:
                        tool_output = (
                            f"Semantic Scholar search failed: {error}. "
                            "Please rephrase the query and try again. Avoid wildcard characters such as *."
                        )
                    else:
                        tool_output, citations = format_search_results(
                            search_response, str(tool_call.get("id") or case_id)
                        )
                        citation_catalogue.update(citations)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_call.get("id") or f"call-{tool_calls_used}"),
                    "name": str(function.get("name") or "snippet_search"),
                    "content": tool_output,
                }
            )
            transcript.append(tool_output)
            tool_calls_used += 1


def load_input(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    for index, sample in enumerate(data):
        if "question" not in sample:
            raise ValueError(f"Input row {index} is missing 'question'")
    return data


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {str(row["case_id"]): row for row in rows}


def write_ordered_outputs(
    samples: list[dict[str, Any]], results: dict[str, dict[str, Any]], raw_output: Path, asta_output: Path
) -> None:
    ordered = []
    for sample in samples:
        case_id = str(sample.get("case_id") or sample.get("id") or sample["question"])
        if case_id in results:
            ordered.append(results[case_id])
    raw_output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered))
    asta_output.write_text(json.dumps([row["asta_record"] for row in ordered], ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--raw-output-file", type=Path, required=True)
    parser.add_argument("--asta-output-file", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt-file", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=10)
    parser.add_argument("--num-docs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--s2-timeout", type=float, default=180)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("S2_API_KEY"):
        parser.error("S2_API_KEY must be set")
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.max_samples == 0 or args.max_samples < -1:
        parser.error("--max-samples must be -1 or a positive integer")

    samples = load_input(args.input_file)
    end = None if args.max_samples == -1 else args.offset + args.max_samples
    samples = samples[args.offset : end]
    args.raw_output_file.parent.mkdir(parents=True, exist_ok=True)
    args.asta_output_file.parent.mkdir(parents=True, exist_ok=True)
    results = load_existing(args.raw_output_file) if args.resume else {}
    if not args.resume:
        args.raw_output_file.write_text("")
    system_prompt = args.system_prompt_file.read_text().strip()

    pending = []
    for sample in samples:
        case_id = str(sample.get("case_id") or sample.get("id") or sample["question"])
        if case_id not in results:
            pending.append(sample)

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_sample = {
            executor.submit(
                generate_case,
                sample,
                api_base=args.api_base,
                model=args.model,
                system_prompt=system_prompt,
                max_tool_calls=args.max_tool_calls,
                num_docs=args.num_docs,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                request_timeout=args.request_timeout,
                s2_timeout=args.s2_timeout,
            ): sample
            for sample in pending
        }
        for completed, future in enumerate(concurrent.futures.as_completed(future_to_sample), start=1):
            sample = future_to_sample[future]
            case_id = str(sample.get("case_id") or sample.get("id") or sample["question"])
            try:
                result = future.result()
                results[case_id] = result
                with args.raw_output_file.open("a") as output_file:
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"Completed {completed}/{len(pending)}: {case_id}", flush=True)
            except Exception as error:
                errors.append((case_id, str(error)))
                print(f"Failed {case_id}: {error}", flush=True)

    write_ordered_outputs(samples, results, args.raw_output_file, args.asta_output_file)
    if errors:
        raise RuntimeError(f"Generation failed for {len(errors)} cases; first failures: {errors[:5]}")
    if len(results) < len(samples):
        raise RuntimeError(f"Expected {len(samples)} results, found {len(results)}")


if __name__ == "__main__":
    main()
