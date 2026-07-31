#!/usr/bin/env python3
"""Generate official IFBench responses through an OpenAI-compatible API."""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from tqdm import tqdm


def extract_answer(response: str) -> str:
    """Remove reasoning and answer wrappers before IFBench scores the response."""
    response = response.split("</think>")[-1]
    answer_sections = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    if answer_sections:
        return answer_sections[-1].strip()
    return response.replace("<answer>", "").replace("</answer>", "").strip()


def generate_response(
    client: httpx.Client,
    api_base: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> str:
    response = client.post(
        f"{api_base.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        },
        timeout=3600,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    prompts = [json.loads(line) for line in Path(args.input_file).read_text().splitlines()]
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_by_prompt = {}
    if args.resume and output_path.exists():
        results_by_prompt = {
            row["prompt"]: row for row in (json.loads(line) for line in output_path.read_text().splitlines())
        }
    remaining = [row for row in prompts if row["prompt"] not in results_by_prompt]

    def save() -> None:
        ordered = [results_by_prompt[row["prompt"]] for row in prompts if row["prompt"] in results_by_prompt]
        output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered))

    errors = []
    with httpx.Client() as client, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_response,
                client,
                args.api_base,
                args.model,
                row["prompt"],
                args.temperature,
                args.top_p,
                args.max_tokens,
                args.seed,
            ): row
            for row in remaining
        }
        for completed, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Generating"), 1):
            row = futures[future]
            try:
                raw_response = future.result()
                results_by_prompt[row["prompt"]] = {
                    "prompt": row["prompt"],
                    "response": extract_answer(raw_response),
                    "raw_response": raw_response,
                }
            except Exception as error:
                errors.append((row["key"], str(error)))
            if completed % 10 == 0:
                save()
    save()

    if errors:
        raise RuntimeError(f"Generation failed for {len(errors)} prompts; first errors: {errors[:5]}")
    if len(results_by_prompt) != len(prompts):
        raise RuntimeError(f"Expected {len(prompts)} responses, found {len(results_by_prompt)}")


if __name__ == "__main__":
    main()
