#!/usr/bin/env python
"""Exercise one real generative-critic optimizer step at a requested pack length."""

import argparse
import json
import time
from types import SimpleNamespace

import torch

from open_instruct.grpo_fast_genvalue import GenValueTrainerActor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pack-length", type=int, required=True)
    parser.add_argument("--example-length", type=int, default=4096)
    parser.add_argument("--generated-length", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pack_length <= 0 or args.example_length <= 1 or args.generated_length <= 0:
        raise ValueError("Pack, example, and generation lengths must be positive.")
    if args.generated_length >= args.example_length:
        raise ValueError("Generated length must be smaller than the complete example length.")
    trainer_class = GenValueTrainerActor.__ray_metadata__.modified_class
    trainer = trainer_class(
        model_path=args.model,
        model_revision=None,
        tokenizer_path=args.model,
        tokenizer_revision=None,
        learning_rate=2e-7,
        score_min=0.0,
        score_max=10.0,
        max_sequence_tokens=32768,
        pack_length=args.pack_length,
        attn_implementation="flash_attention_2",
        gradient_checkpointing=True,
        temperature=1.0,
        reinforce_coef=1.0,
        reinforce_baseline="none",
        pool_shared_state_returns=False,
        fused_optimizer=True,
        trace_reservoir_size=0,
    )

    training_pairs = []
    example_lengths = []
    remaining_tokens = args.pack_length
    while remaining_tokens:
        sequence_length = min(args.example_length, remaining_tokens)
        if sequence_length <= args.generated_length:
            raise ValueError(
                "The final synthetic example must be longer than the generated length; "
                f"got {sequence_length} <= {args.generated_length}."
            )
        example_lengths.append(sequence_length)
        remaining_tokens -= sequence_length

    for index, sequence_length in enumerate(example_lengths):
        prompt_length = sequence_length - args.generated_length
        prompt_ids = [42 + index] * prompt_length
        generated_ids = [142 + index] * args.generated_length
        completion = SimpleNamespace(
            text="<answer>5</answer>", token_ids=generated_ids, logprobs=[-8.0] * args.generated_length
        )
        training_pairs.append(
            {
                "outcome": float(index % 2),
                "state_kind": "segment_start",
                "request_output": SimpleNamespace(prompt_token_ids=prompt_ids, outputs=[completion]),
            }
        )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    metrics = trainer.reinforce_step(training_pairs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at
    result = {
        "pack_length": args.pack_length,
        "num_examples": len(example_lengths),
        "example_lengths": example_lengths,
        "example_length": args.example_length,
        "generated_length": args.generated_length,
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
