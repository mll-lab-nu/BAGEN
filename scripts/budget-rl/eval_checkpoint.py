#!/usr/bin/env python3
"""
Evaluate a model checkpoint on the held-out budget probe test set.

Talks to a vLLM OpenAI-compatible server (start it separately) to run
inference, then scores each prediction with budget_probe_reward.compute_score.

Outputs JSON with summary metrics + per-sample details.

Usage:
    python eval_checkpoint.py \
        --vllm-url http://localhost:8000 \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --test-parquet ablation_data/eval_test/train.parquet \
        --output results/baseline.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets
import openai

from budget_probe_reward import compute_score, parse_answer


def infer_one(client, model_name, messages, max_tokens, temperature):
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=120,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"<<ERROR: {e}>>"


def classify(predicted, ground_truth):
    """Return (pred_class, gt_class) where class is 'impossible'/'possible'/'invalid'."""
    if predicted is None:
        return "invalid", "impossible" if ground_truth == "impossible" else "possible"
    pred_class = "impossible" if predicted == "impossible" else "possible"
    gt_class = "impossible" if ground_truth == "impossible" else "possible"
    return pred_class, gt_class


def covered(predicted, remaining_tokens):
    """For interval predictions, did the interval cover the actual?"""
    if isinstance(predicted, tuple):
        lo, hi = predicted
        return lo <= remaining_tokens <= hi
    if isinstance(predicted, int):
        return predicted == remaining_tokens
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-url", required=True,
                    help="vLLM OpenAI-API base URL, e.g. http://localhost:8000")
    ap.add_argument("--model-name", required=True,
                    help="Model name to send to vLLM (must match served model id)")
    ap.add_argument("--test-parquet", required=True,
                    help="Path to held-out test parquet (rl format)")
    ap.add_argument("--output", required=True,
                    help="Path to write summary JSON")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()

    base_url = args.vllm_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    client = openai.OpenAI(base_url=base_url, api_key=args.api_key)

    ds = datasets.load_dataset("parquet", data_files=args.test_parquet)["train"]
    print(f"loaded {len(ds)} test samples from {args.test_parquet}", flush=True)

    samples = [
        {
            "idx": i,
            "messages": ds[i]["prompt"],
            "ground_truth": ds[i]["reward_model"]["ground_truth"],
            "remaining_tokens": int(ds[i]["extra_info"].get("remaining_tokens", 0)),
            "is_possible": bool(ds[i]["extra_info"].get("is_possible", False)),
            "custom_id": ds[i]["extra_info"].get("custom_id", str(i)),
        }
        for i in range(len(ds))
    ]

    t0 = time.time()
    results = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
        futures = {
            ex.submit(infer_one, client, args.model_name, s["messages"],
                      args.max_tokens, args.temperature): s["idx"]
            for s in samples
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            response_text = fut.result()
            s = samples[idx]
            predicted = parse_answer(response_text)
            extra_info = {"remaining_tokens": s["remaining_tokens"]}
            reward = compute_score(
                "budget_probe_sokoban",
                response_text,
                s["ground_truth"],
                extra_info=extra_info,
            )
            pred_class, gt_class = classify(predicted, s["ground_truth"])
            cov = covered(predicted, s["remaining_tokens"])

            results[idx] = {
                "idx": idx,
                "custom_id": s["custom_id"],
                "ground_truth": s["ground_truth"],
                "remaining_tokens": s["remaining_tokens"],
                "is_possible": s["is_possible"],
                "response": response_text,
                "predicted": str(predicted),
                "reward": reward,
                "pred_class": pred_class,
                "gt_class": gt_class,
                "covered": cov,
            }
            done += 1
            if done % 50 == 0 or done == len(samples):
                elapsed = time.time() - t0
                print(f"  [{done}/{len(samples)}] elapsed={elapsed:.0f}s", flush=True)

    # Aggregate
    n = len(results)
    rewards = [r["reward"] for r in results]
    class_correct = sum(1 for r in results if r["pred_class"] == r["gt_class"])
    parsed = sum(1 for r in results if r["predicted"] != "None")

    # Among possible-truth samples
    possible = [r for r in results if r["gt_class"] == "possible"]
    impossible = [r for r in results if r["gt_class"] == "impossible"]

    cover_rate = (
        sum(1 for r in possible if r["covered"]) / len(possible)
        if possible else 0.0
    )
    pred_hit_possible = (
        sum(1 for r in possible if r["pred_class"] == "possible") / len(possible)
        if possible else 0.0
    )
    pred_hit_impossible = (
        sum(1 for r in impossible if r["pred_class"] == "impossible") / len(impossible)
        if impossible else 0.0
    )

    # Mean Relative Error among possible-and-numeric predictions
    mres = []
    for r in possible:
        pred = parse_answer(r["response"])
        if isinstance(pred, tuple):
            mid = (pred[0] + pred[1]) / 2.0
        elif isinstance(pred, int):
            mid = float(pred)
        else:
            continue
        if r["remaining_tokens"] > 0:
            mres.append(abs(mid - r["remaining_tokens"]) / r["remaining_tokens"])
    mean_mre = sum(mres) / len(mres) if mres else 0.0
    median_mre = sorted(mres)[len(mres) // 2] if mres else 0.0

    summary = {
        "model_name": args.model_name,
        "test_parquet": args.test_parquet,
        "n_samples": n,
        "n_possible": len(possible),
        "n_impossible": len(impossible),
        "mean_reward": sum(rewards) / n if n else 0.0,
        "format_valid_rate": parsed / n if n else 0.0,
        "class_accuracy": class_correct / n if n else 0.0,
        "pred_hit_possible": pred_hit_possible,
        "pred_hit_impossible": pred_hit_impossible,
        "cover_rate_possible": cover_rate,
        "mean_relative_error": mean_mre,
        "median_relative_error": median_mre,
        "wallclock_seconds": time.time() - t0,
    }

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print()
    print("=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nFull results -> {args.output}")


if __name__ == "__main__":
    main()
