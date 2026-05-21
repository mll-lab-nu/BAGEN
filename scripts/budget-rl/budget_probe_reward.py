"""
Rule-based reward function for budget probe estimation.

Reward:
  invalid format (no <answer>...</answer>)             -> 0.0
  predicts "impossible" when GT is impossible           -> IMPOSSIBLE_WEIGHT
  predicts "impossible" when GT is possible             -> 0.0
  predicts interval/scalar when GT is impossible        -> 0.0
  predicts interval [L, H] when GT is possible:
      uncovered (actual ∉ [L,H])                       -> 0.0
      covered:  accuracy = max(0, 1 - width/actual)    -> accuracy * POSSIBLE_WEIGHT

Also logs per-sample detailed metrics to a JSONL side file
(path from env REWARD_METRICS_LOG, default /tmp/reward_metrics.jsonl).
"""

import json
import math
import os
import re
import threading

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
INTERVAL_PATTERN = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")
SCALAR_PATTERN = re.compile(r"^\s*(\d+)\s*$")

POSSIBLE_WEIGHT = 1.8
IMPOSSIBLE_WEIGHT = 0.2
FORMAT_REWARD = 0.0
COVERAGE_SCALE = 1.0

# --- Metrics side-channel ---
_log_file = None
_log_lock = threading.Lock()
_batch_buffer = []
_flush_every = 200  # aggregate every N samples

def _ensure_log():
    global _log_file
    if _log_file is None:
        path = os.environ.get("REWARD_METRICS_LOG", "/tmp/reward_metrics.jsonl")
        _log_file = open(path, "a", buffering=1)

def _log_sample(info: dict):
    _batch_buffer.append(info)
    if len(_batch_buffer) >= _flush_every:
        _flush_batch()

def _flush_batch():
    global _batch_buffer
    if not _batch_buffer:
        return
    buf = list(_batch_buffer)
    _batch_buffer = []

    n = len(buf)
    n_possible_gt = sum(1 for s in buf if s["gt_class"] == "possible")
    n_impossible_gt = n - n_possible_gt

    # Classification
    class_correct = sum(1 for s in buf if s["pred_class"] == s["gt_class"])
    pred_hit_pos = sum(1 for s in buf if s["gt_class"] == "possible" and s["pred_class"] == "possible")
    pred_hit_imp = sum(1 for s in buf if s["gt_class"] == "impossible" and s["pred_class"] == "impossible")

    # Coverage (among possible-gt AND interval-pred samples)
    possible_with_interval = [s for s in buf if s["gt_class"] == "possible" and s["has_interval"]]
    n_covered = sum(1 for s in possible_with_interval if s["covered"])
    n_interval = len(possible_with_interval)

    # Width & MRE (among covered)
    covered = [s for s in possible_with_interval if s["covered"]]
    widths = [s["width"] for s in possible_with_interval if s["width"] is not None]
    mres = [s["rel_error"] for s in possible_with_interval if s["rel_error"] is not None]

    # Reward breakdown
    rewards_possible = [s["reward"] for s in buf if s["gt_class"] == "possible"]
    rewards_impossible = [s["reward"] for s in buf if s["gt_class"] == "impossible"]

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0
    def safe_median(lst):
        if not lst: return 0.0
        s = sorted(lst)
        return s[len(s) // 2]

    summary = {
        "n": n,
        "class_accuracy": class_correct / n if n else 0,
        "pred_hit_possible": pred_hit_pos / n_possible_gt if n_possible_gt else 0,
        "pred_hit_impossible": pred_hit_imp / n_impossible_gt if n_impossible_gt else 0,
        "cover_rate": n_covered / n_interval if n_interval else 0,
        "n_covered": n_covered,
        "n_interval": n_interval,
        "reward_all_mean": safe_mean([s["reward"] for s in buf]),
        "reward_possible_mean": safe_mean(rewards_possible),
        "reward_impossible_mean": safe_mean(rewards_impossible),
        "width_mean": safe_mean(widths),
        "width_median": safe_median(widths),
        "mre_mean": safe_mean(mres),
        "mre_median": safe_median(mres),
    }

    _ensure_log()
    with _log_lock:
        _log_file.write(json.dumps({"type": "batch_summary", **summary}) + "\n")


def parse_answer(solution_str):
    match = ANSWER_PATTERN.search(solution_str)
    if match is None:
        return None
    answer_text = match.group(1).strip()
    if answer_text.lower() == "impossible":
        return "impossible"
    interval_match = INTERVAL_PATTERN.search(answer_text)
    if interval_match:
        return int(interval_match.group(1)), int(interval_match.group(2))
    scalar_match = SCALAR_PATTERN.match(answer_text)
    if scalar_match:
        return int(scalar_match.group(1))
    return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    predicted = parse_answer(solution_str)
    gt_is_impossible = ground_truth == "impossible"
    remaining_tokens = extra_info.get("remaining_tokens", 0) if extra_info else 0

    # Build per-sample info for metrics
    sample_info = {
        "gt_class": "impossible" if gt_is_impossible else "possible",
        "pred_class": "invalid",
        "has_interval": False,
        "covered": False,
        "width": None,
        "rel_error": None,
        "reward": 0.0,
    }

    if predicted is None:
        _log_sample(sample_info)
        return 0.0

    if gt_is_impossible:
        sample_info["pred_class"] = "impossible" if predicted == "impossible" else "possible"
        reward = (1.0 if predicted == "impossible" else 0.0) * IMPOSSIBLE_WEIGHT
        sample_info["reward"] = reward
        _log_sample(sample_info)
        return reward

    # GT is possible
    if predicted == "impossible":
        sample_info["pred_class"] = "impossible"
        _log_sample(sample_info)
        return 0.0

    sample_info["pred_class"] = "possible"

    if isinstance(predicted, tuple):
        est_low, est_high = predicted
        sample_info["has_interval"] = True
        if est_low > est_high:
            _log_sample(sample_info)
            return 0.0

        sample_info["width"] = est_high - est_low
        mid = (est_low + est_high) / 2.0
        if remaining_tokens > 0:
            sample_info["rel_error"] = abs(mid - remaining_tokens) / remaining_tokens

        if remaining_tokens == 0:
            accuracy = 1.0 if (est_low == 0 and est_high == 0) else 0.0
        elif est_low <= remaining_tokens <= est_high:
            accuracy = max(0.0, 1.0 - (est_high - est_low) / remaining_tokens)
            sample_info["covered"] = True
        else:
            accuracy = 0.0
    else:
        # Scalar prediction — not a valid interval format, no reward
        accuracy = 0.0

    reward = (FORMAT_REWARD + COVERAGE_SCALE * accuracy) * POSSIBLE_WEIGHT
    sample_info["reward"] = reward
    _log_sample(sample_info)
    return reward
