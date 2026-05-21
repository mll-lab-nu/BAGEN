#!/usr/bin/env python3
"""
Pretty-print metrics from an eval_checkpoint.py results.json.

Layout (Plan A):
  - Confusion-matrix-style classification block
  - Numeric estimation block (only samples where both truth and pred are intervals)

Usage:
    python analyze_eval.py results/baseline.json
    python analyze_eval.py results/sft_point.json results/sft_interval_pct10.json  # multiple
"""

import argparse
import json
import re
import statistics
import sys

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
INTERVAL_PATTERN = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")
SCALAR_PATTERN = re.compile(r"^\s*(\d+)\s*$")


def parse_pred(response: str):
    m = ANSWER_PATTERN.search(response)
    if not m:
        return None
    text = m.group(1).strip()
    if text.lower() == "impossible":
        return ("impossible",)
    iv = INTERVAL_PATTERN.search(text)
    if iv:
        return ("interval", int(iv.group(1)), int(iv.group(2)))
    sc = SCALAR_PATTERN.match(text)
    if sc:
        return ("scalar", int(sc.group(1)))
    return None


def safe_div(a, b):
    return a / b if b else 0.0


def fmt_pct(x):
    return f"{x*100:.1f}%"


def report(path: str):
    with open(path) as f:
        d = json.load(f)
    summary = d.get("summary", {})
    results = d.get("results", [])

    print("=" * 70)
    print(f"FILE: {path}")
    print(f"MODEL: {summary.get('model_name', '?')}")
    print(f"SAMPLES: {len(results)}")
    print()

    # Classify each sample
    n_pos = sum(1 for r in results if r["gt_class"] == "possible")
    n_imp = sum(1 for r in results if r["gt_class"] == "impossible")

    # Confusion: rows = GT, cols = Pred
    cm = {("possible", "possible"): 0, ("possible", "impossible"): 0, ("possible", "invalid"): 0,
          ("impossible", "possible"): 0, ("impossible", "impossible"): 0, ("impossible", "invalid"): 0}
    for r in results:
        cm[(r["gt_class"], r["pred_class"])] += 1

    print("[Classification quality]")
    print(f"                            Pred=possible       Pred=impossible      Pred=invalid")
    p_p = cm[("possible", "possible")]
    p_i = cm[("possible", "impossible")]
    p_v = cm[("possible", "invalid")]
    i_p = cm[("impossible", "possible")]
    i_i = cm[("impossible", "impossible")]
    i_v = cm[("impossible", "invalid")]
    def cell(n, total):
        if total == 0: return f"{n} (n/a)"
        return f"{n} ({fmt_pct(n/total)})"
    print(f"  GT=possible  (n={n_pos:>3}):  {cell(p_p, n_pos):>20}  {cell(p_i, n_pos):>20}  {cell(p_v, n_pos):>13}")
    print(f"  GT=impossible(n={n_imp:>3}):  {cell(i_p, n_imp):>20}  {cell(i_i, n_imp):>20}  {cell(i_v, n_imp):>13}")
    print()
    overall_acc = safe_div(p_p + i_i, n_pos + n_imp)
    print(f"  → Recall(possible)   = {fmt_pct(safe_div(p_p, n_pos))}    "
          f"(of all really-possible samples)")
    print(f"  → Recall(impossible) = {fmt_pct(safe_div(i_i, n_imp))}    "
          f"(of all really-impossible samples)")
    print(f"  → Overall accuracy   = {fmt_pct(overall_acc)}")
    print(f"  → Format valid rate  = {fmt_pct(safe_div(len(results) - p_v - i_v, len(results)))}")
    print()

    # Numeric estimation: only samples with GT=possible AND pred=interval-or-scalar
    numeric_samples = []
    for r in results:
        if r["gt_class"] != "possible":
            continue
        pred = parse_pred(r["response"])
        if not pred or pred[0] not in ("interval", "scalar"):
            continue
        actual = int(r["remaining_tokens"])
        if actual <= 0:
            continue
        if pred[0] == "interval":
            lo, hi = pred[1], pred[2]
            mid = (lo + hi) / 2.0
            width = hi - lo
            covered = lo <= actual <= hi
        else:
            mid = float(pred[1])
            width = 0.0
            covered = pred[1] == actual
        rel_err = abs(mid - actual) / actual
        rel_width = width / actual
        numeric_samples.append({
            "actual": actual,
            "mid": mid,
            "width": width,
            "rel_err": rel_err,
            "rel_width": rel_width,
            "covered": covered,
        })

    print(f"[Numeric estimation]  (only samples with GT=possible AND pred is a number/interval)")
    if not numeric_samples:
        print("  (no samples in this category)")
        print()
        return
    n = len(numeric_samples)
    covered = [s for s in numeric_samples if s["covered"]]
    missed  = [s for s in numeric_samples if not s["covered"]]

    def stats(lst, key):
        if not lst: return None, None
        vals = [s[key] for s in lst]
        return statistics.mean(vals), statistics.median(vals)

    print(f"  total numeric predictions: {n}  (covered={len(covered)}, missed={len(missed)})")
    print()
    headers = ["subset", "n", "midpoint vs actual (mean)", "midpoint vs actual (median)",
               "rel_width (mean)", "rel_width (median)"]
    rows = []
    for label, lst in [("ALL numeric", numeric_samples),
                        ("covered (in range)", covered),
                        ("missed  (out of range)", missed)]:
        if not lst:
            rows.append([label, 0, "n/a", "n/a", "n/a", "n/a"])
            continue
        re_mean, re_med = stats(lst, "rel_err")
        rw_mean, rw_med = stats(lst, "rel_width")
        rows.append([label, len(lst),
                     fmt_pct(re_mean), fmt_pct(re_med),
                     f"{rw_mean:.2f}", f"{rw_med:.2f}"])

    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))
    print()

    # Cover rate as fraction of all numeric predictions
    cov_rate = len(covered) / n
    print(f"  Cover rate (interval contains actual): {len(covered)}/{n} = {fmt_pct(cov_rate)}")
    print(f"  Miss rate (out of range):              {len(missed)}/{n} = {fmt_pct(1 - cov_rate)}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="results.json files to analyze")
    args = ap.parse_args()
    for p in args.paths:
        report(p)


if __name__ == "__main__":
    main()
