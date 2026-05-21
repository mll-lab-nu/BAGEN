#!/usr/bin/env python3
"""
Convert RAGEN rollout .pkl files to OpenAI-compatible JSONL format.

Usage:
    python scripts/convert_to_jsonl.py --input results/eval/val_rollouts_*.pkl --output trajectories.jsonl
    python scripts/convert_to_jsonl.py --input results/eval/val_rollouts_*.pkl  # auto-generates output filename
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from verl import DataProto


def extract_openai_messages(history: List[Dict]) -> List[Dict[str, str]]:
    """
    Extract OpenAI-compatible message format from history.

    Format: [{"role": "user"|"assistant", "content": str}, ...]
    """
    messages = []

    for i, turn in enumerate(history):
        # Add user message (environment state)
        if 'state' in turn:
            state_content = turn['state']
            if i == 0:
                # Initial state
                messages.append({
                    "role": "user",
                    "content": state_content
                })
            else:
                # Feedback from environment after action
                reward = turn.get('reward', 0)
                info_str = f" (reward: {reward})" if reward != 0 else ""
                messages.append({
                    "role": "user",
                    "content": f"{state_content}{info_str}"
                })

        # Add assistant message (LLM response with actions)
        if 'llm_response' in turn:
            llm_content = turn.get('llm_raw_response', turn.get('llm_response', ''))
            if llm_content:
                messages.append({
                    "role": "assistant",
                    "content": str(llm_content)
                })

    return messages


def infer_success(history: List[Dict], total_reward: float) -> bool:
    for turn in history:
        info = turn.get("info") or {}
        if bool(info.get("success", False)):
            return True
        if bool(turn.get("success", False)):
            return True
    return bool(total_reward > 0)


def rollout_to_openai_format(item: Any, index: int) -> Dict[str, Any]:
    """
    Convert a single rollout to OpenAI-compatible format.

    Returns:
        {
            "custom_id": "traj_{index}",
            "messages": [...],
            "metadata": {
                "success": bool,
                "reward": float,
                "num_turns": int,
                "env_id": int,
                "group_id": int,
                ...
            }
        }
    """
    ntb = item.non_tensor_batch or {}
    meta = item.meta_info or {}

    # Extract history
    history = ntb.get('history', [])
    messages = extract_openai_messages(history)

    # Extract metadata - safe access to batch (avoid tensordict boolean conversion)
    total_reward = 0.0
    try:
        if item.batch is not None and 'rm_scores' in item.batch:
            rm_scores = item.batch['rm_scores']
            total_reward = float(np.sum(rm_scores))
    except (AttributeError, KeyError, TypeError):
        pass

    metadata = {
        "env_id": int(ntb.get('env_ids', index)),
        "group_id": int(ntb.get('group_ids', 0)),
        "num_turns": len([h for h in history if 'actions' in h]),
        "total_reward": total_reward,
        "success": infer_success(history, total_reward),
    }

    # Add metrics if available
    if 'metrics' in ntb:
        metrics = ntb['metrics']
        if isinstance(metrics, dict):
            metadata['success'] = bool(metrics.get('success', metadata['success']))
            metadata.update({k: v for k, v in metrics.items() if k != 'success'})

    # Add entropy info if available
    if 'entropys' in ntb:
        metadata['entropy'] = float(ntb['entropys'])
    if 'n_generated_tokens' in ntb:
        metadata['n_tokens'] = int(ntb['n_generated_tokens'])

    return {
        "custom_id": f"traj_{index}",
        "messages": messages,
        "metadata": metadata
    }


def convert_pkl_to_jsonl(input_path: Path, output_path: Path) -> None:
    """Convert a DataProto .pkl file to OpenAI-compatible JSONL."""
    print(f"Loading rollout data from {input_path}...")
    data = DataProto.load_from_disk(str(input_path))

    total = len(data)
    print(f"Found {total} trajectories")

    success_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx in range(total):
            try:
                item = data[idx]
                openai_obj = rollout_to_openai_format(item, idx)
                f.write(json.dumps(openai_obj, ensure_ascii=False) + '\n')
                success_count += 1
            except Exception as e:
                print(f"Warning: Failed to convert trajectory {idx}: {e}")
                continue

    print(f"Successfully converted {success_count}/{total} trajectories to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RAGEN rollout .pkl files to OpenAI-compatible JSONL"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input .pkl file"
    )
    parser.add_argument(
        "--output",
        help="Path to output .jsonl file (default: auto-generated from input)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Auto-generate output path if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.jsonl')

    convert_pkl_to_jsonl(input_path, output_path)


if __name__ == "__main__":
    main()
