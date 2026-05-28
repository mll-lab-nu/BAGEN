#!/usr/bin/env python3

import os
import sys
from huggingface_hub import snapshot_download

# Default project dataset repo. Override with RAGEN_DATA_REPO_ID.
DEFAULT_DATA_REPO_ID = os.environ.get("RAGEN_DATA_REPO_ID", "ZihanWang314/ragen-datasets")


def download_datasets(repo_id=DEFAULT_DATA_REPO_ID, local_dir="data"):
    """
    Download all datasets from Hugging Face Hub to local directory.

    Args:
        repo_id (str): Hugging Face repository ID
        local_dir (str): Local directory to save datasets
    """
    if not repo_id:
        raise ValueError(
            "No dataset repo id provided. Set RAGEN_DATA_REPO_ID to a Hugging Face "
            "dataset repo id (default: ZihanWang314/ragen-datasets)."
        )

    print(f"Downloading datasets from {repo_id}...")

    countdown_data_url = os.environ.get("COUNTDOWN_DATA_URL", "")
    if countdown_data_url:
        os.makedirs("data/countdown", exist_ok=True)
        os.system(f"wget {countdown_data_url} -O data/countdown/train.parquet")

    # Create the data directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    try:
        # Download the entire repository
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"\nDatasets successfully downloaded to {local_dir}/")
        return True

    except Exception as e:
        print(f"Error downloading datasets: {e}")
        return False


if __name__ == "__main__":
    ok = download_datasets()
    if not ok:
        sys.exit(1)
