#!/usr/bin/env python3
"""Download focused-domain corpora for local or cloud training."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict

import requests


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Skipping existing file: {destination}")
        return
    print(f"Downloading {url} -> {destination}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)


def _download_zip(url: str, destination_dir: Path, inner_name: str | None = None) -> None:
    import tempfile
    import zipfile

    destination_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        print(f"Fetching archive {url}")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        tmp.write(response.content)
        tmp.flush()
        with zipfile.ZipFile(tmp.name) as zf:
            members = zf.namelist()
            targets = members if inner_name is None else [name for name in members if inner_name in name]
            for member in targets:
                print(f"Extracting {member}")
                zf.extract(member, destination_dir)


def download_science(destination: Path) -> None:
    files: Dict[str, str] = {
        "einstein_relativity.txt": "https://www.gutenberg.org/cache/epub/30155/pg30155.txt",
        "maxwell_treatise.txt": "https://www.gutenberg.org/cache/epub/38447/pg38447.txt",
        "bayesian_inference.txt": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/naive_bayes.rst",
    }
    for name, url in files.items():
        _download_file(url, destination / name)


def download_data_analysis(destination: Path) -> None:
    files: Dict[str, str] = {
        "pandas_groupby.rst": "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/source/user_guide/groupby.rst",
        "sklearn_linear_model.rst": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/linear_model.rst",
        "mlflow_tracking.mdx": "https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/tracking/index.mdx",
        "spark_ml_guide.md": "https://raw.githubusercontent.com/apache/spark/master/docs/ml-guide.md",
    }
    for name, url in files.items():
        _download_file(url, destination / name)


def download_code(destination: Path) -> None:
    files: Dict[str, str] = {
        "pytorch_mnist.py": "https://raw.githubusercontent.com/pytorch/examples/main/mnist/main.py",
        "jax_mnist_classifier.py": "https://raw.githubusercontent.com/google/jax/main/examples/mnist_classifier.py",
        "jax_vae.py": "https://raw.githubusercontent.com/google/jax/main/examples/mnist_vae.py",
        "numpy_linalg.py": "https://raw.githubusercontent.com/numpy/numpy/main/numpy/linalg/linalg.py",
        "sklearn_gb.py": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/ensemble/_gb.py",
    }
    for name, url in files.items():
        _download_file(url, destination / name)


def download_chess(destination: Path) -> None:
    # PGN Mentor archives (public domain / permissive)
    archives = {
        "Morphy": "https://www.pgnmentor.com/players/Morphy.zip",
        "Capablanca": "https://www.pgnmentor.com/players/Capablanca.zip",
    }
    for label, url in archives.items():
        target_dir = destination / label
        _download_zip(url, target_dir)


def download_general(destination: Path) -> None:
    url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
    _download_file(url, destination / "dolly_15k.jsonl")


DOMAIN_DOWNLOADERS = {
    "science": download_science,
    "data": download_data_analysis,
    "code": download_code,
    "chess": download_chess,
    "general": download_general,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download focused-domain corpora")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/cloud"))
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=sorted(DOMAIN_DOWNLOADERS.keys()) + ["all"],
        default=["science", "data", "code", "chess", "general"],
        help="Domain corpora to download",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing domain directories before downloading",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = DOMAIN_DOWNLOADERS.keys() if "all" in args.domains else args.domains

    for domain in selected:
        destination = args.output_dir / domain
        if args.clean and destination.exists():
            print(f"Removing existing data directory: {destination}")
            shutil.rmtree(destination)
        DOMAIN_DOWNLOADERS[domain](destination)

    print("Download complete. Domains available in:")
    for domain in selected:
        print(f"  - {args.output_dir / domain}")

if __name__ == "__main__":
    main()
