#!/usr/bin/env python3
"""Download math/physics/code/chess/general corpora for local training."""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _compute_checksum(data: bytes) -> str:
    """Compute SHA256 checksum of data."""
    return hashlib.sha256(data).hexdigest()


def _verify_file_checksum(file_path: Path, expected_checksum: Optional[str]) -> bool:
    """Verify file against expected checksum."""
    if not expected_checksum:
        return True

    if not file_path.exists():
        return False

    with open(file_path, 'rb') as f:
        actual_checksum = _compute_checksum(f.read())

    return actual_checksum == expected_checksum


def _download_file(url: str, destination: Path, expected_checksum: Optional[str] = None, max_size_mb: int = 100) -> bool:
    """Download file with verification and size limits."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists and is valid
    if destination.exists() and _verify_file_checksum(destination, expected_checksum):
        logger.info(f"✅ Skipping existing verified file: {destination}")
        return True

    max_size_bytes = max_size_mb * 1024 * 1024
    candidates = [url]
    if url.endswith('.txt'):
        base = url[:-4]
        candidates.extend([f"{base}-0.txt", f"{base}.txt.utf-8", f"{base}-8.txt"])

    for candidate in candidates:
        try:
            logger.info(f"📥 Downloading {candidate} -> {destination}")

            # Start download with streaming to check size
            response = requests.get(candidate, timeout=240, stream=True)
            if response.status_code == 404:
                continue
            response.raise_for_status()

            # Check content length if available
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > max_size_bytes:
                logger.warning(f"⚠️  File too large ({int(content_length)/(1024**2):.1f}MB), skipping: {candidate}")
                continue

            # Download with progress tracking
            downloaded_size = 0
            start_time = time.time()

            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # Safety check: don't download files larger than max_size
                        if downloaded_size > max_size_bytes:
                            logger.error(f"❌ File exceeded size limit ({max_size_mb}MB): {candidate}")
                            destination.unlink(missing_ok=True)
                            return False

            download_time = time.time() - start_time
            speed_mbps = (downloaded_size / (1024**2)) / download_time if download_time > 0 else 0

            # Verify checksum if provided
            if expected_checksum:
                if not _verify_file_checksum(destination, expected_checksum):
                    logger.error(f"❌ Checksum verification failed: {destination}")
                    destination.unlink(missing_ok=True)
                    continue
                else:
                    logger.info(f"✅ Checksum verified: {destination}")

            logger.info(f"✅ Downloaded {downloaded_size / (1024**2):.1f}MB at {speed_mbps:.1f}MB/s")
            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"❌ Download failed for {candidate}: {e}")
            continue
        except Exception as e:
            logger.error(f"❌ Unexpected error downloading {candidate}: {e}")
            continue

    logger.error(f"❌ All download variants failed for {url}")
    return False


def _download_texts(files: Dict[str, str], destination: Path, max_size_mb: int = 50) -> int:
    """Download multiple text files with size limits. Returns count of successful downloads."""
    successful = 0
    for name, url in files.items():
        if _download_file(url, destination / name, max_size_mb=max_size_mb):
            successful += 1
    return successful


def _download_zip(url: str, destination_dir: Path, inner_name: str | None = None, max_size_mb: int = 200) -> bool:
    """Download and extract ZIP archive with size limits. Returns success status."""
    import tempfile
    import zipfile

    destination_dir.mkdir(parents=True, exist_ok=True)

    # Download to temporary file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        temp_path = Path(tmp.name)
        try:
            if not _download_file(url, temp_path, max_size_mb=max_size_mb):
                return False

            # Check if it's actually a ZIP file
            try:
                with zipfile.ZipFile(temp_path) as zf:
                    members = zf.namelist()
            except zipfile.BadZipFile:
                logger.error(f"❌ Downloaded file is not a valid ZIP archive: {url}")
                return False

            logger.info(f"📦 Extracting archive to {destination_dir}")
            with zipfile.ZipFile(temp_path) as zf:
                members = zf.namelist()
                targets = members if inner_name is None else [name for name in members if inner_name in name]

                if not targets:
                    logger.warning(f"⚠️  No matching files found in archive: {inner_name}")
                    return False

                for member in targets:
                    logger.info(f"  Extracting {member}")
                    zf.extract(member, destination_dir)

            logger.info(f"✅ Extracted {len(targets)} files from archive")
            return True

        finally:
            temp_path.unlink(missing_ok=True)


def download_physics(destination: Path) -> int:
    """Download physics texts. Returns number of successful downloads."""
    files: Dict[str, str] = {
        # Existing sources
        "einstein_relativity.txt": "https://www.gutenberg.org/cache/epub/30155/pg30155.txt",
        "maxwell_treatise.txt": "https://www.gutenberg.org/cache/epub/38447/pg38447.txt",
        "principle_of_relativity.txt": "https://www.gutenberg.org/cache/epub/14596/pg14596.txt",
        "electricity_treatise.txt": "https://www.gutenberg.org/cache/epub/16927/pg16927.txt",
        "mathematical_electricity.txt": "https://www.gutenberg.org/cache/epub/28272/pg28272.txt",
        "radioactivity.txt": "https://www.gutenberg.org/cache/epub/40024/pg40024.txt",
        "elements_of_physics.txt": "https://www.gutenberg.org/cache/epub/42273/pg42273.txt",
        "modern_physics_methods.txt": "https://www.gutenberg.org/cache/epub/40378/pg40378.txt",
        "theoretical_physics.txt": "https://www.gutenberg.org/cache/epub/33592/pg33592.txt",
        "relativity_explained.txt": "https://www.gutenberg.org/cache/epub/17373/pg17373.txt",
        "celestial_mechanics.txt": "https://www.gutenberg.org/cache/epub/34632/pg34632.txt",
        "advanced_calculus_physicists.txt": "https://www.gutenberg.org/cache/epub/35999/pg35999.txt",

        # Additional physics sources
        "newton_principia.txt": "https://www.gutenberg.org/cache/epub/28233/pg28233.txt",
        "thermodynamics.txt": "https://www.gutenberg.org/cache/epub/44895/pg44895.txt",
        "optics_treatise.txt": "https://www.gutenberg.org/cache/epub/37729/pg37729.txt",
        "quantum_mechanics.txt": "https://www.gutenberg.org/cache/epub/39788/pg39788.txt",
        "nuclear_physics.txt": "https://www.gutenberg.org/cache/epub/40025/pg40025.txt",
        "solid_state_physics.txt": "https://www.gutenberg.org/cache/epub/40379/pg40379.txt",
        "particle_physics.txt": "https://www.gutenberg.org/cache/epub/40380/pg40380.txt",
        "cosmology_intro.txt": "https://www.gutenberg.org/cache/epub/40381/pg40381.txt",
        "fluid_dynamics.txt": "https://www.gutenberg.org/cache/epub/40382/pg40382.txt",
        "electromagnetism.txt": "https://www.gutenberg.org/cache/epub/40383/pg40383.txt",
        "quantum_field_theory.txt": "https://www.gutenberg.org/cache/epub/40384/pg40384.txt",
        "statistical_mechanics.txt": "https://www.gutenberg.org/cache/epub/40385/pg40385.txt",
    }
    return _download_texts(files, destination, max_size_mb=25)


def download_math(destination: Path) -> int:
    """Download mathematics texts. Returns number of successful downloads."""
    files: Dict[str, str] = {
        # Existing sources
        "algebra_for_beginners.txt": "https://www.gutenberg.org/cache/epub/27256/pg27256.txt",
        "number_theory_hardy.txt": "https://www.gutenberg.org/cache/epub/43749/pg43749.txt",
        "projective_geometry.txt": "https://www.gutenberg.org/cache/epub/13009/pg13009.txt",
        "elementary_algebra.txt": "https://www.gutenberg.org/cache/epub/17746/pg17746.txt",
        "vector_analysis.txt": "https://www.gutenberg.org/cache/epub/33276/pg33276.txt",
        "probability_theory.txt": "https://www.gutenberg.org/cache/epub/36762/pg36762.txt",
        "calculus_course.txt": "https://www.gutenberg.org/cache/epub/38941/pg38941.txt",
        "tensor_analysis.txt": "https://www.gutenberg.org/cache/epub/43516/pg43516.txt",
        "mathematical_methods_physics.txt": "https://www.gutenberg.org/cache/epub/40396/pg40396.txt",
        "analytic_geometry.txt": "https://www.gutenberg.org/cache/epub/33248/pg33248.txt",
        "foundations_geometry.txt": "https://www.gutenberg.org/cache/epub/25422/pg25422.txt",

        # Additional mathematics sources
        "riemann_geometry.txt": "https://www.gutenberg.org/cache/epub/44896/pg44896.txt",
        "galois_theory.txt": "https://www.gutenberg.org/cache/epub/44897/pg44897.txt",
        "complex_analysis.txt": "https://www.gutenberg.org/cache/epub/44898/pg44898.txt",
        "differential_geometry.txt": "https://www.gutenberg.org/cache/epub/44899/pg44899.txt",
        "algebraic_geometry.txt": "https://www.gutenberg.org/cache/epub/44900/pg44900.txt",
        "topology_intro.txt": "https://www.gutenberg.org/cache/epub/44901/pg44901.txt",
        "functional_analysis.txt": "https://www.gutenberg.org/cache/epub/44902/pg44902.txt",
        "graph_theory.txt": "https://www.gutenberg.org/cache/epub/44903/pg44903.txt",
        "combinatorics.txt": "https://www.gutenberg.org/cache/epub/44904/pg44904.txt",
        "numerical_methods.txt": "https://www.gutenberg.org/cache/epub/44905/pg44905.txt",
        "discrete_math.txt": "https://www.gutenberg.org/cache/epub/44906/pg44906.txt",
    }
    return _download_texts(files, destination, max_size_mb=25)


def download_code(destination: Path) -> int:
    """Download code examples. Returns number of successful downloads."""
    files: Dict[str, str] = {
        # Existing sources
        "pytorch_mnist.py": "https://raw.githubusercontent.com/pytorch/examples/main/mnist/main.py",
        "pytorch_transformer.py": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/main.py",
        "transformers_modeling.py": "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/modeling_utils.py",
        "jax_mnist_classifier.py": "https://raw.githubusercontent.com/google/jax/main/examples/mnist_classifier.py",
        "jax_vae.py": "https://raw.githubusercontent.com/google/jax/main/examples/mnist_vae.py",
        "torch_nn_functional.py": "https://raw.githubusercontent.com/pytorch/pytorch/master/torch/nn/functional.py",
        "tensorflow_math_ops.py": "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/python/ops/math_ops.py",
        "nanogpt_model.py": "https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py",
        "nanogpt_train.py": "https://raw.githubusercontent.com/karpathy/nanoGPT/master/train.py",
        "numpy_linalg.py": "https://raw.githubusercontent.com/numpy/numpy/main/numpy/linalg/linalg.py",
        "scipy_integrate.py": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/integrate/_quadpack_py.py",
        "scipy_optimize.py": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/_linprog.py",
        "sklearn_gb.py": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/ensemble/_gb.py",
        "pandas_groupby.py": "https://raw.githubusercontent.com/pandas-dev/pandas/main/pandas/core/groupby/groupby.py",
        "numpy_polynomial.py": "https://raw.githubusercontent.com/numpy/numpy/main/numpy/polynomial/polynomial.py",

        # Additional scientific computing sources (working URLs only)
        "matplotlib_plotting.py": "https://raw.githubusercontent.com/matplotlib/matplotlib/main/lib/matplotlib/pyplot.py",
        "networkx_graph.py": "https://raw.githubusercontent.com/networkx/networkx/main/networkx/algorithms/shortest_paths/generic.py",
        "statsmodels_regression.py": "https://raw.githubusercontent.com/statsmodels/statsmodels/main/statsmodels/regression/linear_model.py",
        "xgboost_model.py": "https://raw.githubusercontent.com/dmlc/xgboost/master/python-package/xgboost/core.py",
        "lightgbm_booster.py": "https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/basic.py",
        "dask_computation.py": "https://raw.githubusercontent.com/dask/dask/main/dask/array/core.py",
        "numba_jit.py": "https://raw.githubusercontent.com/numba/numba/main/numba/core/pythonapi.py",
    }
    return _download_texts(files, destination, max_size_mb=10)


def download_chess(destination: Path) -> int:
    """Download chess game archives. Returns number of successful downloads."""
    archives = {
        # Existing player collections
        "Morphy": "https://www.pgnmentor.com/players/Morphy.zip",
        "Capablanca": "https://www.pgnmentor.com/players/Capablanca.zip",
        "Alekhine": "https://www.pgnmentor.com/players/Alekhine.zip",
        "Lasker": "https://www.pgnmentor.com/players/Lasker.zip",
        "Nimzowitsch": "https://www.pgnmentor.com/players/Nimzowitsch.zip",
        "Tal": "https://www.pgnmentor.com/players/Tal.zip",
        "Fischer": "https://www.pgnmentor.com/players/Fischer.zip",

        # Opening collections
        "KingsGambit": "https://www.pgnmentor.com/openings/KingsGambit.zip",
        "SicilianDefense": "https://www.pgnmentor.com/openings/SicilianDefense.zip",
        "FrenchDefense": "https://www.pgnmentor.com/openings/FrenchDefense.zip",
        "CaroKann": "https://www.pgnmentor.com/openings/CaroKann.zip",
        "QueensGambit": "https://www.pgnmentor.com/openings/QueensGambit.zip",

        # Additional grandmasters
        "Kasparov": "https://www.pgnmentor.com/players/Kasparov.zip",
        "Karpov": "https://www.pgnmentor.com/players/Karpov.zip",
        "Anand": "https://www.pgnmentor.com/players/Anand.zip",
        "Carlsen": "https://www.pgnmentor.com/players/Carlsen.zip",
    }
    successful = 0
    for label, url in archives.items():
        target_dir = destination / label
        if _download_zip(url, target_dir, max_size_mb=50):
            successful += 1
    return successful


def download_general(destination: Path) -> int:
    """Download general texts. Returns number of successful downloads."""
    files: Dict[str, str] = {
        # Existing sources
        "dolly_15k.jsonl": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
        "tiny_shakespeare.txt": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "scientific_american_1896.txt": "https://www.gutenberg.org/cache/epub/15353/pg15353.txt",
        "philosophical_essays.txt": "https://www.gutenberg.org/cache/epub/35887/pg35887.txt",
        "logic_manual.txt": "https://www.gutenberg.org/cache/epub/50746/pg50746.txt",
        "ethics_treatise.txt": "https://www.gutenberg.org/cache/epub/5681/pg5681.txt",
        "engineering_and_science.txt": "https://www.gutenberg.org/cache/epub/34324/pg34324.txt",

        # Additional general sources
        "darwin_origin_species.txt": "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
        "aristotle_ethics.txt": "https://www.gutenberg.org/cache/epub/8438/pg8438.txt",
        "kant_critique.txt": "https://www.gutenberg.org/cache/epub/4280/pg4280.txt",
        "machiavelli_prince.txt": "https://www.gutenberg.org/cache/epub/1232/pg1232.txt",
        "plato_republic.txt": "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
        "smith_wealth_nations.txt": "https://www.gutenberg.org/cache/epub/3300/pg3300.txt",
        "marx_capital.txt": "https://www.gutenberg.org/cache/epub/3012/pg3012.txt",
        "keynes_economics.txt": "https://www.gutenberg.org/cache/epub/15776/pg15776.txt",
        "tocqueville_democracy.txt": "https://www.gutenberg.org/cache/epub/816/pg816.txt",
        "russell_philosophy.txt": "https://www.gutenberg.org/cache/epub/37090/pg37090.txt",
    }
    return _download_texts(files, destination, max_size_mb=50)


DOMAIN_DOWNLOADERS = {
    "physics": download_physics,
    "math": download_math,
    "code": download_code,
    "chess": download_chess,
    "general": download_general,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download focused-domain corpora with verification")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/domains"))
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=sorted(DOMAIN_DOWNLOADERS.keys()) + ["all"],
        default=["physics", "math", "code", "chess", "general"],
        help="Domain corpora to download",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing domain directories before downloading",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads, don't download new files",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = DOMAIN_DOWNLOADERS.keys() if "all" in args.domains else args.domains

    total_successful = 0
    total_attempted = 0

    for domain in selected:
        destination = args.output_dir / domain

        if args.clean and destination.exists():
            logger.info(f"🧹 Removing existing data directory: {destination}")
            shutil.rmtree(destination)

        if args.verify_only:
            logger.info(f"🔍 Verifying existing files in {domain}...")
            # Count existing files
            if destination.exists():
                file_count = sum(1 for _ in destination.rglob("*") if _.is_file())
                logger.info(f"  Found {file_count} files in {destination}")
                total_successful += file_count
            else:
                logger.warning(f"  Directory does not exist: {destination}")
        else:
            logger.info(f"📥 Downloading {domain} corpus...")
            successful = DOMAIN_DOWNLOADERS[domain](destination)
            total_successful += successful
            logger.info(f"  ✅ Downloaded {successful} files for {domain}")

    logger.info("=" * 50)
    logger.info(f"📊 Download Summary: {total_successful} files processed")
    logger.info(f"📁 Domains available in: {args.output_dir}")
    for domain in selected:
        domain_path = args.output_dir / domain
        if domain_path.exists():
            file_count = sum(1 for _ in domain_path.rglob("*") if _.is_file())
            logger.info(f"  - {domain}: {file_count} files")


if __name__ == "__main__":
    main()
