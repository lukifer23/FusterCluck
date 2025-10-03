#!/usr/bin/env python3
"""Monitor text-only training runs and surface recovery guidance."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import psutil

LOG_PATH = Path("logs/text_training_monitor.log")
TARGET_COMMANDS = {"run_text_pipeline.py", "run_stage0.py"}


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH, mode="a")],
    )
    return logging.getLogger(__name__)


def _cmdline_matches(cmdline: Iterable[str]) -> bool:
    for item in cmdline:
        for target in TARGET_COMMANDS:
            if target in item:
                return True
    return False


def find_training_process() -> psutil.Process | None:
    """Locate an active training process."""
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if _cmdline_matches(cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def check_accelerator_usage() -> Dict[str, Any]:
    """Report accelerator utilisation for MPS or CUDA."""
    try:
        import torch

        if torch.backends.mps.is_available():
            allocated = torch.mps.driver_allocated_memory()
            limit = torch.mps.driver_available_memory()
            return {
                "status": "healthy",
                "backend": "mps",
                "allocated_mb": allocated / (1024 ** 2),
                "available_mb": limit / (1024 ** 2),
            }
        if torch.cuda.is_available():
            stats = torch.cuda.memory_stats()
            return {
                "status": "healthy",
                "backend": "cuda",
                "allocated_mb": stats.get("active_bytes.all.current", 0) / (1024 ** 2),
                "reserved_mb": stats.get("reserved_bytes.all.current", 0) / (1024 ** 2),
            }
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            util, used, total, temp = (int(x) for x in line.split(", "))
            return {
                "status": "healthy",
                "backend": "nvidia-smi",
                "util_percent": util,
                "memory_used_mb": used,
                "memory_total_mb": total,
                "temperature_c": temp,
            }
        return {"status": "error", "message": "nvidia-smi failed"}

    return {"status": "info", "message": "No accelerator detected"}


def check_training_logs(log_file: Path) -> Dict[str, Any]:
    if not log_file.exists():
        return {"status": "error", "message": "Log file not found"}

    lines = log_file.read_text(errors="ignore").splitlines()
    tail = lines[-60:]
    recent = [line for line in tail if "step=" in line and "loss=" in line]
    mtime = log_file.stat().st_mtime
    fresh = (time.time() - mtime) < 300
    return {
        "status": "healthy" if fresh and recent else "stale",
        "recent": recent[-3:],
        "last_modified": mtime,
    }


def check_checkpoints(checkpoint_dir: Path) -> Dict[str, Any]:
    if not checkpoint_dir.exists():
        return {"status": "error", "message": "Checkpoint directory missing"}

    checkpoints = sorted(checkpoint_dir.glob("step-*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        return {"status": "warning", "message": "No checkpoints yet"}

    latest = checkpoints[-1]
    age_min = (time.time() - latest.stat().st_mtime) / 60
    return {
        "status": "healthy",
        "count": len(checkpoints),
        "latest": latest,
        "age_minutes": age_min,
    }


def recovery_commands(config_path: Path, stage: str | None) -> list[str]:
    cmd = ["python", "scripts/run_text_pipeline.py", str(config_path)]
    if stage:
        cmd.extend(["--stages", stage])
    return ["# Resume pipeline", " ".join(cmd)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor FusterCluck text training")
    parser.add_argument("--log-file", type=Path, default=Path("logs/text_pipeline.log"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/checkpoints/pipeline/main_pretrain_12k"),
    )
    parser.add_argument("--interval", type=int, default=90, help="Polling interval in seconds")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/text_pipeline.yaml"),
        help="Pipeline config used for recovery commands",
    )
    parser.add_argument("--stage", type=str, default="main_pretrain_12k", help="Stage name to resume")

    args = parser.parse_args()
    logger = setup_logging()
    logger.info("Monitoring started; log=%s checkpoint_dir=%s", args.log_file, args.checkpoint_dir)

    try:
        while True:
            proc = find_training_process()
            if proc:
                logger.info("✅ Training PID %d | CPU %.1f%% | RSS %.2f GB", proc.pid, proc.cpu_percent(), proc.memory_info().rss / (1024 ** 3))
            else:
                logger.warning("❌ Training process not found")

            accel = check_accelerator_usage()
            if accel["status"] == "healthy":
                logger.info("⚙️  Accelerator: %s", accel)
            else:
                logger.info("⚙️  Accelerator info: %s", accel.get("message", accel["status"]))

            log_status = check_training_logs(args.log_file)
            if log_status["status"] == "healthy":
                for line in log_status["recent"]:
                    logger.info("📝 %s", line)
            elif log_status["status"] == "stale":
                minutes = (time.time() - log_status["last_modified"]) / 60
                logger.warning("⚠️  Log inactive for %.1f minutes", minutes)
            else:
                logger.error("Log check failed: %s", log_status["message"])

            ckpt_status = check_checkpoints(args.checkpoint_dir)
            if ckpt_status["status"] == "healthy":
                logger.info("💾 %d checkpoints; latest age %.1f min", ckpt_status["count"], ckpt_status["age_minutes"])
            else:
                logger.warning("💾 %s", ckpt_status["message"])

            if not proc:
                logger.info("Suggested recovery:")
                for cmd in recovery_commands(args.config, args.stage):
                    logger.info("   %s", cmd)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Monitoring error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
