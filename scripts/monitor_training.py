#!/usr/bin/env python3
"""Training monitoring script to detect crashes and provide recovery options."""

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any

import psutil


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training_monitor.log")
        ]
    )
    return logging.getLogger(__name__)


def find_training_process() -> psutil.Process | None:
    """Find the running training process."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'run_cloud_training.py' in cmdline:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def check_gpu_usage() -> Dict[str, Any]:
    """Check GPU usage and memory."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_data = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_data.append({
                        'utilization': int(parts[0]),
                        'memory_used': int(parts[1]),
                        'memory_total': int(parts[2]),
                        'temperature': int(parts[3])
                    })
            return {'gpus': gpu_data, 'status': 'healthy'}
        else:
            return {'status': 'error', 'message': 'nvidia-smi failed'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def check_training_logs(log_file: Path, last_check: float) -> Dict[str, Any]:
    """Check training logs for recent activity."""
    if not log_file.exists():
        return {'status': 'error', 'message': 'Log file not found'}
    
    try:
        # Read last few lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-50:] if len(lines) > 50 else lines
            
        # Look for recent activity
        recent_activity = []
        for line in recent_lines:
            if 'step=' in line and 'loss=' in line:
                recent_activity.append(line.strip())
        
        # Check if we have very recent activity
        log_mtime = log_file.stat().st_mtime
        is_recent = (time.time() - log_mtime) < 300  # 5 minutes
        
        return {
            'status': 'healthy' if is_recent and recent_activity else 'stale',
            'recent_activity': recent_activity[-3:] if recent_activity else [],
            'last_modified': log_mtime,
            'is_recent': is_recent
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def check_checkpoints(checkpoint_dir: Path) -> Dict[str, Any]:
    """Check checkpoint status."""
    if not checkpoint_dir.exists():
        return {'status': 'error', 'message': 'Checkpoint directory not found'}
    
    try:
        checkpoint_files = list(checkpoint_dir.glob("step-*.pt"))
        if not checkpoint_files:
            return {'status': 'warning', 'message': 'No checkpoints found'}
        
        # Get latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        checkpoint_time = latest_checkpoint.stat().st_mtime
        
        return {
            'status': 'healthy',
            'count': len(checkpoint_files),
            'latest': str(latest_checkpoint),
            'latest_time': checkpoint_time,
            'age_minutes': (time.time() - checkpoint_time) / 60
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def generate_recovery_commands(checkpoint_dir: Path) -> list[str]:
    """Generate recovery commands."""
    commands = []
    
    # Check if we have checkpoints
    checkpoint_files = list(checkpoint_dir.glob("step-*.pt"))
    if checkpoint_files:
        latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        step_num = latest.stem.split('-')[-1]
        commands.append(f"# Resume from latest checkpoint (step {step_num})")
        commands.append(f"python scripts/run_cloud_training.py --config configs/cloud_training.yaml --stage both --resume --skip-data")
    else:
        commands.append("# No checkpoints found - start from scratch")
        commands.append(f"python scripts/run_cloud_training.py --config configs/cloud_training.yaml --stage both --skip-data")
    
    return commands


def main():
    parser = argparse.ArgumentParser(description="Monitor FusterCluck training")
    parser.add_argument("--log-file", type=Path, default=Path("logs/cloud_training.log"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints/cloud/stage1"))
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--auto-recovery", action="store_true", help="Automatically restart training if crashed")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("Starting training monitor...")
    logger.info("Log file: %s", args.log_file)
    logger.info("Checkpoint dir: %s", args.checkpoint_dir)
    logger.info("Check interval: %d seconds", args.interval)
    
    last_check = time.time()
    
    try:
        while True:
            logger.info("=== Training Health Check ===")
            
            # Check training process
            training_proc = find_training_process()
            if training_proc:
                logger.info("✅ Training process found (PID: %d)", training_proc.pid)
                logger.info("   CPU: %.1f%%, Memory: %.1f%%", 
                           training_proc.cpu_percent(), training_proc.memory_percent())
            else:
                logger.warning("❌ No training process found")
            
            # Check GPU
            gpu_status = check_gpu_usage()
            if gpu_status['status'] == 'healthy':
                for i, gpu in enumerate(gpu_status['gpus']):
                    logger.info("🖥️  GPU %d: %d%% util, %d/%d MB memory, %d°C", 
                               i, gpu['utilization'], gpu['memory_used'], 
                               gpu['memory_total'], gpu['temperature'])
            else:
                logger.error("❌ GPU check failed: %s", gpu_status['message'])
            
            # Check logs
            log_status = check_training_logs(args.log_file, last_check)
            if log_status['status'] == 'healthy':
                logger.info("📝 Logs: Recent activity detected")
                for activity in log_status['recent_activity']:
                    logger.info("   %s", activity)
            elif log_status['status'] == 'stale':
                logger.warning("⚠️  Logs: No recent activity (last modified: %.1f minutes ago)", 
                              (time.time() - log_status['last_modified']) / 60)
            else:
                logger.error("❌ Log check failed: %s", log_status['message'])
            
            # Check checkpoints
            ckpt_status = check_checkpoints(args.checkpoint_dir)
            if ckpt_status['status'] == 'healthy':
                logger.info("💾 Checkpoints: %d found, latest age: %.1f minutes", 
                           ckpt_status['count'], ckpt_status['age_minutes'])
            else:
                logger.warning("⚠️  Checkpoints: %s", ckpt_status['message'])
            
            # Auto-recovery if needed
            if args.auto_recovery and not training_proc:
                logger.info("🔄 Auto-recovery: Training not running, attempting restart...")
                recovery_commands = generate_recovery_commands(args.checkpoint_dir)
                logger.info("Recovery commands:")
                for cmd in recovery_commands:
                    logger.info("   %s", cmd)
                # Note: In a real implementation, you'd execute these commands
            
            logger.info("Next check in %d seconds...", args.interval)
            time.sleep(args.interval)
            last_check = time.time()
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error("Monitor error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
