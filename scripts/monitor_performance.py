#!/usr/bin/env python3
"""Monitor training performance."""

import argparse
import time
import psutil
import torch
import subprocess

def get_gpu_utilization():
    """Get GPU utilization percentage."""
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    
    return 0.0

def get_memory_usage():
    """Get memory usage statistics."""
    memory = psutil.virtual_memory()
    return {
        "total": memory.total,
        "available": memory.available,
        "used": memory.used,
        "percentage": memory.percent
    }

def get_gpu_memory():
    """Get GPU memory usage."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated()
        }
    return {}

def monitor_performance(interval: int = 60):
    """Monitor performance metrics."""
    
    while True:
        gpu_util = get_gpu_utilization()
        memory = get_memory_usage()
        gpu_memory = get_gpu_memory()
        
        print(f"GPU Utilization: {gpu_util}%")
        print(f"System Memory: {memory['percentage']}% used")
        
        if gpu_memory:
            print(f"GPU Memory: {gpu_memory['allocated'] / 1e9:.2f}GB allocated")
        
        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Monitor training performance")
    parser.add_argument("--interval", type=int, default=60, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    monitor_performance(args.interval)

if __name__ == "__main__":
    main()
