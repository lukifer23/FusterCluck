#!/usr/bin/env python3
"""Track training costs."""

import argparse
import time
import json
from pathlib import Path
from datetime import datetime

class CostTracker:
    def __init__(self, log_file: str = "logs/cost_tracking.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.start_time = time.time()
        self.cost_per_hour = 0.79  # RTX 4090 on RunPod
        
    def log_cost(self, stage: str, step: int = None):
        """Log current cost."""
        elapsed_hours = (time.time() - self.start_time) / 3600
        current_cost = elapsed_hours * self.cost_per_hour
        
        cost_data = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "step": step,
            "elapsed_hours": elapsed_hours,
            "current_cost": current_cost,
            "cost_per_hour": self.cost_per_hour
        }
        
        # Load existing data
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(cost_data)
        
        # Save updated data
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Cost tracking: {elapsed_hours:.2f}h elapsed, ${current_cost:.2f} spent")
        
    def get_total_cost(self):
        """Get total cost so far."""
        elapsed_hours = (time.time() - self.start_time) / 3600
        return elapsed_hours * self.cost_per_hour

def main():
    parser = argparse.ArgumentParser(description="Track training costs")
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--log-file", type=str, default="logs/cost_tracking.json")
    
    args = parser.parse_args()
    
    tracker = CostTracker(args.log_file)
    tracker.log_cost(args.stage, args.step)

if __name__ == "__main__":
    main()
