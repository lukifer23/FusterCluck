"""General training utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThroughputTracker:
    """Tracks tokens processed per second over a moving window."""

    num_tokens: int = 0
    elapsed: float = 0.0

    def update(self, tokens: int, seconds: float) -> None:
        self.num_tokens += tokens
        self.elapsed += seconds

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed == 0:
            return 0.0
        return self.num_tokens / self.elapsed

    def reset(self) -> None:
        self.num_tokens = 0
        self.elapsed = 0.0


def format_timespan(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
