"""
ReplayBuffer — experience replay for online learning.

Stores (user, stock, context, reward) tuples in a ring buffer.
Supports uniform and prioritized sampling (PER).
Prioritized sampling ensures recent high-reward/high-error experiences
are replayed more often, stabilizing online learning.
"""
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque

from stock_recommender.config import CONFIG


@dataclass
class ReplayEvent:
    """A single training example from user interaction."""
    user_id: int
    stock_id: int
    user_features: np.ndarray            # profile feature vector
    stock_features: np.ndarray           # technical indicator snapshot
    reward: float                        # signal strength (can be updated later)
    is_positive: bool                    # positive interaction (liked) or negative
    timestamp: float = field(default_factory=time.time)
    priority: float = 1.0               # for prioritized replay


class ReplayBuffer:
    """
    Fixed-capacity ring buffer for experience replay.

    Supports:
      • Uniform sampling (baseline)
      • Prioritized Experience Replay (PER) — samples high-priority events more often
        Priority is proportional to |TD error| or reward magnitude
    """

    def __init__(
        self,
        capacity: int = CONFIG.training.replay_buffer_capacity,
        alpha: float = 0.6,    # PER exponent (0 = uniform, 1 = fully prioritized)
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_annealing_steps: int = 100_000,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_step = (beta_end - beta_start) / max(beta_annealing_steps, 1)

        self._buffer: List[Optional[ReplayEvent]] = [None] * capacity
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._idx = 0          # next write position
        self._size = 0         # current fill level
        self._max_priority = 1.0

    def push(self, event: ReplayEvent) -> None:
        """Add an event with maximum priority (ensures it gets sampled at least once)."""
        event.priority = self._max_priority
        self._buffer[self._idx] = event
        self._priorities[self._idx] = self._max_priority
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push_batch(self, events: List[ReplayEvent]) -> None:
        for e in events:
            self.push(e)

    def sample(
        self, batch_size: int, strategy: str = "prioritized"
    ) -> Tuple[List[ReplayEvent], np.ndarray, np.ndarray]:
        """
        Sample a batch of events.

        Returns:
            events       : list of ReplayEvent
            indices      : buffer indices (needed to update priorities)
            is_weights   : importance sampling weights (for prioritized replay)
        """
        n = min(batch_size, self._size)
        if n == 0:
            return [], np.array([]), np.array([])

        active_priorities = self._priorities[: self._size]

        if strategy == "prioritized":
            probs = (active_priorities ** self.alpha)
            probs /= probs.sum()
            indices = np.random.choice(self._size, size=n, replace=False, p=probs)
            # Importance sampling weights to correct for non-uniform sampling
            is_weights = (self._size * probs[indices]) ** (-self.beta)
            is_weights /= is_weights.max()
            self.beta = min(self.beta + self.beta_step, self.beta_end)
        else:   # uniform
            indices = np.random.choice(self._size, size=n, replace=False)
            is_weights = np.ones(n, dtype=np.float32)

        events = [self._buffer[i] for i in indices]
        return events, indices, is_weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities after computing TD errors during a training step."""
        new_prios = (np.abs(td_errors) + 1e-6).astype(np.float32)
        self._priorities[indices] = new_prios
        self._max_priority = float(max(self._max_priority, new_prios.max()))

    def update_reward(self, event_idx: int, reward: float) -> None:
        """Update the reward of a stored event once it's been attributed (delayed)."""
        if self._buffer[event_idx] is not None:
            self._buffer[event_idx].reward = reward
            # Boost priority of events with strong reward signal
            self._priorities[event_idx] = max(abs(reward), 0.1)

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size >= CONFIG.training.replay_min_fill

    def __len__(self) -> int:
        return self._size

    def recent_reward_stats(self, n: int = 100) -> dict:
        """Diagnostic — average reward of the N most recently added events."""
        recent_idx = [
            (self._idx - 1 - i) % self.capacity
            for i in range(min(n, self._size))
        ]
        rewards = [self._buffer[i].reward for i in recent_idx if self._buffer[i] is not None]
        if not rewards:
            return {"n": 0, "mean_reward": 0.0, "positive_rate": 0.0}
        return {
            "n": len(rewards),
            "mean_reward": float(np.mean(rewards)),
            "positive_rate": float(np.mean([1 if r > 0 else 0 for r in rewards])),
        }
