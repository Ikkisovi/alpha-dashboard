"""
Stratified Replay Buffer for Self-Improving GFlowNet Training.

Three-tier buffer system:
- B_high: Elite samples (top q% by reward) - learn what works
- B_mid: Medium reward samples - learn the boundary
- B_div: High diversity samples - prevent mode collapse
"""

import torch
import random
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from collections import deque


@dataclass
class ReplayEntry:
    """Single trajectory entry for replay."""
    states_tensor: torch.Tensor  # (max_len+1, state_dim)
    actions_tensor: torch.Tensor  # (max_len,)
    reward: float
    log_reward_tb: float = 0.0 # Target Log Reward for TB
    log_probs: Optional[torch.Tensor] = None  # (max_len,) for importance sampling
    embedding: Optional[torch.Tensor] = None  # For diversity computation
    when_done: int = 0  # Termination step

    def to_device(self, device: torch.device) -> 'ReplayEntry':
        """Move tensors to device."""
        return ReplayEntry(
            states_tensor=self.states_tensor.to(device),
            actions_tensor=self.actions_tensor.to(device),
            reward=self.reward,
            log_reward_tb=self.log_reward_tb,
            log_probs=self.log_probs.to(device) if self.log_probs is not None else None,
            embedding=self.embedding.to(device) if self.embedding is not None else None,
            when_done=self.when_done,
        )


class StratifiedReplayBuffer:
    """
    Three-tier stratified replay buffer.

    Maintains three separate buffers:
    1. B_high: Top percentile by reward (elite samples)
    2. B_mid: Medium reward samples
    3. B_div: High diversity samples (regardless of reward)

    This enables learning from:
    - Success patterns (B_high)
    - Boundary cases (B_mid)
    - Novel structures (B_div)
    """

    def __init__(
        self,
        capacity_high: int = 500,
        capacity_mid: int = 1000,
        capacity_div: int = 500,
        high_percentile: float = 0.05,  # Top 5% go to B_high
        diversity_threshold: float = 0.7,  # Min embedding distance for B_div
        device: torch.device = None,
    ):
        """
        Initialize stratified replay buffer.

        Args:
            capacity_high: Max size of elite buffer
            capacity_mid: Max size of medium buffer
            capacity_div: Max size of diversity buffer
            high_percentile: Percentile threshold for elite (0.05 = top 5%)
            diversity_threshold: Min cosine distance to qualify as diverse
            device: Torch device for tensors
        """
        self.capacity_high = capacity_high
        self.capacity_mid = capacity_mid
        self.capacity_div = capacity_div
        self.high_percentile = high_percentile
        self.diversity_threshold = diversity_threshold
        self.device = device or torch.device('cpu')

        # Three buffers (deques for efficient FIFO eviction)
        self.high_buffer: deque = deque(maxlen=capacity_high)
        self.mid_buffer: deque = deque(maxlen=capacity_mid)
        self.div_buffer: deque = deque(maxlen=capacity_div)

        # Track reward statistics for dynamic percentile computation
        self.reward_history: deque = deque(maxlen=10000)
        self.high_threshold: float = float('inf')  # Will be updated dynamically

        # Embeddings for diversity buffer (for distance computation)
        self.div_embeddings: List[torch.Tensor] = []

    def __len__(self) -> int:
        """Total number of entries across all buffers."""
        return len(self.high_buffer) + len(self.mid_buffer) + len(self.div_buffer)

    @property
    def size_high(self) -> int:
        return len(self.high_buffer)

    @property
    def size_mid(self) -> int:
        return len(self.mid_buffer)

    @property
    def size_div(self) -> int:
        return len(self.div_buffer)

    def _update_high_threshold(self):
        """Update the reward threshold for high-tier based on history."""
        if len(self.reward_history) < 100:
            # Not enough data, use a reasonable default
            self.high_threshold = 0.5
        else:
            rewards = list(self.reward_history)
            self.high_threshold = np.percentile(rewards, (1 - self.high_percentile) * 100)

    def _compute_diversity(self, embedding: torch.Tensor) -> float:
        """
        Compute minimum cosine distance to existing diversity buffer embeddings.

        Returns 1.0 if buffer is empty (maximally diverse).
        """
        if len(self.div_embeddings) == 0:
            return 1.0

        # Stack existing embeddings
        existing = torch.stack(self.div_embeddings).to(embedding.device)

        # Normalize for cosine similarity
        embedding_norm = embedding / (embedding.norm() + 1e-8)
        existing_norm = existing / (existing.norm(dim=-1, keepdim=True) + 1e-8)

        # Cosine similarity
        similarities = torch.matmul(existing_norm, embedding_norm)

        # Convert to distance (1 - similarity)
        min_distance = 1.0 - similarities.max().item()

        return min_distance

    def add(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        reward: float,
        log_reward_tb: float = 0.0,
        log_probs: Optional[torch.Tensor] = None,
        embedding: Optional[torch.Tensor] = None,
        when_done: int = 0,
    ):
        """
        Add a single trajectory to the appropriate buffer.

        Stratification logic:
        1. If reward >= high_threshold → B_high
        2. If embedding is diverse enough → B_div (regardless of reward)
        3. Otherwise → B_mid

        Args:
            states_tensor: (max_len+1, state_dim) state sequence
            actions_tensor: (max_len,) action sequence
            reward: Final reward
            log_probs: (max_len,) log probabilities for IS correction
            embedding: State embedding for diversity computation
            when_done: Termination timestep
        """
        # Update reward history and threshold
        self.reward_history.append(reward)
        self._update_high_threshold()

        entry = ReplayEntry(
            states_tensor=states_tensor.detach().cpu(),
            actions_tensor=actions_tensor.detach().cpu(),
            reward=reward,
            log_reward_tb=log_reward_tb,
            log_probs=log_probs.detach().cpu() if log_probs is not None else None,
            embedding=embedding.detach().cpu() if embedding is not None else None,
            when_done=when_done,
        )

        # Stratification logic
        if reward >= self.high_threshold:
            # Elite sample
            self.high_buffer.append(entry)
        elif embedding is not None:
            # Check diversity
            diversity = self._compute_diversity(embedding)
            if diversity >= self.diversity_threshold:
                # Diverse sample
                self.div_buffer.append(entry)
                self.div_embeddings.append(embedding.cpu())
                # Keep embeddings list in sync with buffer
                if len(self.div_embeddings) > self.capacity_div:
                    self.div_embeddings.pop(0)
            else:
                # Medium sample
                self.mid_buffer.append(entry)
        else:
            # No embedding, just add to mid
            self.mid_buffer.append(entry)

    def add_batch(
        self,
        states_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        rewards: torch.Tensor,
        log_rewards_tb: Optional[torch.Tensor] = None,
        log_probs_batch: Optional[torch.Tensor] = None,
        embeddings_batch: Optional[torch.Tensor] = None,
        when_done_batch: Optional[torch.Tensor] = None,
    ):
        """
        Add a batch of trajectories.

        Args:
            states_batch: (batch, max_len+1, state_dim)
            actions_batch: (batch, max_len)
            rewards: (batch,)
            log_rewards_tb: (batch,) or None
            log_probs_batch: (batch, max_len) or None
            embeddings_batch: (batch, hidden_dim) or None
            when_done_batch: (batch,) or None
        """
        batch_size = rewards.shape[0]

        for i in range(batch_size):
            self.add(
                states_tensor=states_batch[i] if states_batch.dim() > 2 else states_batch,
                actions_tensor=actions_batch[i] if actions_batch.dim() > 1 else actions_batch,
                reward=rewards[i].item(),
                log_reward_tb=log_rewards_tb[i].item() if log_rewards_tb is not None else 0.0,
                log_probs=log_probs_batch[i] if log_probs_batch is not None else None,
                embedding=embeddings_batch[i] if embeddings_batch is not None else None,
                when_done=when_done_batch[i].item() if when_done_batch is not None else 0,
            )

    def sample(
        self,
        n_high: int = 0,
        n_mid: int = 0,
        n_div: int = 0,
    ) -> Tuple[List[ReplayEntry], List[ReplayEntry], List[ReplayEntry]]:
        """
        Sample from each buffer.

        Args:
            n_high: Number of samples from B_high
            n_mid: Number of samples from B_mid
            n_div: Number of samples from B_div

        Returns:
            Tuple of (high_samples, mid_samples, div_samples)
        """
        # Sample from each buffer (with replacement if needed)
        high_samples = []
        mid_samples = []
        div_samples = []

        if n_high > 0 and len(self.high_buffer) > 0:
            k = min(n_high, len(self.high_buffer))
            indices = random.sample(range(len(self.high_buffer)), k)
            high_samples = [self.high_buffer[i] for i in indices]

        if n_mid > 0 and len(self.mid_buffer) > 0:
            k = min(n_mid, len(self.mid_buffer))
            indices = random.sample(range(len(self.mid_buffer)), k)
            mid_samples = [self.mid_buffer[i] for i in indices]

        if n_div > 0 and len(self.div_buffer) > 0:
            k = min(n_div, len(self.div_buffer))
            indices = random.sample(range(len(self.div_buffer)), k)
            div_samples = [self.div_buffer[i] for i in indices]

        return high_samples, mid_samples, div_samples

    def sample_mixed(
        self,
        total_samples: int,
        high_ratio: float = 0.5,
        mid_ratio: float = 0.3,
        div_ratio: float = 0.2,
    ) -> List[ReplayEntry]:
        """
        Sample a mixed batch from all buffers according to ratios.

        Args:
            total_samples: Total number of samples
            high_ratio: Fraction from B_high
            mid_ratio: Fraction from B_mid
            div_ratio: Fraction from B_div

        Returns:
            Combined list of samples
        """
        n_high = int(total_samples * high_ratio)
        n_mid = int(total_samples * mid_ratio)
        n_div = total_samples - n_high - n_mid

        high_samples, mid_samples, div_samples = self.sample(n_high, n_mid, n_div)

        # Combine and shuffle
        all_samples = high_samples + mid_samples + div_samples
        random.shuffle(all_samples)

        return all_samples

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics for logging."""
        return {
            'size_high': self.size_high,
            'size_mid': self.size_mid,
            'size_div': self.size_div,
            'total_size': len(self),
            'high_threshold': self.high_threshold,
            'avg_reward_high': np.mean([e.reward for e in self.high_buffer]) if self.high_buffer else 0,
            'avg_reward_mid': np.mean([e.reward for e in self.mid_buffer]) if self.mid_buffer else 0,
            'avg_reward_div': np.mean([e.reward for e in self.div_buffer]) if self.div_buffer else 0,
        }

    def clear(self):
        """Clear all buffers."""
        self.high_buffer.clear()
        self.mid_buffer.clear()
        self.div_buffer.clear()
        self.div_embeddings.clear()
        self.reward_history.clear()
        self.high_threshold = float('inf')
