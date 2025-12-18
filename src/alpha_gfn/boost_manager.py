"""
Unified management of all reward boost and guidance mechanisms for GFlowNet training.

This module consolidates:
1. Having injection (stagnation trigger)
2. Dynamic macro updates (periodic)
3. Imitation learning (periodic when pool healthy)
4. Weight scheduling (SSL/novelty decay)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .config import MACRO_CONFIG


@dataclass
class BoostConfig:
    """Configuration for boost mechanisms."""
    # Having injection
    having_check_interval: int = 100
    having_top_k: int = 5

    # Dynamic macro updates
    macro_update_freq: int = 500
    macro_diversity_threshold: float = 0.7
    n_dynamic_macros: int = 20

    # Imitation learning
    imitation_interval: int = 1000
    imitation_n_steps: int = 50
    imitation_batch_size: int = 16
    imitation_min_pool_ratio: float = 0.5  # Pool must be >= this ratio of capacity

    @classmethod
    def from_dict(cls, d: dict) -> 'BoostConfig':
        """Create config from dictionary, using defaults for missing keys."""
        return cls(
            having_check_interval=d.get('having_check_interval', 100),
            having_top_k=d.get('having_top_k', 5),
            macro_update_freq=d.get('dynamic_update_freq', 500),
            macro_diversity_threshold=d.get('macro_diversity_threshold', 0.7),
            n_dynamic_macros=d.get('n_dynamic_macros', 20),
            imitation_interval=d.get('imitation_interval', 1000),
            imitation_n_steps=d.get('imitation_n_steps', 50),
            imitation_batch_size=d.get('imitation_batch_size', 16),
            imitation_min_pool_ratio=d.get('imitation_min_pool_ratio', 0.5),
        )


class BoostManager:
    """
    Unified management of all guidance mechanisms for GFlowNet training.

    Coordinates:
    1. Having injection (when pool stagnates)
    2. Dynamic macro updates (periodic refresh from pool)
    3. Imitation learning (periodic behavioral cloning on pool winners)

    Usage:
        boost_manager = BoostManager(pool, env, pf, optimizer, config)
        for episode in range(n_episodes):
            # ... training step ...
            applied = boost_manager.step(episode, pool._current_scores())
            if applied:
                print(f"Applied boosts: {applied}")
    """

    def __init__(self, pool, env, pf, optimizer, config: BoostConfig = None):
        """
        Initialize the boost manager.

        Args:
            pool: AlphaPoolGFN instance
            env: GFNEnvCore instance
            pf: Forward policy estimator
            optimizer: Optimizer for the policy network
            config: BoostConfig instance (uses defaults if None)
        """
        self.pool = pool
        self.env = env
        self.pf = pf
        self.optimizer = optimizer
        self.config = config or BoostConfig.from_dict(MACRO_CONFIG)

        # State tracking for Having injection
        self.last_injection_episode = -1
        self.best_pool_score = None

        # State tracking for macro updates
        self.last_macro_update = -1

        # State tracking for imitation learning
        self.last_imitation_episode = -1

    def step(self, episode: int, current_scores: np.ndarray = None) -> List[str]:
        """
        Check and apply all boost mechanisms for the current episode.

        Args:
            episode: Current training episode number
            current_scores: Optional pre-computed pool scores

        Returns:
            List of applied boost names (e.g., ['having', 'macros', 'imitation:0.0234'])
        """
        if current_scores is None:
            current_scores = self.pool._current_scores() if self.pool.size > 0 else np.array([])

        applied = []

        # 1. Having injection (pool full + stagnant)
        if self._should_inject_having(episode, current_scores):
            self._apply_having_injection(current_scores)
            applied.append('having')

        # 2. Dynamic macro update (periodic)
        if self._should_update_macros(episode):
            n_updated = self._apply_macro_update()
            applied.append(f'macros:{n_updated}')

        # 3. Imitation learning (periodic + pool healthy)
        if self._should_run_imitation(episode):
            loss = self._apply_imitation()
            applied.append(f'imitation:{loss:.4f}')

        return applied

    # ==================== Having Injection ====================

    def _should_inject_having(self, episode: int, current_scores: np.ndarray) -> bool:
        """Check if Having injection should be triggered."""
        # Pool must be full
        if self.pool.size < self.pool.capacity:
            return False

        # Check interval since last injection
        if (episode - self.last_injection_episode) < self.config.having_check_interval:
            return False

        # Check for stagnation (no improvement)
        if current_scores.size == 0:
            return False

        current_best = float(np.max(current_scores))

        if self.best_pool_score is not None:
            if current_best > self.best_pool_score + 1e-6:
                # Improvement detected - update best and don't inject
                self.best_pool_score = current_best
                return False

        return True

    def _apply_having_injection(self, current_scores: np.ndarray):
        """Apply Having injection to top-K pool factors."""
        from seed_pool import add_having_variants

        # Select top-K factors
        top_indices = np.argsort(-current_scores)[:self.config.having_top_k]

        # Apply Having variants
        add_having_variants(
            self.pool,
            max_exprs=len(top_indices),
            base_indices=top_indices.tolist()
        )

        # Update state
        self.last_injection_episode = self.last_injection_episode  # Will be updated by caller
        # Refresh best score after injection
        refreshed_scores = self.pool._current_scores() if self.pool.size > 0 else np.array([])
        if refreshed_scores.size > 0:
            self.best_pool_score = float(np.max(refreshed_scores))

    def update_injection_episode(self, episode: int):
        """Update the last injection episode (called after injection)."""
        self.last_injection_episode = episode

    # ==================== Dynamic Macro Updates ====================

    def _should_update_macros(self, episode: int) -> bool:
        """Check if dynamic macros should be updated."""
        if self.config.n_dynamic_macros == 0:
            return False
        if self.pool.size == 0:
            return False
        if episode == 0:
            return False
        return (episode - self.last_macro_update) >= self.config.macro_update_freq

    def _apply_macro_update(self) -> int:
        """Update dynamic macros from pool's top factors."""
        from seed_pool import get_diverse_macros_from_pool

        # Get diverse expressions from pool
        diverse_exprs = get_diverse_macros_from_pool(
            self.pool,
            top_k=self.config.n_dynamic_macros * 2,  # Request more for diversity filtering
            correlation_threshold=self.config.macro_diversity_threshold
        )

        if diverse_exprs:
            self.env.update_dynamic_macros(diverse_exprs[:self.config.n_dynamic_macros])

        self.last_macro_update = self.last_macro_update  # Will be updated by caller
        return len(diverse_exprs[:self.config.n_dynamic_macros]) if diverse_exprs else 0

    def update_macro_episode(self, episode: int):
        """Update the last macro update episode (called after update)."""
        self.last_macro_update = episode

    # ==================== Imitation Learning ====================

    def _should_run_imitation(self, episode: int) -> bool:
        """Check if imitation learning should run."""
        if episode == 0:
            return False
        if self.pool.size == 0:
            return False

        # Check interval
        if (episode - self.last_imitation_episode) < self.config.imitation_interval:
            return False

        # Pool must be sufficiently full
        min_size = int(self.pool.capacity * self.config.imitation_min_pool_ratio)
        if self.pool.size < min_size:
            return False

        return True

    def _apply_imitation(self) -> float:
        """Run imitation learning on pool expressions."""
        loss = self._run_pool_imitation(
            n_steps=self.config.imitation_n_steps,
            batch_size=self.config.imitation_batch_size
        )
        self.last_imitation_episode = self.last_imitation_episode  # Will be updated by caller
        return loss

    def update_imitation_episode(self, episode: int):
        """Update the last imitation episode (called after imitation)."""
        self.last_imitation_episode = episode

    def _run_pool_imitation(self, n_steps: int = 50, batch_size: int = 16) -> float:
        """
        Run imitation learning on the current pool's accepted expressions.

        Teaches the policy network to reproduce token sequences from successful factors.

        Args:
            n_steps: Number of gradient steps
            batch_size: Batch size for sampling (state, action) pairs

        Returns:
            Average loss over training steps
        """
        from seed_pool import expr_to_action_sequence
        from alphagen.data.tokens import BEG_TOKEN
        from .config import MAX_EXPR_LENGTH

        if self.pool.size == 0:
            return 0.0

        # Convert pool expressions to (state, action) pairs
        all_pairs = []
        for i in range(self.pool.size):
            expr = self.pool.exprs[i]
            if expr is None:
                continue

            action_seq = expr_to_action_sequence(expr, self.env)
            if action_seq is None:
                continue

            # Build trajectory: series of (state, action) pairs
            state = torch.full((MAX_EXPR_LENGTH,), -1, dtype=torch.long, device=self.env.device)
            state[0] = self.env.token_to_id_map[BEG_TOKEN]

            for j, action_id in enumerate(action_seq):
                all_pairs.append((state.clone(), action_id))
                # Update state for next step (except for terminal SEP action)
                if j + 1 < len(action_seq) and j + 1 < MAX_EXPR_LENGTH:
                    state[j + 1] = action_id

        if len(all_pairs) < batch_size:
            print(f"[Imitation] Not enough pairs ({len(all_pairs)}) for batch size {batch_size}")
            return 0.0

        # Train policy on sampled batches
        total_loss = 0.0
        for step in range(n_steps):
            # Sample batch
            indices = np.random.choice(len(all_pairs), size=batch_size, replace=False)
            batch_states = torch.stack([all_pairs[i][0] for i in indices]).to(self.env.device)
            batch_actions = torch.tensor([all_pairs[i][1] for i in indices], device=self.env.device)

            # Forward pass through policy
            # Create DiscreteStates wrapper for the policy
            from gfn.states import DiscreteStates
            states = DiscreteStates(
                tensor=batch_states,
                forward_masks=torch.ones(batch_size, self.env.n_actions, dtype=torch.bool, device=self.env.device),
                backward_masks=torch.ones(batch_size, self.env.n_actions, dtype=torch.bool, device=self.env.device)
            )

            # Get policy logits
            policy_output = self.pf(states)
            logits = policy_output.logits

            # Cross-entropy loss
            loss = F.cross_entropy(logits, batch_actions)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / n_steps
        print(f"[Imitation] Completed {n_steps} steps, avg loss: {avg_loss:.4f}")
        return avg_loss

    # ==================== Utility Methods ====================

    def get_stats(self) -> dict:
        """Get current boost manager statistics."""
        return {
            'last_injection_episode': self.last_injection_episode,
            'best_pool_score': self.best_pool_score,
            'last_macro_update': self.last_macro_update,
            'last_imitation_episode': self.last_imitation_episode,
            'pool_size': self.pool.size,
            'pool_capacity': self.pool.capacity,
        }

    def reset_state(self):
        """Reset all state tracking variables."""
        self.last_injection_episode = -1
        self.best_pool_score = None
        self.last_macro_update = -1
        self.last_imitation_episode = -1
