"""
Curriculum Scheduler for GFN Training.

Manages progressive difficulty (expression length) during training.
Separated from train_gfn.py to avoid import bloat.
"""
import numpy as np
from typing import Optional, List, Tuple


class CurriculumScheduler:
    """
    Progressive curriculum scheduler that increases max expression length
    as training progresses and performance improves.
    """
    
    def __init__(
        self,
        env,
        n_episodes: int,
        pool=None,
        min_reward_to_progress: float = 0.01,
        min_avg_ic_to_progress: float = 0.02,
    ):
        """
        Args:
            env: GFNEnvCore - environment to control max_len on
            n_episodes: Total training episodes for progress calculation
            pool: AlphaPoolGFN - optional, for performance-gated progression
            min_reward_to_progress: Min r_env to allow phase advancement
            min_avg_ic_to_progress: Min avg IC to allow phase advancement
        """
        self.env = env
        self.n_episodes = n_episodes
        self.pool = pool
        self.min_reward_to_progress = float(min_reward_to_progress)
        self.min_avg_ic_to_progress = float(min_avg_ic_to_progress)
        self.found_rewardable_in_phase = False
        
        # Phase configuration: (fraction_threshold, max_length)
        self.phases: List[Tuple[float, int]] = [
            (0.20, 8),   # First 20%: Short (len 8)
            (0.60, 12),  # Next 40%: Medium (len 12)
            (0.80, 16),  # Next 20%: Medium-Long (len 16)
            (1.00, 20)   # Last 20%: Full length (len 20)
        ]
        self.current_phase = 0
        self.episodes_in_phase = 0
        
        # Enforce initial phase constraint
        if self.env is not None:
            self.env.max_len = self.phases[0][1]
        
    def notify_reward(self, reward: float):
        """Called by pool.try_new_expr() when a factor is evaluated."""
        if reward >= self.min_reward_to_progress:
            self.found_rewardable_in_phase = True
        
    def step(self, episode: int):
        """Update curriculum state after each episode."""
        self.episodes_in_phase += 1
        progress = episode / self.n_episodes
        
        # Find applicable phase based on progress
        target_phase = 0
        for i, (frac, length) in enumerate(self.phases):
            if progress < frac:
                target_phase = i
                break
            target_phase = i
        
        # Gate: Require performance threshold before advancing
        if target_phase > self.current_phase:
            avg_ic = 0.0
            if self.pool is not None:
                if hasattr(self.pool, 'get_avg_ic'):
                    avg_ic = self.pool.get_avg_ic()
                elif hasattr(self.pool, 'single_ics') and self.pool.size > 0:
                    avg_ic = float(np.mean(np.abs(self.pool.single_ics[:self.pool.size])))
            
            can_progress = (avg_ic >= self.min_avg_ic_to_progress) or self.found_rewardable_in_phase
            
            if not can_progress:
                if self.episodes_in_phase % 100 == 0:
                    print(
                        f"[Curriculum] Locked in Phase {self.current_phase} "
                        f"(avg_ic={avg_ic:.4f} < {self.min_avg_ic_to_progress:.4f}, "
                        f"found_rewardable={self.found_rewardable_in_phase})"
                    )
                return

            # Phase transition
            self.current_phase = target_phase
            self.episodes_in_phase = 0
            self.found_rewardable_in_phase = False
            new_len = self.phases[target_phase][1]
            print(f"[Curriculum] 🚀 Promoting to Phase {target_phase}: Max Length = {new_len} (Avg IC {avg_ic:.4f})")
            
            if hasattr(self.env, 'max_len'):
                self.env.max_len = new_len
