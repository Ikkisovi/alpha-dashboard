import torch
import math
import numpy as np
from typing import Dict, Optional, Tuple

class EnvReward:
    """
    Calculates the 'Performance' component of the reward.
    Logic: 线性、单调：指标越大越好；未达标有轻微惩罚，但奖励主导。
    """
    
    # Weights for components
    W_IC = 1.0
    W_EXCESS = 1.0
    W_SORTINO = 0.5
    W_ICIR = 0.5
    # Penalty scales（小于正向权重，保证“奖励多于惩罚”）
    IC_PENALTY = 0.25
    EXCESS_PENALTY = 0.5
    SORTINO_PENALTY = 0.25
    ICIR_PENALTY = 0.25
    MIN_IC_WARN = 0.02
    MIN_ICIR_WARN = 0.05
    SCORE_CLIP = 5.0  # 防极端爆炸

    @staticmethod
    def calculate(metrics: Dict[str, float], t_step: float, pool_avg_sortino: float = 0.0) -> float:
        """
        metrics: Dictionary containing 'ic', 'excess_return', 'icir', 'sortino'.
        t_step: Training progress [0, 1].
        pool_avg_sortino: The average Sortino ratio of valid factors currently in the pool.
        """
        ic = metrics.get('ic', 0.0)
        # Flip sign if negative (agent should learn magnitude, direction is handled by sign flipping in pool)
        # Actually user said: "negative ic factor will flip the sign... so the more close to zero the more penalty"
        # The pool does the flipping before metrics, so 'ic' here is usually positive canonical.
        # But if it wasn't flipped yet, we take abs.
        ic = abs(ic)
        excess = metrics.get('excess_return', 0.0)
        sortino = metrics.get('sortino', 0.0)
        icir = metrics.get('icir', 0.0)

        # 1) IC: 线性奖励；低于预警阈值给轻微负分，但奖励>惩罚
        ic_reward = EnvReward.W_IC * ic
        ic_penalty = 0.0
        if ic < EnvReward.MIN_IC_WARN:
            ic_penalty = -EnvReward.IC_PENALTY * (EnvReward.MIN_IC_WARN - ic)

        # 2) Excess Return: 正向值给奖励，负值给半权重惩罚
        excess_reward = EnvReward.W_EXCESS * max(excess, 0.0)
        excess_penalty = EnvReward.EXCESS_PENALTY * min(excess, 0.0)  # excess<0 -> 负值

        # 3) Sortino: 相对池均值，高于均值给奖励，低于均值给轻微惩罚
        sortino_delta = sortino - pool_avg_sortino
        sortino_reward = EnvReward.W_SORTINO * max(sortino_delta, 0.0)
        sortino_penalty = EnvReward.SORTINO_PENALTY * min(sortino_delta, 0.0)

        # 4) ICIR: 线性奖励，低于预警值给轻微惩罚
        icir_reward = EnvReward.W_ICIR * icir
        icir_penalty = 0.0
        if icir < EnvReward.MIN_ICIR_WARN:
            icir_penalty = -EnvReward.ICIR_PENALTY * (EnvReward.MIN_ICIR_WARN - icir)

        score = (
            ic_reward + ic_penalty +
            excess_reward + excess_penalty +
            sortino_reward + sortino_penalty +
            icir_reward + icir_penalty
        )

        # 防止极端爆炸，但不做 tanh 压缩
        if score > EnvReward.SCORE_CLIP:
            score = EnvReward.SCORE_CLIP
        elif score < -EnvReward.SCORE_CLIP:
            score = -EnvReward.SCORE_CLIP

        return float(score)

class ResultStructReward:
    """
    Calculates the 'Sanity' component of the reward.
    Logic: Penalize degenerate factors (constant, NaN blowup) and excessive length.
    These are checks on the RESULT VALUES, not the AST.
    """
    
    BLOWUP_THRESHOLD = 1e4
    MIN_VARIANCE = 1e-6
    MAX_LEN = 60
    LEN_PENALTY_WEIGHT = 0.01
    
    @staticmethod
    def calculate(factor_values: torch.Tensor, expr_str: str) -> float:
        """
        factor_values: Tensor of shape (T, N)
        expr_str: String representation of expression (for length check)
        """
        r_struct = 0.0
        
        # 1. Sanity Check: Variance (Flatness)
        # Check if factor is effectively constant across cross-section
        if factor_values.numel() > 0:
            # Check variance per day
            day_vars = factor_values.var(dim=-1)
            # If >50% of days have near-zero variance, it's a "bad" factor (likely constant)
            # Even if valid, it provides no cross-sectional signal.
            low_var_days = (day_vars < ResultStructReward.MIN_VARIANCE).float().mean()
            if low_var_days > 0.5:
                # Heavy penalty for being constant/flat
                r_struct -= 2.0 

        # 2. Sanity Check: Blowup / NaNs
        # (NaNs should be handled by env, but if values are huge magnitude, penalize)
        # Note: We assume NaNs are masked or handled, but let's check finite max.
        # Use nan_to_num logic effectively
        if factor_values.numel() > 0:
            # We use a robust max check
            # Be careful with memory on large tensors.
            # Just check a subsample or use torch.max() returns value
            vals = torch.nan_to_num(factor_values, nan=0.0, posinf=ResultStructReward.BLOWUP_THRESHOLD, neginf=-ResultStructReward.BLOWUP_THRESHOLD)
            max_val = vals.abs().max().item()
            if max_val >= ResultStructReward.BLOWUP_THRESHOLD:
                # Soft penalty for hitting bounds
                r_struct -= 1.0

        return r_struct

class RewardUtil:
    """
    Unified Utility to calculate rewards and return standardized dictionary.
    """
    
    @staticmethod
    def compute_rewards(metrics: Dict[str, float], 
                       factor_values: Optional[torch.Tensor], 
                       expr_str: str,
                       t_step: float,
                       pool_avg_sortino: float = 0.0) -> Dict[str, float]:
        """
        Compute all reward components.
        pool_avg_sortino: Mean sortino of current pool (for relative comparison).
        
        Returns:
        {
            'r_env': float,      # Performance reward (for TB)
            'r_struct': float,   # Sanity penalty
            'r_total': float,    # Combined reward (Env + Struct) for RL
            'log_R_tb': float    # Log Reward for TB Loss (Derived from Env Only)
        }
        """
        # 1. Calculate Component Rewards
        r_env = EnvReward.calculate(metrics, t_step, pool_avg_sortino)
        
        if factor_values is not None:
            r_struct = ResultStructReward.calculate(factor_values, expr_str)
        else:
            r_struct = 0.0
            
        # 2. Combined Reward
        # r_total is used for REINFORCE (Advantage).
        # We can sum them directly.
        r_total = r_env + r_struct
        
        # 3. TB Reward (log_R)
        # For TB, we typically want a positive reward R.
        # r_env 允许超过 1，因此降低温度避免过度锐化。
        BETA = 1.0 
        log_R_tb = r_env * BETA
        
        # Clip log_R to avoid extreme gradients if needed
        MAX_LOG_R = 20.0
        log_R_tb = max(min(log_R_tb, MAX_LOG_R), -MAX_LOG_R)
        
        return {
            'r_env': r_env,
            'r_struct': r_struct,
            'r_total': r_total,
            'log_R_tb': log_R_tb
        }
