"""
SmartTrainer: Orchestrator for Self-Improving GFlowNet Training.

Implements the "越挖越聪明" (gets smarter the more it mines) training paradigm:
1. On-policy sampling with current policy
2. Off-policy replay from stratified buffers
3. Combined loss (Policy + Value + Flow)
4. Curriculum learning for progressive difficulty
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from gfn.containers import Trajectories
from gfn.samplers import Sampler

from .stratified_replay import StratifiedReplayBuffer, ReplayEntry
from .combined_loss import CombinedLoss


class SmartTrainer:
    """
    Smart Trainer for self-improving factor mining.

    Key features:
    1. Hybrid on-policy/off-policy training
    2. Stratified replay buffer (high/mid/diversity)
    3. Value function training for better credit assignment
    4. Curriculum learning for progressive expression complexity
    """

    def __init__(
        self,
        gfn,  # EntropyTBGFlowNet
        env,  # GFNEnvCore
        encoder,  # SequenceEncoder
        pool,  # AlphaPoolGFN
        sampler: Sampler,
        # Curriculum
        curriculum=None,  # CurriculumScheduler
        curriculum_enabled: bool = True,
        # Replay config
        replay_enabled: bool = True,
        replay_capacity_high: int = 500,
        replay_capacity_mid: int = 1000,
        replay_capacity_div: int = 500,
        high_percentile: float = 0.05,
        diversity_threshold: float = 0.7,
        # Training config
        on_policy_ratio: float = 0.5,  # 50% on-policy, 50% off-policy
        batch_size: int = 32,
        # Replay mini-step (for batch_size=1 support)
        replay_ministep: bool = True,  # Always do a replay step even when n_off_policy=0
        replay_ministep_size: int = 1,  # Number of replay samples per mini-step
        # Loss weights
        lambda_policy: float = 1.0,
        lambda_value: float = 0.5,
        lambda_flow: float = 1.0,
        value_clip: float = 0.2,
        # Advantage shaping
        advantage_scale_init: float = 5.0,
        advantage_scale_final: float = 1.0,
        advantage_clip: float = 20.0,
        normalize_log_advantages: bool = True,
        # Baseline type: 'batch_mean' (compare each trajectory to batch mean)
        baseline_type: str = 'batch_mean',
        # Device
        device: torch.device = None,
        # Logging
        log_interval: int = 100,
        # Debug/profiling
        debug: bool = False,
        debug_first_n: int = 3,
    ):
        """
        Initialize SmartTrainer.

        Args:
            gfn: GFlowNet model
            env: Factor mining environment
            encoder: Sequence encoder with value head
            pool: Alpha factor pool
            sampler: Trajectory sampler
            curriculum: Curriculum scheduler (optional)
            curriculum_enabled: Whether to use curriculum
            replay_enabled: Whether to use replay buffer
            replay_capacity_*: Buffer capacities
            high_percentile: Top % for elite buffer
            diversity_threshold: Min distance for diversity buffer
            on_policy_ratio: Fraction of on-policy samples per batch
            batch_size: Total batch size
            replay_ministep: Enable replay mini-step for batch_size=1 (always sample from replay)
            replay_ministep_size: Number of replay samples per mini-step
            lambda_*: Loss weights
            value_clip: PPO-style clipping
            advantage_scale_init: Initial multiplier on log-advantages
            advantage_scale_final: Final multiplier after annealing
            advantage_clip: Clamp magnitude of normalized advantages
            normalize_log_advantages: Whether to z-score log-advantages per batch
            baseline_type: Baseline for advantage computation ('batch_mean')
            device: Torch device
            log_interval: Steps between detailed logging
        """
        self.gfn = gfn
        self.env = env
        self.encoder = encoder
        self.pool = pool
        self.sampler = sampler
        self.curriculum = curriculum
        self.curriculum_enabled = curriculum_enabled and curriculum is not None
        self.replay_enabled = replay_enabled
        self.on_policy_ratio = on_policy_ratio
        self.batch_size = batch_size
        self.replay_ministep = replay_ministep
        self.replay_ministep_size = replay_ministep_size
        self.device = device or torch.device('cpu')
        self.log_interval = log_interval
        self.debug = bool(debug)
        self.debug_first_n = max(0, int(debug_first_n))
        self.advantage_scale_init = advantage_scale_init
        self.advantage_scale_final = advantage_scale_final
        self.advantage_clip = advantage_clip
        self.normalize_log_advantages = normalize_log_advantages

        # Initialize replay buffer
        if self.replay_enabled:
            self.replay_buffer = StratifiedReplayBuffer(
                capacity_high=replay_capacity_high,
                capacity_mid=replay_capacity_mid,
                capacity_div=replay_capacity_div,
                high_percentile=high_percentile,
                diversity_threshold=diversity_threshold,
                device=self.device,
            )
        else:
            self.replay_buffer = None

        # Initialize combined loss
        self.loss_fn = CombinedLoss(
            gfn=gfn,
            encoder=encoder,
            lambda_policy=lambda_policy,
            lambda_value=lambda_value,
            lambda_flow=lambda_flow,
            value_clip=value_clip,
        )

        # Tracking
        self.total_episodes = 0
        self.total_samples = 0
        self.on_policy_samples = 0
        self.off_policy_samples = 0

        # Baseline type for advantage computation
        self.baseline_type = baseline_type

    @staticmethod
    def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute a robust Pearson correlation between two 1D tensors."""
        if a is None or b is None:
            return 0.0
        if a.numel() < 2 or b.numel() < 2:
            return 0.0
        a = a.float()
        b = b.float()
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom = (a0.std(unbiased=False) * b0.std(unbiased=False)).clamp_min(1e-8)
        corr = (a0 * b0).mean() / denom
        if torch.isnan(corr) or torch.isinf(corr):
            return 0.0
        return corr.item()

    def step(
        self,
        n_trajectories: int = None,
        save_estimator_outputs: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Execute one training step with batch mean baseline.
        """
        import time

        metrics: Dict[str, Any] = {}
        t_step_start = time.time()

        # Determine sample sizes
        if n_trajectories is None:
            n_on_policy = max(1, int(self.batch_size * self.on_policy_ratio))
        else:
            n_on_policy = n_trajectories
        n_off_policy = max(self.batch_size - n_on_policy, 0)

        # ============================================
        # 1. Sample On-Policy Trajectories
        # ============================================
        if self.debug and self.total_episodes < self.debug_first_n:
            print("[SmartTrainer][Debug] Sampling on-policy trajectories...")
        t0 = time.time()
        with torch.no_grad():
            on_policy_trajs = self.sampler.sample_trajectories(
                env=self.env,
                n_trajectories=n_on_policy,
                save_estimator_outputs=save_estimator_outputs,
                temperature=1.0,  # Explicitly force exploration (Softmax sampling)
            )
        metrics["time/sample_on_policy_s"] = time.time() - t0
        try:
            metrics["traj/on_policy_T"] = int(on_policy_trajs.states.tensor.shape[0])
            metrics["traj/on_policy_done_max"] = int(on_policy_trajs.when_is_done.max().item())
            metrics["traj/on_policy_done_mean"] = float(on_policy_trajs.when_is_done.float().mean().item())
        except Exception:
            pass

        if self.debug and self.total_episodes < self.debug_first_n:
            try:
                shape = tuple(on_policy_trajs.states.tensor.shape)
                done = on_policy_trajs.when_is_done.detach().cpu().tolist()
                print(f"[SmartTrainer][Debug] On-policy traj states shape: {shape}, when_done={done}")
            except Exception:
                pass

        # Get rewards for on-policy trajectories
        if self.debug and self.total_episodes < self.debug_first_n:
            print("[SmartTrainer][Debug] Computing on-policy rewards...")
        t0 = time.time()
        with torch.no_grad():
            on_policy_rewards, reward_components = self.env.reward(on_policy_trajs.last_states, return_components=True)
        metrics["time/reward_on_policy_s"] = time.time() - t0

        # ============================================
        # 1b. Batch Mean Baseline (trajectory-to-trajectory comparison)
        # ============================================
        # Each trajectory is compared against the mean reward of the batch.
        # Advantage_i = log(R_i) - log(mean(R_batch))
        # This is simpler and statistically justified - no caching needed.

        batch_mean_reward = on_policy_rewards.mean().item()
        metrics['baseline/batch_mean_reward'] = batch_mean_reward
        metrics['baseline/batch_std_reward'] = on_policy_rewards.std().item() if on_policy_rewards.numel() > 1 else 0.0
        metrics['baseline/batch_min_reward'] = on_policy_rewards.min().item()
        metrics['baseline/batch_max_reward'] = on_policy_rewards.max().item()

        # Log separated reward components
        if reward_components:
            for k, v_tensor in reward_components.items():
                metrics[f'reward_components/{k}_mean'] = v_tensor.mean().item()

        # ============================================
        # Advantage Computation: log(R_i) - log(mean(R_batch))
        # ============================================
        # Use Log-Scale Advantage for multiplicative reward structure.
        # Each trajectory compared against batch mean.

        # Safe log for on-policy rewards
        log_on_policy_rewards = torch.log(on_policy_rewards.clamp_min(1e-12))

        # Safe log for batch mean baseline
        log_batch_mean = math.log(max(batch_mean_reward, 1e-12))

        # Raw advantage: how much better/worse than batch mean
        raw_advantages = log_on_policy_rewards - log_batch_mean

        # Optional per-batch normalization to control variance
        adv_norm = raw_advantages
        if self.normalize_log_advantages and adv_norm.numel() > 1:
            adv_std = adv_norm.std(unbiased=False)
            if adv_std > 1e-6:
                adv_norm = (adv_norm - adv_norm.mean()) / (adv_std + 1e-8)

        # Optional clipping for stability
        if self.advantage_clip is not None and self.advantage_clip > 0:
            adv_norm = adv_norm.clamp(-self.advantage_clip, self.advantage_clip)

        # Anneal advantage scale from init -> final using training progress if available
        progress = 0.0
        if hasattr(self.pool, "training_progress"):
            try:
                progress = float(self.pool.training_progress)
            except Exception:
                progress = 0.0
        progress = max(0.0, min(1.0, progress))
        adv_scale = self.advantage_scale_init + (self.advantage_scale_final - self.advantage_scale_init) * progress

        advantages = adv_norm * adv_scale

        metrics['advantage/raw_mean'] = raw_advantages.mean().item()
        metrics['advantage/raw_std'] = raw_advantages.std(unbiased=False).item() if raw_advantages.numel() > 1 else 0.0
        metrics['advantage/scale'] = adv_scale

        metrics['advantage/mean'] = advantages.mean().item()
        metrics['advantage/std'] = advantages.std().item() if advantages.numel() > 1 else 0.0
        metrics['advantage/min'] = advantages.min().item()
        metrics['advantage/max'] = advantages.max().item()
        metrics['advantage/active_cnt'] = (advantages.abs() > 1e-4).float().sum().item()

        # Extract terminal state tokens for embeddings (no value head)
        done_steps = on_policy_trajs.when_is_done.to(self.device)
        max_state_idx = on_policy_trajs.states.tensor.shape[0] - 1
        done_steps = torch.clamp(done_steps, 0, max_state_idx)
        batch_indices = torch.arange(n_on_policy, device=self.device)
        on_policy_state_tokens = on_policy_trajs.states.tensor[done_steps, batch_indices]
        
        if self.debug and self.total_episodes < self.debug_first_n:
            print("[SmartTrainer][Debug] Computing on-policy embeddings...")
        t0 = time.time()
        with torch.no_grad():
            on_policy_embeddings = self.encoder(on_policy_state_tokens)
            # Value head removed
        metrics["time/encode_on_policy_s"] = time.time() - t0

        metrics['on_policy_samples'] = n_on_policy

        # ============================================
        # 2. Compute On-Policy Loss (Batch Mean Baseline)
        # ============================================
        # Pass pre-computed advantages as "Rewards" to CombinedLoss.
        # Advantage = log(R_i) - log(mean(R_batch)) for each trajectory.
        
        on_policy_log_r_tb = None
        if reward_components and 'log_R_tb' in reward_components:
            on_policy_log_r_tb = reward_components['log_R_tb']

        # advantages calculated above
        dummy_values = torch.zeros_like(on_policy_rewards)
        
        # Note: We pass advantages as 'rewards' to trick the old loss function
        # into computing the correct policy gradient.
        # We must disable normalization in standard loss or ensure it's okay.
        if self.debug and self.total_episodes < self.debug_first_n:
            print("[SmartTrainer][Debug] Computing on-policy loss...")
        t0 = time.time()
        on_policy_loss, on_metrics = self.loss_fn(
            trajectories=on_policy_trajs,
            rewards=advantages,
            state_tokens=on_policy_state_tokens,
            log_rewards_tb=on_policy_log_r_tb,
            old_log_probs=None,
            old_values=dummy_values, # Value Matching will try to match 0 (bad for critic, but we removed critic head)
        )
        metrics["time/loss_on_policy_s"] = time.time() - t0

        for k, v in on_metrics.items():
            metrics[f'on_policy/{k}'] = v

        # ============================================
        # 2b. Alignment / "Tug-of-war" Diagnostics
        # ============================================
        # Compare total log-reward (used by RL) vs TB target log-reward.
        if on_policy_log_r_tb is not None:
            log_r_tb = on_policy_log_r_tb.detach()
            log_r_total = torch.log(on_policy_rewards.detach().clamp_min(1e-12))
            diff = log_r_total - log_r_tb

            metrics['alignment/logR_total_mean'] = log_r_total.mean().item()
            metrics['alignment/logR_tb_mean'] = log_r_tb.mean().item()
            metrics['alignment/logR_total_logRtb_corr'] = self._safe_corr(log_r_total, log_r_tb)
            metrics['alignment/logR_total_minus_logRtb_mean'] = diff.mean().item()
            metrics['alignment/logR_total_minus_logRtb_std'] = diff.std(unbiased=False).item() if diff.numel() > 1 else 0.0
            metrics['alignment/logR_total_minus_logRtb_abs_mean'] = diff.abs().mean().item()
            metrics['alignment/adv_logRtb_corr'] = self._safe_corr(advantages.detach(), log_r_tb)

        # Relative loss scale diagnostics (not gradients, but useful signal)
        pg_loss_val = float(on_metrics.get('policy_loss', 0.0))
        flow_loss_val = float(on_metrics.get('flow_loss', 0.0))
        weighted_pg = self.loss_fn.lambda_policy * pg_loss_val
        weighted_flow = self.loss_fn.lambda_flow * flow_loss_val
        metrics['alignment/weighted_policy_loss'] = weighted_pg
        metrics['alignment/weighted_flow_loss'] = weighted_flow
        metrics['alignment/policy_flow_loss_ratio'] = abs(weighted_pg) / (abs(weighted_flow) + 1e-8)

        total_loss = on_policy_loss

        # ============================================
        # 3. Off-Policy from Replay Buffer
        # ============================================
        off_policy_loss = None
        replay_sample_count = 0

        # Determine how many replay samples to use
        if self.replay_enabled and self.replay_buffer is not None and len(self.replay_buffer) > 0:
            if n_off_policy > 0:
                # Normal case: use n_off_policy from the batch split
                replay_sample_count = n_off_policy
            elif self.replay_ministep:
                # Mini-step mode: batch_size=1 but we still want replay learning
                # Do a separate mini-step with replay_ministep_size samples
                replay_sample_count = self.replay_ministep_size

        if replay_sample_count > 0:
            # Sample from replay buffer (mixed from all tiers)
            if self.debug and self.total_episodes < self.debug_first_n:
                print("[SmartTrainer][Debug] Sampling replay buffer...")
            t0 = time.time()
            replay_samples = self.replay_buffer.sample_mixed(
                total_samples=replay_sample_count,
                high_ratio=0.5,  # 50% from elite
                mid_ratio=0.3,  # 30% from medium
                div_ratio=0.2,  # 20% from diversity
            )
            metrics["time/sample_replay_s"] = time.time() - t0

            if replay_samples:
                # Compute off-policy loss
                if self.debug and self.total_episodes < self.debug_first_n:
                    print("[SmartTrainer][Debug] Computing replay loss...")
                t0 = time.time()
                off_policy_loss, off_metrics = self._compute_replay_loss(replay_samples)
                metrics["time/loss_replay_s"] = time.time() - t0

                if off_policy_loss is not None:
                    total_loss = total_loss + off_policy_loss
                    for k, v in off_metrics.items():
                        metrics[f'off_policy/{k}'] = v
                    metrics['off_policy_samples'] = len(replay_samples)
                    metrics['replay_ministep_active'] = 1 if (n_off_policy == 0 and self.replay_ministep) else 0

        # ============================================
        # 4. Update Replay Buffer
        # ============================================
        if self.replay_enabled and self.replay_buffer is not None:
            if self.debug and self.total_episodes < self.debug_first_n:
                print("[SmartTrainer][Debug] Updating replay buffer...")
            t0 = time.time()
            self._update_replay_buffer(
                on_policy_trajs,
                on_policy_rewards,
                on_policy_embeddings,
                log_rewards_tb=on_policy_log_r_tb,
            )
            metrics["time/replay_update_s"] = time.time() - t0

            # Log replay buffer stats
            buffer_stats = self.replay_buffer.get_stats()
            for k, v in buffer_stats.items():
                metrics[f'replay/{k}'] = v

        # ============================================
        # 5. Update Curriculum
        # ============================================
        self.total_episodes += 1
        if self.curriculum_enabled and self.curriculum is not None:
            t0 = time.time()
            self.curriculum.step(self.total_episodes)
            metrics["time/curriculum_step_s"] = time.time() - t0
            if hasattr(self.curriculum, 'current_phase'):
                metrics['curriculum/phase'] = self.curriculum.current_phase
            if hasattr(self.env, 'max_len'):
                metrics['curriculum/max_len'] = self.env.max_len

        # ============================================
        # 6. Final Metrics
        # ============================================
        metrics['total_loss'] = total_loss.item() if total_loss is not None else 0.0
        metrics['total_episodes'] = self.total_episodes
        metrics["time/step_total_s"] = time.time() - t_step_start

        if self.debug and self.total_episodes <= self.debug_first_n:
            step_s = metrics.get("time/step_total_s", 0.0)
            print(f"[SmartTrainer][Debug] Step done in {step_s:.3f}s")

        return total_loss, metrics

    def _build_replay_trajectories(
        self, replay_samples: List[ReplayEntry]
    ) -> Optional["Trajectories"]:
        """
        Attempt to rebuild a Trajectories object from replay samples so we can
        reuse the TB flow loss on off-policy data.
        """
        try:
            from gfn.containers import Trajectories
        except Exception:
            return None

        # Pad variable-length trajectories to a common time dimension
        max_state_len = max(e.states_tensor.shape[0] for e in replay_samples)
        max_action_len = max(e.actions_tensor.shape[0] for e in replay_samples)

        padded_states = []
        padded_actions = []
        when_done_list = []

        pad_action_val = -1
        if hasattr(self.env, "dummy_action"):
            try:
                pad_action_val = int(self.env.dummy_action.item())
            except Exception:
                pad_action_val = -1

        for e in replay_samples:
            states = e.states_tensor.to(self.device)
            actions = e.actions_tensor.to(self.device)

            # Pad states by repeating the final state
            if states.shape[0] < max_state_len:
                pad_rows = states[-1:].repeat(max_state_len - states.shape[0], 1)
                states = torch.cat([states, pad_rows], dim=0)

            # Pad actions with dummy action
            if actions.shape[0] < max_action_len:
                pad_actions = torch.full(
                    (max_action_len - actions.shape[0],),
                    fill_value=pad_action_val,
                    device=self.device,
                    dtype=actions.dtype,
                )
                actions = torch.cat([actions, pad_actions], dim=0)

            padded_states.append(states)
            padded_actions.append(actions)
            when_done_list.append(min(e.when_done, max_action_len - 1))

        states_seq = torch.stack(padded_states, dim=1)
        actions_seq = torch.stack(padded_actions, dim=1).unsqueeze(-1)
        when_done = torch.tensor(when_done_list, device=self.device)

        states_obj = self.env.States(tensor=states_seq)
        actions_obj = self.env.Actions(tensor=actions_seq)

        try:
            trajectories = Trajectories(
                states=states_obj,
                actions=actions_obj,
                estimator_outputs=None,
                log_probs=None,
                when_is_done=when_done,
            )
            return trajectories
        except Exception:
            return None

    def _trajectory_log_prob_sum(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        when_done: int,
    ) -> torch.Tensor:
        """
        Recompute current policy log-probability of a stored trajectory.

        Args:
            states_tensor: (max_len+1, state_dim)
            actions_tensor: (max_len,)
            when_done: termination step index
        """
        max_steps = min(actions_tensor.shape[0], max(when_done, 1))
        log_probs = []

        for t in range(max_steps):
            state_t = states_tensor[t].unsqueeze(0)  # (1, state_dim)
            action_t = actions_tensor[t]

            # Wrap into env-specific States object
            states_obj = self.env.States(tensor=state_t)

            # Get valid-action mask from the env to keep logits consistent with sampler
            self.env.update_masks(states_obj)
            forward_masks = getattr(states_obj, "forward_masks", None)
            logits = self.gfn.pf.module(self.gfn.pf.preprocessor(states_obj))

            logits_masked = logits.clone()
            if forward_masks is not None:
                valid_mask = forward_masks[0]
                invalid_actions = ~valid_mask
                logits_masked[:, invalid_actions] = float("-inf")

            log_prob = torch.log_softmax(logits_masked, dim=-1)[0, action_t.item()]
            log_probs.append(log_prob)

            if action_t.item() == self.env.idx_exit:
                break

        if log_probs:
            return torch.stack(log_probs).sum()

        return torch.tensor(0.0, device=self.device)

    def _compute_replay_loss(
        self,
        replay_samples: List[ReplayEntry],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """
        Compute loss from replay buffer samples.

        Uses importance sampling correction for off-policy learning.

        Args:
            replay_samples: List of ReplayEntry from replay buffer

        Returns:
            loss: Combined loss for replay samples
            metrics: Loss component metrics
        """
        if not replay_samples:
            return None, {}

        # Stack replay data into batches
        batch_size = len(replay_samples)

        # Collect tensors
        final_states = []
        rewards_list = []
        log_r_tb_list = []
        log_prob_sums = []

        for entry in replay_samples:
            entry = entry.to_device(self.device)

            # Select terminal state tokens
            when_done = entry.when_done if entry.when_done is not None else 0
            state_idx = min(max(when_done, 0), entry.states_tensor.shape[0] - 1)
            final_states.append(entry.states_tensor[state_idx])
            rewards_list.append(entry.reward)
            log_r_tb_list.append(getattr(entry, 'log_reward_tb', 0.0))

            # log_prob_sums calculation removed as it is unused for off-policy (policy_loss=0)
            # and causes massive slowdown due to step-wise forward passes.

        final_state_tokens = torch.stack(final_states).to(self.device)
        rewards = torch.tensor(rewards_list, device=self.device)
        # log_prob_tensor = torch.stack(log_prob_sums) # Unused

        # Get embeddings (Values removed)
        embeddings = self.encoder(final_state_tokens)
        values = torch.zeros(batch_size, device=self.device)

        # Replay samples are used ONLY for GFlowNet Trajectory Balance (Flow Loss).
        # We skip Policy Gradient (REINFORCE is on-policy) and Value Loss (removed).
        policy_loss = torch.tensor(0.0, device=self.device)
        value_loss = torch.tensor(0.0, device=self.device)

        # Optional flow loss using reconstructed trajectories
        flow_loss_val = None
        replay_trajs = self._build_replay_trajectories(replay_samples)
        if replay_trajs is not None and self.loss_fn.lambda_flow > 0:
            try:
                # Construct log_rewards_tb tensor
                replay_log_r_tb = torch.tensor(log_r_tb_list, device=self.device, dtype=torch.float)
                
                # Recalculate logprobs ensures off-policy validity for TB
                flow_loss_val, _, _ = self.loss_fn.flow_loss(
                    replay_trajs, 
                    log_rewards_tb=replay_log_r_tb,
                    recalculate_logprobs=True
                )
            except Exception:
                flow_loss_val = None

        total_loss = torch.tensor(0.0, device=self.device)
        if flow_loss_val is not None:
            total_loss = self.loss_fn.lambda_flow * flow_loss_val

        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'mean_reward': rewards.mean().item(),
            'mean_value': values.mean().item(),
        }
        if flow_loss_val is not None:
            metrics['flow_loss'] = flow_loss_val.item()

        return total_loss, metrics

    def _update_replay_buffer(
        self,
        trajectories: Trajectories,
        rewards: torch.Tensor,
        embeddings: torch.Tensor,
        log_rewards_tb: Optional[torch.Tensor] = None,
    ):
        """
        Add trajectories to replay buffer.

        Args:
            trajectories: Batch of trajectories
            rewards: Rewards for each trajectory
            embeddings: State embeddings for diversity computation
            log_rewards_tb: log R for TB
        """
        batch_size = trajectories.n_trajectories

        for i in range(batch_size):
            # Extract single trajectory data
            # States: (max_len+1, batch, state_dim) -> (max_len+1, state_dim)
            states_tensor = trajectories.states.tensor[:, i, :]

            # Actions: (max_len, batch) -> (max_len,)
            actions_tensor = trajectories.actions.tensor[:, i].squeeze(-1)

            # Log probs: if available
            log_probs = None
            if hasattr(trajectories, 'log_probs') and trajectories.log_probs is not None:
                log_probs = trajectories.log_probs[:, i]

            # When done
            when_done = trajectories.when_is_done[i].item()

            # Add to buffer
            self.replay_buffer.add(
                states_tensor=states_tensor,
                actions_tensor=actions_tensor,
                reward=rewards[i].item(),
                log_reward_tb=log_rewards_tb[i].item() if log_rewards_tb is not None else 0.0,
                log_probs=log_probs,
                embedding=embeddings[i] if embeddings is not None else None,
                when_done=when_done,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {
            'total_episodes': self.total_episodes,
            'total_samples': self.total_samples,
        }

        if self.replay_enabled and self.replay_buffer is not None:
            stats.update({f'replay/{k}': v for k, v in self.replay_buffer.get_stats().items()})

        if self.curriculum_enabled and self.curriculum is not None:
            stats['curriculum_phase'] = self.curriculum.current_phase
            if hasattr(self.env, 'max_len'):
                stats['max_expression_length'] = self.env.max_len

        return stats


def create_smart_trainer(
    gfn,
    env,
    encoder,
    pool,
    sampler,
    curriculum=None,
    # Flags
    smart_training: bool = True,
    replay_enabled: bool = True,
    curriculum_enabled: bool = True,
    # Config
    batch_size: int = 32,
    on_policy_ratio: float = 0.5,
    # Replay mini-step (for batch_size=1 support)
    replay_ministep: bool = True,
    replay_ministep_size: int = 1,
    # Loss weights
    lambda_policy: float = 1.0,
    lambda_value: float = 0.5,
    lambda_flow: float = 1.0,
    replay_capacity_high: int = 500,
    replay_capacity_mid: int = 1000,
    replay_capacity_div: int = 500,
    high_percentile: float = 0.05,
    device: torch.device = None,
    debug: bool = False,
    debug_first_n: int = 3,
) -> Optional[SmartTrainer]:
    """
    Factory function to create SmartTrainer.

    Returns None if smart_training is False.
    """
    if not smart_training:
        return None

    return SmartTrainer(
        gfn=gfn,
        env=env,
        encoder=encoder,
        pool=pool,
        sampler=sampler,
        curriculum=curriculum,
        curriculum_enabled=curriculum_enabled,
        replay_enabled=replay_enabled,
        replay_capacity_high=replay_capacity_high,
        replay_capacity_mid=replay_capacity_mid,
        replay_capacity_div=replay_capacity_div,
        high_percentile=high_percentile,
        on_policy_ratio=on_policy_ratio,
        batch_size=batch_size,
        replay_ministep=replay_ministep,
        replay_ministep_size=replay_ministep_size,
        lambda_policy=lambda_policy,
        lambda_value=lambda_value,
        lambda_flow=lambda_flow,
        device=device,
        debug=debug,
        debug_first_n=debug_first_n,
    )
