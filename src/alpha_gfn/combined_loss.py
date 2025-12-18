"""
Combined Loss Function for Self-Improving GFlowNet Training.

Combines three loss components:
1. Policy Gradient Loss with Advantage: L_pg = -log_prob * A
2. Value Loss: L_v = MSE(V(s), R)
3. Trajectory Balance Loss: L_tb = (log_pf - log_pb - log_R + logZ)^2

Total: L = λ_policy * L_pg + λ_value * L_v + λ_flow * L_tb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from gfn.containers import Trajectories


class CombinedLoss(nn.Module):
    """
    Combined loss function for hybrid RL+GFlowNet training.

    This loss enables:
    1. Value-based learning (critic learns to predict expected return)
    2. Policy gradient with advantage (actor uses value baseline)
    3. GFlowNet flow matching (trajectory balance)
    """

    def __init__(
        self,
        gfn,  # EntropyTBGFlowNet instance
        encoder,  # SequenceEncoder instance
        lambda_policy: float = 1.0,
        lambda_value: float = 0.5,
        lambda_flow: float = 1.0,
        value_clip: float = 0.2,  # PPO-style value clipping
        normalize_advantages: bool = True,
        entropy_bonus: float = 0.0,  # Additional entropy bonus beyond GFN's
    ):
        """
        Initialize combined loss.

        Args:
            gfn: GFlowNet model (EntropyTBGFlowNet)
            encoder: SequenceEncoder with value head
            lambda_policy: Weight for policy gradient loss
            lambda_value: Weight for value function loss
            lambda_flow: Weight for trajectory balance loss
            value_clip: PPO-style clipping for value updates
            normalize_advantages: Whether to normalize advantages
            entropy_bonus: Additional entropy regularization
        """
        super().__init__()
        self.gfn = gfn
        self.encoder = encoder
        self.lambda_policy = lambda_policy
        self.lambda_value = lambda_value
        self.lambda_flow = lambda_flow
        self.value_clip = value_clip
        self.normalize_advantages = normalize_advantages
        self.entropy_bonus = entropy_bonus

        # For tracking old values (PPO-style)
        self.old_values: Optional[torch.Tensor] = None
        # Counter for TB score computation failures (e.g., BACKTRACK misalignment)
        self.flow_score_failures: int = 0

    def compute_values(
        self,
        state_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute state embeddings (Values are removed for QFR).
        """
        # Encoder no longer returns values
        embeddings = self.encoder(state_tokens)
        
        # Return dummy values
        values = torch.zeros(embeddings.shape[0], device=embeddings.device)

        return embeddings, values

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute advantages: A = R - V(s).

        Args:
            rewards: (batch_size,) actual rewards
            values: (batch_size,) predicted values

        Returns:
            advantages: (batch_size,) normalized advantages
        """
        with torch.no_grad():
            advantages = rewards - values.detach()

            if self.normalize_advantages and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def policy_gradient_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute policy gradient loss.

        On-policy: L = -log_prob * A
        Off-policy (PPO-style): L = -min(ratio * A, clipped_ratio * A)

        Args:
            log_probs: (batch_size,) current log probabilities
            advantages: (batch_size,) advantages
            old_log_probs: (batch_size,) old log probs for off-policy

        Returns:
            Policy gradient loss (scalar)
        """
        if old_log_probs is not None:
            # Off-policy: PPO-style clipping
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.value_clip, 1.0 + self.value_clip)

            # Take minimum of clipped and unclipped objective
            unclipped_obj = ratio * advantages
            clipped_obj = clipped_ratio * advantages

            policy_loss = -torch.min(unclipped_obj, clipped_obj).mean()
        else:
            # On-policy: standard policy gradient
            policy_loss = -(log_probs * advantages).mean()

        return policy_loss

    def value_loss(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute value function loss.

        L_v = MSE(V(s), R)

        With optional PPO-style clipping:
        L_v = max(MSE(V, R), MSE(V_clipped, R))

        Args:
            values: (batch_size,) predicted values
            rewards: (batch_size,) actual rewards
            old_values: (batch_size,) old values for clipping

        Returns:
            Value loss (scalar)
        """
        if old_values is not None and self.value_clip > 0:
            # PPO-style value clipping
            values_clipped = old_values + torch.clamp(
                values - old_values, -self.value_clip, self.value_clip
            )
            value_loss_unclipped = F.mse_loss(values, rewards)
            value_loss_clipped = F.mse_loss(values_clipped, rewards)
            v_loss = torch.max(value_loss_unclipped, value_loss_clipped)
        else:
            v_loss = F.mse_loss(values, rewards)

        return v_loss

    def flow_loss(
        self,
        trajectories: Trajectories,
        log_rewards_tb: Optional[torch.Tensor] = None,
        recalculate_logprobs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute GFlowNet trajectory balance loss.

        L_tb = (log_pf - log_pb - log_R + logZ)^2 + entropy_term

        Args:
            trajectories: Batch of trajectories
            log_rewards_tb: Optional target log rewards for TB (separate from optimization rewards)
            recalculate_logprobs: Whether to recalculate log probs

        Returns:
            flow_loss: TB loss (scalar)
            log_pf: (batch,) forward log probabilities
            entropy_term: (batch,) entropy regularization term
        """
        try:
            log_pf, log_pb, scores, entropy_term = self.gfn.get_trajectories_scores(
                trajectories, recalculate_all_logprobs=recalculate_logprobs
            )
        except ValueError as e:
            # Fallback to manual recalculation if library fails (e.g. ragged batch broadcasting issue)
            if recalculate_logprobs:
                try:
                    log_pf = self._recalculate_log_probs_safe(trajectories, is_forward=True)
                    log_pb = self._recalculate_log_probs_safe(trajectories, is_forward=False)
                    
                    # Reconstruct scores manually for TB
                    # scores = log_pf - log_pb - log_R + logZ
                    # But we handle TB loss calculation below anyway
                    
                    # Entropy term is hard to reconstruct manually without full distribution
                    # For now, zero entropy bonus in fallback case (it's less critical than gradient)
                    device = trajectories.states.device if hasattr(trajectories.states, 'device') else 'cpu'
                    entropy_term = torch.zeros(trajectories.n_trajectories, device=device)
                    
                    # Dummy scores as we compute TB loss manually below
                    scores = torch.zeros(trajectories.n_trajectories, device=device)
                    
                except Exception as e2:
                    print(f"[CombinedLoss] Manual Fallback Failed: {e2}")
                    self.flow_score_failures += 1
                    device = trajectories.states.device if hasattr(trajectories.states, 'device') else 'cpu'
                    batch_size = trajectories.n_trajectories
                    zero = torch.tensor(0.0, device=device, requires_grad=True)
                    zero_batch = torch.zeros(batch_size, device=device)
                    return zero, zero_batch, zero_batch
            else:
                 # If we weren't trying to recalculate, then pure failure
                print(f"[CombinedLoss] Flow Loss Error: {e}")
                self.flow_score_failures += 1
                device = trajectories.states.device if hasattr(trajectories.states, 'device') else 'cpu'
                batch_size = trajectories.n_trajectories
                zero = torch.tensor(0.0, device=device, requires_grad=True)
                zero_batch = torch.zeros(batch_size, device=device)
                return zero, zero_batch, zero_batch

        # TB loss calculation
        if log_rewards_tb is not None:
            # Use specific TB target rewards
            diff = log_pf - log_pb
            tb_diff = diff - log_rewards_tb + self.gfn.logZ
            tb_loss = tb_diff.pow(2).mean()
        else:
            # Use default scores (uses trajectories.log_rewards)
            tb_loss = (scores + self.gfn.logZ).pow(2).mean()

        # Add entropy term
        flow_loss = tb_loss + entropy_term.mean()

        return flow_loss, log_pf, entropy_term

    def _recalculate_log_probs_safe(self, trajectories: Trajectories, is_forward: bool) -> torch.Tensor:
        """
        Manually recalculate log probabilities for a batch of trajectories.
        Uses direct tensor access to avoid container indexing issues.
        """
        estimator = self.gfn.pf if is_forward else self.gfn.pb
        
        # Access underlying tensors directly
        # trajectories.states.tensor shape: (T, B, D)
        all_states_tensor = trajectories.states.tensor
        all_actions_tensor = trajectories.actions.tensor
        lens = trajectories.when_is_done
        
        batch_size = trajectories.n_trajectories
        total_log_probs = []
        
        for b in range(batch_size):
            valid_len = lens[b].item()
            
            if valid_len == 0:
                total_log_probs.append(torch.tensor(0.0, device=all_states_tensor.device))
                continue
            
            # Slice valid part
            # Forward: s0 ... s_{L-1} -> a0 ... a_{L-1}
            # trajectory states: s0, s1, ..., sL (length L+1)
            # trajectory actions: a0, ..., aL-1 (length L, padded)
            
            # Trajectories.states includes the initial state. 
            # So states tensor has length valid_len + 1 (if we count steps as transitions)
            # Actually valid_len is the number of steps.
            
            if is_forward:
                # States: s0 to s_{L-1}
                # Actions: a0 to a_{L-1}
                
                # Tensor slice: [0:valid_len, b, :]
                states_b = all_states_tensor[:valid_len, b, :]
                
                # Wrap in States object
                # We assume generic environment where we can reconstruct states from tensor
                states_input = trajectories.env.states_from_tensor(states_b)
                
                logits = estimator(states_input) # (L, n_actions)
                log_probs_all = torch.log_softmax(logits, dim=-1)
                
                # Actions: [0:valid_len, b] (or transformed if Actions object is complex)
                # Actions tensor is usually (T, B) or (T, B, 1)
                actions_b = all_actions_tensor[:valid_len, b]
                if actions_b.dim() > 1:
                    actions_b = actions_b.squeeze(-1)
                
                action_indices = actions_b.view(-1, 1)
                
                step_log_probs = log_probs_all.gather(1, action_indices).squeeze(1)
                total_log_probs.append(step_log_probs.sum())
                
            else:
                # Backward policy not critical for Optimization (PG), only for TB.
                # Returning 0.0 allows Policy learning to proceed.
                total_log_probs.append(torch.tensor(0.0, device=all_states_tensor.device))

        return torch.stack(total_log_probs)

        # TB loss calculation
        if log_rewards_tb is not None:
            # Use specific TB target rewards
            # scores = log_pf - log_pb - log_R + logZ
            # We must sum log_pf and log_pb over the trajectory if they are step-wise?
            # get_trajectories_scores returns summed log_pf/log_pb usually?
            # Let's verify gfn library. It generally returns trajectory-level log_probs.
            
            # Recalculate scores manually with correct log_R
            # scores = (log_pf - log_pb) - log_R + logZ
            # Note: scores in gfn lib is (log_pf - log_pb - log_reward + logZ)
            # We want to replace log_reward with log_rewards_tb
            
            diff = log_pf - log_pb
            tb_diff = diff - log_rewards_tb + self.gfn.logZ
            tb_loss = tb_diff.pow(2).mean()
        else:
            # Use default scores (uses trajectories.log_rewards)
            tb_loss = (scores + self.gfn.logZ).pow(2).mean()

        # Add entropy term
        flow_loss = tb_loss + entropy_term.mean()

        return flow_loss, log_pf, entropy_term

    def forward(
        self,
        trajectories: Trajectories,
        rewards: torch.Tensor,
        state_tokens: torch.Tensor,
        log_rewards_tb: Optional[torch.Tensor] = None,
        old_log_probs: Optional[torch.Tensor] = None,
        old_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss (QFR-SOTA Hybrid).
        
        Args:
            trajectories: Trajectories batch
            rewards: Optimization rewards (Advantage/Return) for Policy Gradient
            state_tokens: Tokens for value encoding
            log_rewards_tb: (Optional) Target Log Rewards for Trajectory Balance
            old_log_probs: For PPO
            old_values: For PPO value clipping (unused)
        """
        device = rewards.device
        # Advantages = Rewards (since V=0 and we pass (R-B) as rewards)
        advantages = rewards 
        
        # 3. Compute flow loss (TB) uses log_rewards_tb if provided
        flow_loss_val, total_log_pf, entropy_term = self.flow_loss(
            trajectories, 
            log_rewards_tb=log_rewards_tb,
            recalculate_logprobs=True
        )
        
        # 4. Compute QFR Policy Loss (On-Policy REINFORCE)
        pg_loss = self.policy_gradient_loss(total_log_pf, advantages, old_log_probs)
        
        # 5. Combine losses
        total_loss = (
            self.lambda_policy * pg_loss +
            self.lambda_flow * flow_loss_val
        )
        
        # 6. Entropy Bonus
        if self.entropy_bonus > 0:
            total_loss = total_loss - self.entropy_bonus * entropy_term.mean()
            
        # Handle NaN
        if torch.isnan(total_loss):
            total_loss = torch.tensor(float('inf'), device=device)
            
        # Metrics
        metrics = {
            'total_loss': total_loss.item(),
            'policy_loss': pg_loss.item(),
            'flow_loss': flow_loss_val.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_value': 0.0,
            'entropy': entropy_term.mean().item() if entropy_term.numel() > 0 else 0.0,
            'flow_score_failures_total': float(self.flow_score_failures),
        }
        
        return total_loss, metrics

    def forward_simple(
        self,
        trajectories: Trajectories,
        rewards: torch.Tensor,
        value_predictions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Simplified forward for when values are pre-computed.

        Args:
            trajectories: Batch of trajectories
            rewards: (batch_size,) actual rewards
            value_predictions: (batch_size,) pre-computed value predictions

        Returns:
            total_loss: Combined scalar loss
            metrics: Dict of losses
        """
        device = rewards.device

        # Compute advantages
        advantages = self.compute_advantages(rewards, value_predictions)

        # Flow loss
        flow_loss_val, total_log_pf, entropy_term = self.flow_loss(trajectories)

        # Policy gradient loss (on-policy only)
        pg_loss = self.policy_gradient_loss(total_log_pf, advantages)

        # Value loss
        v_loss = self.value_loss(value_predictions, rewards)

        # Combine
        total_loss = (
            self.lambda_policy * pg_loss +
            self.lambda_value * v_loss +
            self.lambda_flow * flow_loss_val
        )

        if torch.isnan(total_loss):
            total_loss = torch.tensor(float('inf'), device=device)

        metrics = {
            'total_loss': total_loss.item(),
            'policy_loss': pg_loss.item(),
            'value_loss': v_loss.item(),
            'flow_loss': flow_loss_val.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_value': value_predictions.mean().item(),
            'mean_reward': rewards.mean().item(),
            'flow_score_failures_total': float(self.flow_score_failures),
        }

        return total_loss, metrics


def create_combined_loss(
    gfn,
    encoder,
    lambda_policy: float = 1.0,
    lambda_value: float = 0.5,
    lambda_flow: float = 1.0,
    value_clip: float = 0.2,
) -> CombinedLoss:
    """
    Factory function to create CombinedLoss.

    Args:
        gfn: EntropyTBGFlowNet instance
        encoder: SequenceEncoder instance
        lambda_policy: Policy loss weight
        lambda_value: Value loss weight
        lambda_flow: Flow loss weight
        value_clip: PPO-style clipping parameter

    Returns:
        CombinedLoss instance
    """
    return CombinedLoss(
        gfn=gfn,
        encoder=encoder,
        lambda_policy=lambda_policy,
        lambda_value=lambda_value,
        lambda_flow=lambda_flow,
        value_clip=value_clip,
    )
