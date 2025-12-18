import torch
import math
from typing import Tuple
from gfn.containers import Trajectories
from gfn.env import Env
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from torchtyping import TensorType as TT


class EntropyTBGFlowNet(TBGFlowNet):
    def __init__(
        self,
        pf: DiscretePolicyEstimator,
        pb: DiscretePolicyEstimator,
        entropy_coef: float = 0.0,
        entropy_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(pf, pb, **kwargs)
        self.entropy_coef = entropy_coef
        self.entropy_temperature = entropy_temperature

    def get_trajectories_scores(
        self, trajectories: Trajectories, recalculate_all_logprobs: bool = False
    ) -> Tuple[
        TT["n_trajectories", torch.float],
        TT["n_trajectories", torch.float],
        TT["n_trajectories", torch.float],
        TT["n_trajectories", torch.float]
    ]:
        # We reimplement the upstream scoring logic so we can also compute an
        # entropy penalty in a version-compatible way (newer `gfn` Trajectories do
        # not expose `is_done`).

        log_pf_steps, log_pb_steps = self.get_pfs_and_pbs(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )

        total_log_pf = log_pf_steps.sum(dim=0)
        total_log_pb = log_pb_steps.sum(dim=0)

        log_rewards = trajectories.log_rewards
        if log_rewards is None:
            log_rewards = torch.zeros_like(total_log_pf)

        if math.isfinite(getattr(self, "log_reward_clip_min", float("inf"))):
            log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)

        if torch.any(torch.isinf(total_log_pf)) or torch.any(torch.isinf(total_log_pb)):
            raise ValueError("Infinite logprobs found")

        scores = total_log_pf - total_log_pb - log_rewards

        # Entropy penalty (Monte Carlo estimate): E[-log p(a)] == H(p) for on-policy samples.
        # This avoids saving logits during sampling and stays compatible across gfn versions.
        if self.entropy_coef > 0:
            entropy_term = (-log_pf_steps).sum(dim=0) * self.entropy_coef
        else:
            entropy_term = torch.zeros_like(total_log_pf)

        return total_log_pf, total_log_pb, scores, entropy_term
    
    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = False,
        value_predictions: TT["n_trajectories", torch.float] = None,
        value_targets: TT["n_trajectories", torch.float] = None,
        lambda_value: float = 0.0,
    ) -> TT[0, float]:
        """Trajectory balance loss with optional value head loss.

        The trajectory balance loss is described in 2.3 of
        [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259))

        Extended to support auxiliary value head training:
        - value_predictions: V(s) predictions from encoder's value head
        - value_targets: Actual rewards R
        - lambda_value: Weight for value loss (default 0 = disabled)

        Total loss = TB_loss + lambda_value * MSE(V, R)

        Raises:
            ValueError: if the loss is NaN.
        """
        del env  # unused
        _, _, scores, entropy_term = self.get_trajectories_scores(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )

        # Trajectory Balance loss
        tb_loss = (scores + self.logZ).pow(2).mean() + entropy_term.mean()

        # Optional value loss for auxiliary critic training
        if value_predictions is not None and value_targets is not None and lambda_value > 0:
            value_loss = torch.nn.functional.mse_loss(value_predictions, value_targets)
            loss = tb_loss + lambda_value * value_loss
        else:
            loss = tb_loss

        if torch.isnan(loss):
            # set inf
            loss = torch.tensor(float('inf'))

        return loss
