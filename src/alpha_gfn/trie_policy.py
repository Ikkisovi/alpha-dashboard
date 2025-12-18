"""
Trie-augmented policy estimator for GFlowNet.

Wraps DiscretePolicyEstimator to add logit bonuses for tokens
that continue known-good prefixes from the expression trie.
"""

import torch
from gfn.modules import DiscretePolicyEstimator
from gfn.states import DiscreteStates


class TriePolicyEstimator(DiscretePolicyEstimator):
    """
    Wraps a base policy estimator and adds trie-based logit bonuses.
    
    The bonus is applied to actions that continue prefixes in the trie,
    guiding the policy toward known-good expression patterns.
    """
    
    def __init__(self, base_pf: DiscretePolicyEstimator, trie, bonus: float = 0.5):
        """
        Args:
            base_pf: The base forward policy estimator to wrap
            trie: ExpressionTrie instance for computing continuations
            bonus: Logit bonus value for valid continuations
        """
        # Initialize parent with base_pf's config
        super().__init__(
            module=base_pf.module,
            n_actions=base_pf.n_actions,
            preprocessor=base_pf.preprocessor,
            is_backward=base_pf.is_backward
        )
        self.base_pf = base_pf
        self.trie = trie
        self.bonus = bonus
    
    def set_trie(self, trie):
        """Update the trie (e.g., after refreshing from pool)."""
        self.trie = trie
    
    def set_bonus(self, bonus: float):
        """Update the bonus value."""
        self.bonus = bonus
    
    def forward(self, states: DiscreteStates):
        """
        Forward pass with trie bonus applied.
        
        Args:
            states: DiscreteStates with .tensor and .forward_masks
            
        Returns:
            Policy output with bonus-augmented logits
        """
        # Get base policy output (could be tensor or dataclass)
        out = self.base_pf(states)
        
        # Handle both tensor and dataclass outputs
        if isinstance(out, torch.Tensor):
            logits = out
            is_tensor = True
        else:
            logits = out.logits
            is_tensor = False
        
        # Skip if no trie or no bonus
        if self.trie is None or self.bonus <= 0 or self.trie.n_expressions == 0:
            return out
        
        # Compute per-sample bonuses in no_grad block to prevent graph retention
        with torch.no_grad():
            batch_size = states.tensor.shape[0]
            bonuses = []
            
            for i in range(batch_size):
                state_tensor = states.tensor[i]
                forward_mask = states.forward_masks[i]
                
                # Get trie bonus for this state
                bonus_vec = self.trie.get_bonus_mask(
                    state_tensor, 
                    logits.size(-1), 
                    self.bonus
                ).to(logits.device, logits.dtype)
                
                # Mask out invalid actions (safety: never inflate invalid logits)
                bonus_vec = bonus_vec.masked_fill(~forward_mask, 0.0)
                bonuses.append(bonus_vec)
            
            # Stack to create batch bonus tensor
            bonus_tensor = torch.stack(bonuses, dim=0)
        
        # Add to logits (bonus_tensor is detached by default from no_grad)
        augmented_logits = logits + bonus_tensor
        
        # Return in same format as input
        if is_tensor:
            return augmented_logits
        else:
            return type(out)(
                logits=augmented_logits,
                **{k: v for k, v in out.__dict__.items() if k != 'logits'}
            )
