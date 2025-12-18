"""
Trie-based logit bonus for GFlowNet policy guidance.

Provides soft guidance toward known-good token sequences (seeds, pool expressions)
without enlarging the action space or policy head.
"""

from typing import List, Set, Optional
import torch
from torch import Tensor


class ExpressionTrie:
    """
    Prefix trie over token ID sequences.
    
    Used to identify valid continuations of partial expressions,
    enabling logit bonuses that guide the policy toward known-good patterns.
    """
    
    def __init__(self):
        self.root: dict = {}
        self.end_marker = '$END$'
        self.n_expressions = 0
    
    def clear(self):
        """Clear all entries from the trie."""
        self.root = {}
        self.n_expressions = 0
    
    def insert(self, token_ids: List[int]) -> bool:
        """
        Insert a token sequence into the trie.
        
        Args:
            token_ids: List of token IDs (should NOT include BEG token)
        
        Returns:
            True if successfully inserted, False if empty/invalid
        """
        if not token_ids:
            return False
        
        node = self.root
        for tid in token_ids:
            if tid < 0:  # Skip padding
                break
            node = node.setdefault(tid, {})
        node[self.end_marker] = True
        self.n_expressions += 1
        return True
    
    def get_continuations(self, prefix: List[int]) -> Set[int]:
        """
        Get valid next token IDs that continue any trie prefix.
        
        Args:
            prefix: Current token sequence (without BEG token)
        
        Returns:
            Set of token IDs that are valid continuations
        """
        node = self.root
        for tid in prefix:
            if tid < 0:  # Skip padding
                break
            if tid not in node:
                return set()  # Prefix not in trie
            node = node[tid]
        
        # Return all children except end marker
        return {k for k in node.keys() if k != self.end_marker}
    
    def get_bonus_mask(
        self, 
        state: Tensor, 
        n_actions: int, 
        bonus: float = 0.5,
        valid_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute logit bonus tensor for actions continuing trie prefixes.
        
        Args:
            state: State tensor of shape (MAX_EXPR_LENGTH,) with token IDs
            n_actions: Total number of actions in action space
            bonus: Logit bonus value for valid continuations
            valid_mask: Optional mask of valid actions (to avoid boosting invalid)
        
        Returns:
            Tensor of shape (n_actions,) with bonus values
        """
        # Extract prefix: skip BEG token (index 0), get non-padding tokens
        prefix = [t.item() for t in state[1:] if t.item() >= 0]
        
        # Get valid continuations from trie
        continuations = self.get_continuations(prefix)
        
        # Build bonus tensor
        bonus_tensor = torch.zeros(n_actions, device=state.device, dtype=torch.float32)
        for tid in continuations:
            if 0 <= tid < n_actions:
                bonus_tensor[tid] = bonus
        
        # Intersect with valid actions if mask provided
        if valid_mask is not None:
            bonus_tensor = bonus_tensor * valid_mask.float()
        
        return bonus_tensor
    
    def get_batch_bonus(
        self,
        states: Tensor,
        n_actions: int,
        bonus: float = 0.5,
        valid_masks: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute logit bonus for a batch of states.
        
        Args:
            states: Batch of states, shape (batch_size, MAX_EXPR_LENGTH)
            n_actions: Total number of actions
            bonus: Logit bonus value
            valid_masks: Optional masks, shape (batch_size, n_actions)
        
        Returns:
            Tensor of shape (batch_size, n_actions) with bonus values
        """
        batch_size = states.shape[0]
        result = torch.zeros(batch_size, n_actions, device=states.device, dtype=torch.float32)
        
        for i in range(batch_size):
            mask = valid_masks[i] if valid_masks is not None else None
            result[i] = self.get_bonus_mask(states[i], n_actions, bonus, mask)
        
        return result


def build_trie_from_expressions(
    expressions: List,
    env,
    max_length: int = 20
) -> ExpressionTrie:
    """
    Build a trie from a list of Expression objects.
    
    Args:
        expressions: List of Expression objects
        env: GFNEnvCore instance for token ID lookup
        max_length: Maximum expression length to include
    
    Returns:
        Populated ExpressionTrie
    """
    from seed_pool import expr_to_action_sequence
    
    trie = ExpressionTrie()
    skipped = 0
    
    for expr in expressions:
        try:
            # Get action sequence (token IDs) for this expression
            action_seq = expr_to_action_sequence(expr, env)
            if action_seq is None:
                skipped += 1
                continue
            
            # Remove SEP token at end (we want prefix, not termination)
            if action_seq:
                action_seq = action_seq[:-1]
            
            # Skip if too long
            if len(action_seq) > max_length - 1:
                skipped += 1
                continue
            
            trie.insert(action_seq)
            
        except Exception as e:
            skipped += 1
            continue
    
    print(f"[Trie] Built trie with {trie.n_expressions} expressions, skipped {skipped}")
    return trie


def build_trie_from_pool_and_seeds(
    pool,
    env,
    max_length: int = 20
) -> ExpressionTrie:
    """
    Build a trie from pool expressions and seed expressions.
    
    This ensures guidance is available even when pool is empty (uses seeds).
    
    Args:
        pool: AlphaPoolGFN instance
        env: GFNEnvCore instance
        max_length: Maximum expression length
    
    Returns:
        Populated ExpressionTrie
    """
    from seed_pool import get_all_seeds
    
    # Start with seeds (always available)
    all_exprs = get_all_seeds()
    
    # Add pool expressions if pool has content
    if pool.size > 0:
        for i in range(pool.size):
            if pool.exprs[i] is not None:
                all_exprs.append(pool.exprs[i])
    
    return build_trie_from_expressions(all_exprs, env, max_length)
