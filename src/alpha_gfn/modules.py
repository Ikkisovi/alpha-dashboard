import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import Optional
from torch_geometric.nn import global_mean_pool, RGCNConv
from torch_geometric.data import Data, Batch

from .config import *
from .relation_types import RelationType, get_operator_relation_types
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.data.tokens import (
    BEG_TOKEN,
    SEP_TOKEN,
    OperatorToken,
    FeatureToken,
    DeltaTimeToken,
    ConstantToken,
)

def _build_graph_from_rpn(token_ids: list[int], token_embedding_layer: nn.Embedding, beg_token_id: int) -> Data:
    """
    Build graph from RPN tokens (simplified, daily-only, matching upstream).
    No layer tracking, just simple operator-based relation assignment.
    [NEW] Returns stack_mask indicating which nodes are currently in the RPN stack.
    """
    device = token_embedding_layer.weight.device

    # Build the same action list used by GFNEnvCore
    # Order: BEG, operators, features, delta_times, constants, SEP
    action_list = (
        [BEG_TOKEN]
        + [OperatorToken(op) for op in OPERATORS]
        + [FeatureToken(feat) for feat in FEATURES]
        + [DeltaTimeToken(dt) for dt in DELTA_TIMES]
        + [ConstantToken(c) for c in CONSTANTS]
        + [SEP_TOKEN]
    )

    tokens = []
    for tid in token_ids:
        if tid < 0 or tid >= len(action_list):
            raise ValueError(f"Invalid action id {tid} (max valid {len(action_list)-1})")
        tokens.append(action_list[tid])

    edges = []
    edge_types = []
    stack: list[int] = []  # Just node indices, no layer tracking
    
    # [NEW] Track which nodes are currently in the stack
    # We need to simulate the stack state AFTER processing all tokens?
    # No, the graph represents the state *at the end* of the sequence.
    # So we just run the simulation and see what's left in the stack.

    for j, token in enumerate(tokens):
        if isinstance(token, OperatorToken):
            op = token.operator
            n_args = op.n_args()

            if len(stack) < n_args:
                # Insufficient operands; treat as leaf (or invalid state)
                stack.append(j)
                continue

            # Get relation types for this operator
            relation_map = dict(get_operator_relation_types(op.__name__))

            # Pop children and reverse to get expression order
            child_indices = [stack.pop() for _ in range(n_args)]
            child_indices.reverse()

            # Create edges with appropriate relation types
            for child_index, child_node_idx in enumerate(child_indices):
                edges.append((child_node_idx, j))
                rel_type = relation_map.get(child_index, RelationType.UNARY_OPERAND)
                edge_types.append(int(rel_type))

            # Push this operator node
            stack.append(j)
        else:
            # Leaf token (feature, constant, delta time)
            stack.append(j)

    # Build edge tensors
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_type = torch.empty((0,), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long, device=device)

    # Embed tokens
    node_feature_ids = torch.tensor(token_ids, device=device)
    x = token_embedding_layer(node_feature_ids)
    
    # [NEW] Create stack mask
    # 1 if node is in stack, 0 otherwise
    stack_mask = torch.zeros(len(tokens), dtype=torch.long, device=device)
    if stack:
        stack_indices = torch.tensor(stack, device=device)
        stack_mask[stack_indices] = 1

    return Data(x=x, edge_index=edge_index, edge_type=edge_type, stack_mask=stack_mask)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('_pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        return x + self._pe[:seq_len]

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=2, num_relations: int = NUM_RELATIONS):
        super().__init__()
        self.num_relations = num_relations
        self.layers = nn.ModuleList()
        self.layers.append(RGCNConv(input_dim, hidden_dim, self.num_relations))
        for _ in range(n_layers - 1):
            self.layers.append(RGCNConv(hidden_dim, hidden_dim, self.num_relations))
        
        # Value Head removed for QFR
        
        # [NEW] Stack Embedding (0 or 1)
        self.stack_embedding = nn.Embedding(2, hidden_dim)

    def forward(self, data: Batch, return_value: bool = False):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        
        # [NEW] Add stack embedding to node features
        if hasattr(data, 'stack_mask'):
            stack_emb = self.stack_embedding(data.stack_mask)
            x = x + stack_emb
            
        for layer in self.layers:
            x = torch.relu(layer(x, edge_index, edge_type))
            # Guard against NaN/Inf from numerical instability
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        pooled = global_mean_pool(x, batch)
        # Final safety check after pooling
        if torch.isnan(pooled).any() or torch.isinf(pooled).any():
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1e6, neginf=-1e6)
            
        return pooled

class SequenceEncoder(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        encoder_type: str = 'lstm',
        num_relations: int = NUM_RELATIONS,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.n_tokens = n_tokens
        self.beg_token_id = 0
        self.num_relations = num_relations

        # Reserve one extra id for padding; use a valid non-negative padding index
        self.padding_id = self.n_tokens + 1
        self.token_embedding = nn.Embedding(self.n_tokens + 2, HIDDEN_DIM, padding_idx=self.padding_id)

        if encoder_type == 'lstm':
            self.pos_enc = PositionalEncoding(HIDDEN_DIM)
            self.encoder = nn.LSTM(
                input_size=HIDDEN_DIM,
                hidden_size=HIDDEN_DIM,
                num_layers=NUM_ENCODER_LAYERS,
                batch_first=True,
                dropout=DROPOUT
            )
        elif encoder_type == 'gnn':
            self.encoder = GNNEncoder(
                input_dim=HIDDEN_DIM,
                hidden_dim=HIDDEN_DIM,
                num_relations=self.num_relations
            )
        elif encoder_type == 'hybrid_gnn_lstm':
            # Initialize both encoders
            self.pos_enc = PositionalEncoding(HIDDEN_DIM)
            self.lstm_encoder = nn.LSTM(
                input_size=HIDDEN_DIM,
                hidden_size=HIDDEN_DIM,
                num_layers=NUM_ENCODER_LAYERS,
                batch_first=True,
                dropout=DROPOUT
            )
            self.gnn_encoder = GNNEncoder(
                input_dim=HIDDEN_DIM,
                hidden_dim=HIDDEN_DIM,
                num_relations=self.num_relations
            )
            # Fusion layer: [LSTM; GNN] -> Hidden
            self.fusion_layer = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}. Supported: 'lstm', 'gnn', 'hybrid_gnn_lstm'.")

    def _make_padding_mask(self, tokens: Tensor) -> Tensor:
        return tokens == -1
        
    def forward(self, state_tokens: Tensor, return_value: bool = False):
        bs = state_tokens.shape[0]

        # Path 1: GNN (Structure)
        gnn_out = None
        gnn_value = None
        
        if self.encoder_type in ['gnn', 'hybrid_gnn_lstm']:
            # Time graph construction
            import time
            t_graph_start = time.time()
            
            data_list = []
            for i in range(bs):
                token_ids = [tid for tid in state_tokens[i].tolist() if tid > -1]
                graph_data = _build_graph_from_rpn(
                    token_ids,
                    self.token_embedding,
                    self.beg_token_id
                )
                data_list.append(graph_data)
            batched_graph = Batch.from_data_list(data_list)
            
            t_gnn_start = time.time()
            # Use self.gnn_encoder if hybrid, else self.encoder
            gnn_module = self.gnn_encoder if self.encoder_type == 'hybrid_gnn_lstm' else self.encoder
            
            if return_value:
                gnn_out, gnn_value = gnn_module(batched_graph, return_value=True)
            else:
                gnn_out = gnn_module(batched_graph, return_value=False)
                
            t_gnn_end = time.time()
            
            # Store timing stats
            if not hasattr(self, 'graph_build_time'):
                self.graph_build_time = 0.0
                self.gnn_forward_time = 0.0
                self.forward_calls = 0
            self.graph_build_time += (t_gnn_start - t_graph_start)
            self.gnn_forward_time += (t_gnn_end - t_gnn_start)
            self.forward_calls += 1

        # Path 2: LSTM (Sequence)
        lstm_out = None
        if self.encoder_type in ['lstm', 'hybrid_gnn_lstm']:
            beg_tokens = torch.full((bs, 1), fill_value=self.beg_token_id, dtype=torch.long, device=state_tokens.device)

            state_tokens_seq = torch.cat([beg_tokens, state_tokens], dim=1)
            padding_mask = self._make_padding_mask(state_tokens_seq)

            # Remap -1 paddings to a valid padding id before embedding to avoid CUDA asserts
            state_tokens_for_embedding = state_tokens_seq.clone()
            state_tokens_for_embedding[state_tokens_for_embedding == -1] = self.padding_id

            state_embedding = self.pos_enc(self.token_embedding(state_tokens_for_embedding))

            # Use self.lstm_encoder if hybrid, else self.encoder
            seq_module = self.lstm_encoder if self.encoder_type == 'hybrid_gnn_lstm' else self.encoder
            
            if isinstance(seq_module, nn.TransformerEncoder):
                hidden_states = seq_module(state_embedding, src_key_padding_mask=padding_mask)
            else:
                hidden_states, _ = seq_module(state_embedding)

            lengths = (~padding_mask).sum(dim=1)
            last_indices = lengths - 1
            batch_indices = torch.arange(state_tokens.size(0), device=state_tokens.device)
            lstm_out = hidden_states[batch_indices, last_indices]

        # Final Output Selection
        final_emb = None
        if self.encoder_type == 'gnn':
            final_emb = gnn_out
        elif self.encoder_type == 'lstm':
            final_emb = lstm_out
        elif self.encoder_type == 'hybrid_gnn_lstm':
            # Fuse: Concatenate and project
            combined = torch.cat([lstm_out, gnn_out], dim=1)
            fused = self.fusion_layer(combined)
            final_emb = fused
        
        if return_value:
            # If LSTM only, we don't have a value head currently (unless we add one).
            # For now, return gnn_value if available, else None.
            return final_emb, gnn_value
            
        return final_emb
