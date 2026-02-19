import torch
from torch import nn
import math
from typing import Optional, Union, Tuple
import torch.nn.utils.rnn as rnn_utils


class PositionalEncoding(nn.Module):
    """Absolute positional encoding (sinusoidal)"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe[:x.size(1), :]
        x = x + pe.squeeze(1)
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for Transformer-XL"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Create relative position embeddings
        # This will be used to compute position-based bias in attention
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Returns relative positional encodings for a sequence of length seq_len
        Output shape: [2*seq_len-1, d_model]
        """
        # For relative positions from -(seq_len-1) to (seq_len-1)
        return self.pe[:2*seq_len-1]


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding (Transformer-XL style)"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Relative position projections
        self.w_k_r = nn.Linear(d_model, d_model)  # Key projection for relative positions
        
        # Learnable biases for content and position
        self.u = nn.Parameter(torch.zeros(num_heads, self.d_k))  # content bias
        self.v = nn.Parameter(torch.zeros(num_heads, self.d_k))  # position bias
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)
    
    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shift relative position scores to align properly
        Input: [batch, num_heads, q_len, 2*k_len-1]
        Output: [batch, num_heads, q_len, k_len]
        """
        batch_size, num_heads, q_len, k_len_ext = x.size()
        
        # Pad and reshape to shift
        x = torch.nn.functional.pad(x, (1, 0))  # [batch, heads, q_len, k_len_ext+1]
        x = x.view(batch_size, num_heads, k_len_ext + 1, q_len)
        x = x[:, :, 1:, :].view(batch_size, num_heads, q_len, k_len_ext)
        
        # Take only the first k_len columns
        k_len = (k_len_ext + 1) // 2
        return x[:, :, :, :k_len]
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        rel_pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, q_len, d_model]
            key: [batch, k_len, d_model]
            value: [batch, k_len, d_model]
            rel_pos_emb: [2*k_len-1, d_model] relative position embeddings
            mask: Optional attention mask
        Returns:
            [batch, q_len, d_model]
        """
        batch_size, q_len, _ = query.size()
        k_len = key.size(1)
        
        # Project Q, K, V and reshape for multi-head
        q = self.w_q(query).view(batch_size, q_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Project relative positions
        k_r = self.w_k_r(rel_pos_emb).view(-1, self.num_heads, self.d_k).transpose(0, 1)
        # k_r: [num_heads, 2*k_len-1, d_k]
        
        # Compute content-based attention: (q + u)^T @ k^T
        # q: [batch, heads, q_len, d_k]
        # u: [heads, d_k] -> [1, heads, 1, d_k]
        q_with_u = q + self.u.unsqueeze(0).unsqueeze(2)
        content_score = torch.matmul(q_with_u, k.transpose(-2, -1))
        # content_score: [batch, heads, q_len, k_len]
        
        # Compute position-based attention: (q + v)^T @ k_r^T
        q_with_v = q + self.v.unsqueeze(0).unsqueeze(2)
        # q_with_v: [batch, heads, q_len, d_k]
        # k_r: [heads, 2*k_len-1, d_k]
        position_score = torch.matmul(q_with_v, k_r.transpose(-2, -1))
        # position_score: [batch, heads, q_len, 2*k_len-1]
        
        # Shift relative position scores
        position_score = self._rel_shift(position_score)
        # position_score: [batch, heads, q_len, k_len]
        
        # Combine content and position scores
        attn_score = (content_score + position_score) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = torch.softmax(attn_score, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        # out: [batch, heads, q_len, d_k]
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        out = self.out_proj(out)
        
        return out


class RelativeTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with relative positional encoding"""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        super(RelativeTransformerEncoderLayer, self).__init__()
        
        self.self_attn = RelativeMultiHeadAttention(d_model, num_heads, dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        rel_pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.self_attn(x, x, x, rel_pos_emb, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feedforward with residual connection
        ff_out = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        output_activation: Optional[str] = None,
        weight_init: str = 'default',
        use_relative_positions: bool = False,  # New parameter
        *args,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.weight_init = weight_init
        self.use_relative_positions = use_relative_positions

        # Project inputs to d_model
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        if use_relative_positions:
            # Use relative positional encoding
            self.pos_encoder = RelativePositionalEncoding(hidden_size)
            self.encoder_layers = nn.ModuleList([
                RelativeTransformerEncoderLayer(
                    d_model=hidden_size,
                    num_heads=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
        else:
            # Use absolute positional encoding
            self.pos_encoder = PositionalEncoding(hidden_size, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Linear(hidden_size, output_size)

        if self.weight_init != 'default':
            self.my_reset_parameters()

    def my_reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    if self.weight_init in ['kaiming_normal', 'he_normal']:
                        nn.init.kaiming_normal_(param)
                    elif self.weight_init in ['kaiming_uniform', 'he_uniform']:
                        nn.init.kaiming_uniform_(param)
                    elif self.weight_init in ['xavier_normal', 'glorot_normal']:
                        nn.init.xavier_normal_(param)
                    elif self.weight_init in ['xavier_uniform', 'glorot_uniform']:
                        nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 3e-5)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        out_lengths: torch.Tensor,
        max_out_length: int,
        unpack: bool = None,
        return_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Unpack if needed
        if isinstance(x, rnn_utils.PackedSequence):
            unpacked_x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        else:
            unpacked_x = x

        # Project inputs
        h = self.input_proj(unpacked_x)
        
        if self.use_relative_positions:
            # Get relative positional embeddings
            seq_len = h.size(1)
            rel_pos_emb = self.pos_encoder(seq_len)
            
            # Apply encoder layers with relative positions
            for layer in self.encoder_layers:
                h = layer(h, rel_pos_emb)
        else:
            # Use absolute positional encoding
            h = self.pos_encoder(h)
            h = self.transformer_encoder(h)

        if return_hidden_states:
            return h, torch.zeros(1, h.size(0), h.size(2), device=h.device)

        # Output projection
        out = self.fc(h)
        if self.output_activation == "tanh":
            out = nn.functional.tanh(out)
        elif self.output_activation == "relu":
            out = nn.functional.relu(out)

        # Masking and extraction (same as original)
        time_indices = torch.arange(out.size(1), device=out.device).unsqueeze(0)
        zero_output_mask = time_indices >= (lengths + out_lengths).unsqueeze(1)
        out[zero_output_mask] = 0

        copy_mask = (time_indices >= lengths.unsqueeze(1)) & (
            time_indices < (lengths + max_out_length).unsqueeze(1)
        )
        out = out[copy_mask]
        out = out.view(-1, max_out_length, out.size(1))

        return out