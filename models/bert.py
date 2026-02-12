import torch
from torch import nn
import math
from typing import Optional, Union, Tuple
import torch.nn.utils.rnn as rnn_utils


class BERTEmbedding(nn.Module):
    """BERT Embedding: Token + Position + Segment Embeddings"""
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(BERTEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(3, d_model)  # Support up to 3 segments
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, segment_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch, seq_len, input_size] (continuous) or [batch, seq_len] (token ids)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Handle both continuous inputs and token IDs
        if x.dim() == 3:
            # Continuous input - treat as pre-embedded tokens
            token_emb = x
        else:
            # Token IDs
            token_emb = self.token_embedding(x.long())
        
        # Position embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        # Segment embeddings (default to 0 if not provided)
        if segment_ids is None:
            segment_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=x.device)
        seg_emb = self.segment_embedding(segment_ids)
        
        embeddings = token_emb + pos_emb + seg_emb
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class BERT(nn.Module):
    """
    BERT model compatible with the Transformer interface.
    Uses bidirectional encoder architecture.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        output_activation: Optional[str] = None,
        weight_init: str = 'default',
        max_len: int = 5000,
        vocab_size: Optional[int] = None,
        use_embeddings: bool = False,
        *args,
        **kwargs,
    ):
        """
        Args:
            input_size: Input feature dimension (or vocab size if use_embeddings=True)
            hidden_size: Hidden dimension (d_model)
            output_size: Output dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            output_activation: Activation function for output ('tanh', 'relu', or None)
            weight_init: Weight initialization scheme
            max_len: Maximum sequence length
            vocab_size: Vocabulary size (if different from input_size and using embeddings)
            use_embeddings: Whether to use BERT-style embeddings (token+position+segment)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.weight_init = weight_init
        self.use_embeddings = use_embeddings
        
        # Embedding layer (BERT-style or simple projection)
        if use_embeddings:
            self.embedding = BERTEmbedding(
                vocab_size=vocab_size or input_size,
                d_model=hidden_size,
                max_len=max_len,
                dropout=dropout
            )
        else:
            # Simple linear projection to match Transformer interface
            self.input_proj = nn.Linear(input_size, hidden_size)
            self.embedding = None
        
        # BERT Encoder (bidirectional transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',  # BERT uses GELU
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Pooler for [CLS] token representation (optional, useful for sequence classification)
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        if self.weight_init != 'default':
            self.my_reset_parameters()
    
    def my_reset_parameters(self):
        """Custom weight initialization"""
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
        segment_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass matching the Transformer interface.
        
        Args:
            x: Input tensor - PackedSequence or [batch, time, input_size]
            lengths: Actual lengths of input sequences [batch]
            out_lengths: Lengths of output sequences [batch]
            max_out_length: Maximum output length
            unpack: Whether to unpack (legacy parameter)
            return_hidden_states: Whether to return hidden states
            segment_ids: Segment IDs for BERT embeddings [batch, time] (optional)
            attention_mask: Attention mask [batch, time] (optional)
        
        Returns:
            Output tensor [batch, max_out_length, output_size] or
            Tuple of (sequence_output, pooled_output) if return_hidden_states=True
        """
        # Unpack if PackedSequence (to match Transformer interface)
        if isinstance(x, rnn_utils.PackedSequence):
            unpacked_x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        else:
            unpacked_x = x  # [batch, time, input_size]
        
        # Generate embeddings
        if self.use_embeddings and self.embedding is not None:
            h = self.embedding(unpacked_x, segment_ids)
        else:
            h = self.input_proj(unpacked_x)
        
        # Create attention mask if not provided
        # BERT uses additive mask where -inf masks out positions
        if attention_mask is None:
            batch_size, seq_len = h.size(0), h.size(1)
            time_indices = torch.arange(seq_len, device=h.device).unsqueeze(0)
            attention_mask = (time_indices < lengths.unsqueeze(1)).float()
        
        # Convert mask to BERT format (1 for valid, 0 for masked)
        # TransformerEncoder expects bool mask where True = ignore
        src_key_padding_mask = (attention_mask == 0)
        
        # Encode with BERT
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        
        if return_hidden_states:
            # Return full sequence and pooled [CLS] representation
            pooled = self.pooler(h[:, 0, :])  # Pool first token
            return h, pooled.unsqueeze(0)  # Match expected return format
        
        # Project to output space
        out = self.fc(h)
        
        # Apply output activation
        if self.output_activation == "tanh":
            out = torch.tanh(out)
        elif self.output_activation == "relu":
            out = torch.relu(out)
        
        # Mask outputs: zero positions at or beyond lengths + out_lengths
        time_indices = torch.arange(out.size(1), device=out.device).unsqueeze(0)
        zero_output_mask = time_indices >= (lengths + out_lengths).unsqueeze(1)
        out[zero_output_mask] = 0
        
        # Extract region [lengths, lengths + max_out_length)
        copy_mask = (time_indices >= lengths.unsqueeze(1)) & (
            time_indices < (lengths + max_out_length).unsqueeze(1)
        )
        out = out[copy_mask]
        out = out.view(-1, max_out_length, self.output_size)
        
        return out
    
    def get_pooled_output(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Get pooled representation (useful for sequence classification).
        
        Args:
            x: Input tensor
            lengths: Sequence lengths
            
        Returns:
            Pooled output [batch, hidden_size]
        """
        if isinstance(x, rnn_utils.PackedSequence):
            unpacked_x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        else:
            unpacked_x = x
        
        if self.use_embeddings and self.embedding is not None:
            h = self.embedding(unpacked_x)
        else:
            h = self.input_proj(unpacked_x)
        
        # Create mask
        batch_size, seq_len = h.size(0), h.size(1)
        time_indices = torch.arange(seq_len, device=h.device).unsqueeze(0)
        src_key_padding_mask = (time_indices >= lengths.unsqueeze(1))
        
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        pooled = self.pooler(h[:, 0, :])
        
        return pooled