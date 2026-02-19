## tranformer.py

import torch
from torch import nn
import math
from typing import Optional, Union, Tuple
import torch.nn.utils.rnn as rnn_utils


class PositionalEncoding(nn.Module):
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
        # x is [batch, time, d_model] with batch_first
        # pe is [max_len, 1, d_model] after unsqueeze+transpose; adapt indexing
        pe = self.pe[:x.size(1), :]
        x = x + pe.squeeze(1)
        return self.dropout(x)


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
        *args,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # use as d_model
        self.output_size = output_size
        self.output_activation = output_activation
        self.weight_init = weight_init

        # Project one-hot or dense inputs to d_model
        self.input_proj = nn.Linear(input_size, hidden_size)
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
                # Only apply Kaiming/Xavier to weights with at least 2 dimensions
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
                    # For 1D weights (embeddings, etc.), use uniform initialization
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
        # x is a PackedSequence (matching RNN/LSTM usage); unpack to dense
        if isinstance(x, rnn_utils.PackedSequence):
            unpacked_x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        else:
            unpacked_x = x  # assume already [batch, time, input_size]

        # Project inputs to model dimension and add positional encoding
        h = self.input_proj(unpacked_x)
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)

        if return_hidden_states:
            # Return sequence outputs and a placeholder hidden tensor
            return h, torch.zeros(1, h.size(0), h.size(2), device=h.device)

        out = self.fc(h)
        if self.output_activation == "tanh":
            out = nn.functional.tanh(out)
        elif self.output_activation == "relu":
            out = nn.functional.relu(out)

        # Mask: zero positions at or beyond lengths + out_lengths
        time_indices = torch.arange(out.size(1), device=out.device).unsqueeze(0)
        zero_output_mask = time_indices >= (lengths + out_lengths).unsqueeze(1)
        out[zero_output_mask] = 0

        # Extract region [lengths, lengths + max_out_length)
        copy_mask = (time_indices >= lengths.unsqueeze(1)) & (
            time_indices < (lengths + max_out_length).unsqueeze(1)
        )
        out = out[copy_mask]
        out = out.view(-1, max_out_length, out.size(1))

        return out

