import torch
from torch import nn
from typing import Union, Tuple
import torch.nn.utils.rnn as rnn_utils

# lstm
class LSTM(nn.LSTM):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, nonlinearity="tanh", output_activation=None, dropout: float = 0.0, weight_init = 'default', *args, **kwargs):
        super().__init__(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_activation = output_activation
        if self.weight_init != 'default':
            self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if self.weight_init in ['kaiming_normal', 'he_normal']:
                    nn.init.kaiming_normal_(param)
                elif self.weight_init in ['kaiming_uniform', 'he_uniform']:
                    nn.init.kaiming_uniform_(param)
                elif self.weight_init in ['xavier_normal', 'glorot_normal']:
                    nn.init.xavier_normal_(param)
                elif self.weight_init in ['xavier_uniform', 'glorot_uniform']:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 3e-5)  # or some small constant value

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, out_lengths:torch.Tensor, max_out_length, unpack: bool = None, return_hidden_states=False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out, hidden = super().forward(x)
        # unpack out
        unpacked_out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        if return_hidden_states:
            return out, hidden
        else:
            out = self.fc(unpacked_out)
            if self.output_activation == "tanh":
                out = nn.functional.tanh(out)
            elif self.output_activation == "relu":
                out = nn.functional.relu(out)
            #out = self.fc(hidden[-1])
            # create a mask to set every value after lengths + out_lengths to 0
            zero_output_mask = torch.arange(out.size(1), device=out.device).unsqueeze(0) >= (lengths + out_lengths).unsqueeze(1)
            out[zero_output_mask] = 0
            # create a copy mask to copy the values between lengths to lenghts + max_out_length
            copy_mask = (torch.arange(out.size(1), device=out.device).unsqueeze(0) >= lengths.unsqueeze(1)) & (torch.arange(out.size(1), device=out.device).unsqueeze(0) < (lengths + max_out_length).unsqueeze(1))
            # use the mask to get only the relevant entries from output
            out = out[copy_mask]
            # reshape output by (n,m) to (-1,max_output_length,m)
            out = out.view(-1, max_out_length, out.size(1))
        return out