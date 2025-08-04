# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/wenet/transformer/positionwise_feed_forward.py
# Compiled at: 2024-03-06 17:51:40
# Size of source mod 2**32: 1399 bytes
"""Positionwise feed forward layer definition."""
import torch

class PositionwiseFeedForward(torch.nn.Module):
    __doc__ = "Positionwise feed forward layer.\n\n    FeedForward are appied on each position of the sequence.\n    The output dim is same with the input dim.\n\n    Args:\n        idim (int): Input dimenstion.\n        hidden_units (int): The number of hidden units.\n        dropout_rate (float): Dropout rate.\n        activation (torch.nn.Module): Activation function\n    "

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))

