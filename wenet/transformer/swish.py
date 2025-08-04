# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/wenet/transformer/swish.py
# Compiled at: 2024-03-06 17:51:40
# Size of source mod 2**32: 511 bytes
"""Swish() activation function for Conformer."""
import torch

class Swish(torch.nn.Module):
    __doc__ = "Construct an Swish object."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)

