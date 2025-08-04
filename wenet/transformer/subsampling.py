# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/wenet/transformer/subsampling.py
# Compiled at: 2024-03-06 17:51:40
# Size of source mod 2**32: 11606 bytes
"""Subsampling layer definition."""
from typing import Tuple
import torch

class BaseSubsampling(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    __doc__ = "Linear transform the input without subsampling\n\n    Args:\n        idim (int): Input dimension.\n        odim (int): Output dimension.\n        dropout_rate (float): Dropout rate.\n\n    "

    def __init__(self, idim, odim, dropout_rate, pos_enc_class):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(torch.nn.Linear(idim, odim), torch.nn.LayerNorm(odim, eps=1e-12), torch.nn.Dropout(dropout_rate))
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, offset: int=0) -> Tuple[(torch.Tensor, torch.Tensor, torch.Tensor)]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        (x, pos_emb) = self.pos_enc(x, offset)
        return (
         x, pos_emb, x_mask)


class Conv2dNoSubsampling(BaseSubsampling):
    __doc__ = "Convolutional 2D subsampling (to same length).\n\n    Args:\n        idim (int): Input dimension.\n        odim (int): Output dimension.\n        dropout_rate (float): Dropout rate.\n\n    "

    def __init__(self, idim, odim, dropout_rate, pos_enc_class):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, odim, kernel_size=5, stride=1, padding=2), torch.nn.ReLU(), torch.nn.Conv2d(odim, odim, kernel_size=5, stride=1, padding=2), torch.nn.ReLU())
        self.out = torch.nn.Sequential(torch.nn.Linear(odim * idim, odim))
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 1
        self.right_context = 0

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, offset: int=0) -> Tuple[(torch.Tensor, torch.Tensor, torch.Tensor)]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        (b, c, t, f) = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        (x, pos_emb) = self.pos_enc(x, offset)
        return (
         x, pos_emb, x_mask)


class Conv2dSubsampling4(BaseSubsampling):
    __doc__ = "Convolutional 2D subsampling (to 1/4 length).\n\n    Args:\n        idim (int): Input dimension.\n        odim (int): Output dimension.\n        dropout_rate (float): Dropout rate.\n\n    "

    def __init__(self, idim, odim, dropout_rate, pos_enc_class):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, odim, 3, 2), torch.nn.ReLU(), torch.nn.Conv2d(odim, odim, 3, 2), torch.nn.ReLU())
        self.out = torch.nn.Sequential(torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 4
        self.right_context = 6

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, offset: int=0) -> Tuple[(torch.Tensor, torch.Tensor, torch.Tensor)]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        (b, c, t, f) = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        (x, pos_emb) = self.pos_enc(x, offset)
        return (
         x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2])


class Conv2dSubsampling2(BaseSubsampling):
    __doc__ = "Convolutional 2D subsampling (to 1/4 length).\n\n    Args:\n        idim (int): Input dimension.\n        odim (int): Output dimension.\n        dropout_rate (float): Dropout rate.\n\n    "

    def __init__(self, idim, odim, dropout_rate, pos_enc_class):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, odim, 3, 2), torch.nn.ReLU(), torch.nn.Conv2d(odim, odim, 3, 2), torch.nn.ReLU())
        self.out = torch.nn.Sequential(torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 4
        self.right_context = 6

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, offset: int=0) -> Tuple[(torch.Tensor, torch.Tensor, torch.Tensor)]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        (b, c, t, f) = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        (x, pos_emb) = self.pos_enc(x, offset)
        return (
         x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2])


class Conv2dSubsampling6(BaseSubsampling):
    __doc__ = "Convolutional 2D subsampling (to 1/6 length).\n    Args:\n        idim (int): Input dimension.\n        odim (int): Output dimension.\n        dropout_rate (float): Dropout rate.\n        pos_enc (torch.nn.Module): Custom position encoding layer.\n    "

    def __init__(self, idim, odim, dropout_rate, pos_enc_class):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, odim, 3, 2), torch.nn.ReLU(), torch.nn.Conv2d(odim, odim, 5, 3), torch.nn.ReLU())
        self.linear = torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 6
        self.right_context = 14

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, offset: int=0) -> Tuple[(torch.Tensor, torch.Tensor, torch.Tensor)]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        (b, c, t, f) = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        (x, pos_emb) = self.pos_enc(x, offset)
        return (
         x, pos_emb, x_mask[:, :, :-2:2][:, :, :-4:3])


class Conv2dSubsampling8(BaseSubsampling):
    __doc__ = "Convolutional 2D subsampling (to 1/8 length).\n\n    Args:\n        idim (int): Input dimension.\n        odim (int): Output dimension.\n        dropout_rate (float): Dropout rate.\n\n    "

    def __init__(self, idim, odim, dropout_rate, pos_enc_class):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, odim, 3, 2), torch.nn.ReLU(), torch.nn.Conv2d(odim, odim, 3, 2), torch.nn.ReLU(), torch.nn.Conv2d(odim, odim, 3, 2), torch.nn.ReLU())
        self.linear = torch.nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        self.right_context = 14

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, offset: int=0) -> Tuple[(torch.Tensor, torch.Tensor, torch.Tensor)]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        (b, c, t, f) = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        (x, pos_emb) = self.pos_enc(x, offset)
        return (
         x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2])

