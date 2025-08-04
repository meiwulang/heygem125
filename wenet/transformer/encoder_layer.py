# -*- coding: utf-8 -*-
# Copyright 2020 Mobvoi Inc. All Rights Reserved.
# Author: binbin.zhang@mobvoi.com (Binbin Zhang); di.wu@mobvoi.com (Di Wu)
#
# 本文件基于多版本PYC反编译结果整合、分析和重构而成。
# 它定义了构成编码器的基础模块：TransformerEncoderLayer 和 ConformerEncoderLayer。
# 核心逻辑已在各版本间交叉验证，确保了正确性和健壮性。

"""Encoder self-attention layer definition."""
from typing import Optional, Tuple

import torch
from torch import nn


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层模块。

    该模块是构成Transformer编码器的基本单元。它包含两个主要的子层：
    一个多头自注意力（Multi-Head Self-Attention）层和一个位置前馈网络（Position-wise Feed-Forward）。
    在每个子层之后都应用了残差连接和层归一化（Layer Normalization）。

    Args:
        size (int): 输入和输出的维度。
        self_attn (torch.nn.Module): 自注意力模块实例。
        feed_forward (torch.nn.Module): 前馈网络模块实例。
        dropout_rate (float): Dropout的比率。
        normalize_before (bool): 是否在每个子层前进行层归一化 (Pre-LN)。
        concat_after (bool): 是否将注意力层的输入和输出拼接后再进行线性变换。
    """

    def __init__(self,
                 size: int,
                 self_attn: nn.Module,
                 feed_forward: nn.Module,
                 dropout_rate: float,
                 normalize_before: bool = True,
                 concat_after: bool = False):
        """构造一个 TransformerEncoderLayer 对象。"""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear = nn.Linear(size + size, size)

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: torch.Tensor,
            output_cache: Optional[torch.Tensor] = None,
            cnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算编码后的特征。

        Args:
            x (torch.Tensor): 输入张量 (B, T, D)。
            mask (torch.Tensor): 应用于注意力分数的掩码 (B, T, T)。
            pos_emb (torch.Tensor): 位置编码张量。仅为与ConformerEncoderLayer接口兼容。
            mask_pad (torch.Tensor): 填充掩码。仅为与ConformerEncoderLayer接口兼容。
            output_cache (torch.Tensor): 流式解码时使用的注意力历史缓存 (B, T_cache, D)。
            cnn_cache (torch.Tensor): CNN缓存。仅为与ConformerEncoderLayer接口兼容。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - torch.Tensor: 输出张量 (B, T, D)。
                - torch.Tensor: 注意力掩码 (B, T, T)。
                - torch.Tensor: 假的CNN缓存，用于保持API统一。
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        x_q = x
        if output_cache is not None:
            assert output_cache.size(0) == x.size(0)
            assert output_cache.size(2) == self.size
            assert output_cache.size(1) < x.size(1)
            chunk = x.size(1) - output_cache.size(1)
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if output_cache is not None:
            x = torch.cat([output_cache, x], dim=1)

        fake_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        return x, mask, fake_cnn_cache


class ConformerEncoderLayer(nn.Module):
    """Conformer编码器层模块。

    严格按照论文中的顺序实现：(1/2 FFN -> MHA -> Conv -> 1/2 FFN) -> LayerNorm

    Args:
        size (int): 输入维度。
        self_attn (nn.Module): 自注意力模块实例。
        feed_forward (nn.Module): 主前馈网络模块实例。
        feed_forward_macaron (nn.Module): macaron-style的前馈网络模块实例。
        conv_module (nn.Module): 卷积模块实例。
        dropout_rate (float): Dropout比率。
        normalize_before (bool): 是否使用Pre-LN结构。
        concat_after (bool): 是否拼接注意力输入和输出。
    """

    def __init__(self,
                 size: int,
                 self_attn: nn.Module,
                 feed_forward: Optional[nn.Module] = None,
                 feed_forward_macaron: Optional[nn.Module] = None,
                 conv_module: Optional[nn.Module] = None,
                 dropout_rate: float = 0.1,
                 normalize_before: bool = True,
                 concat_after: bool = False):
        """构造一个 ConformerEncoderLayer 对象。"""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module

        self.norm_mha = nn.LayerNorm(size, eps=1e-12)
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0

        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)
            self.norm_final = nn.LayerNorm(size, eps=1e-12)

        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear = nn.Linear(size + size, size)

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: torch.Tensor,
            output_cache: Optional[torch.Tensor] = None,
            cnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算编码后的特征, 严格遵循Conformer的模块顺序。

        Args:
            x (torch.Tensor): 输入张量 (B, T, D)。
            mask (torch.Tensor): 应用于注意力分数的掩码 (B, T, T)。
            pos_emb (torch.Tensor): 位置编码张量，对于Conformer必须提供。
            mask_pad (torch.Tensor): 批次填充掩码，用于卷积模块 (B, 1, T)。
            output_cache (torch.Tensor): 注意力历史缓存 (B, T_cache, D)。
            cnn_cache (torch.Tensor): 卷积历史缓存。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - torch.Tensor: 输出张量 (B, T, D)。
                - torch.Tensor: 注意力掩码 (B, T, T)。
                - torch.Tensor: 新的CNN缓存。
        """
        # 1. Macaron-style Feed Forward (the first half)
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # 2. Multi-headed Self-Attention
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        x_q = x
        if output_cache is not None:
            assert output_cache.size(0) == x.size(0)
            assert output_cache.size(2) == self.size
            assert output_cache.size(1) < x.size(1)
            chunk = x.size(1) - output_cache.size(1)
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        x_att = self.self_attn(x_q, x, x, mask, pos_emb)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)

        if not self.normalize_before:
            x = self.norm_mha(x)

        # 3. Convolution Module
        new_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)

        # 4. Feed Forward (the second half)
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        # 5. Final Layer Norm
        if self.conv_module is not None:
            x = self.norm_final(x)

        if output_cache is not None:
            x = torch.cat([output_cache, x], dim=1)

        return x, mask, new_cnn_cache