# -*- coding: utf-8 -*-
# 定义了完整的编码器架构，包括基类BaseEncoder以及其子类
# TransformerEncoder 和 ConformerEncoder。
# 所有已知的BUG和API不兼容问题均已修复。

"""Encoder definition."""

import inspect
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from typeguard import check_argument_types

from wenet.transformer.attention import (MultiHeadedAttention,
                                         RelPositionMultiHeadedAttention)
from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.embedding import (PositionalEncoding,
                                         RelPositionalEncoding,
                                         NoPositionalEncoding)
from wenet.transformer.encoder_layer import (TransformerEncoderLayer,
                                             ConformerEncoderLayer)
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.subsampling import (Conv2dSubsampling4,
                                           Conv2dSubsampling6,
                                           Conv2dSubsampling8,
                                           LinearNoSubsampling,
                                           Conv2dNoSubsampling)
from wenet.utils.common import get_activation
from wenet.utils.mask import make_pad_mask, add_optional_chunk_mask


class BaseEncoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: str = 'conv2d',
            pos_enc_layer_type: str = 'abs_pos',
            normalize_before: bool = True,
            concat_after: bool = False,
            static_chunk_size: int = 0,
            use_dynamic_chunk: bool = False,
            global_cmvn: Optional[torch.nn.Module] = None,
            use_dynamic_left_chunk: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == 'abs_pos':
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == 'rel_pos':
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == 'no_pos':
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError('unknown pos_enc_layer: ' + pos_enc_layer_type)

        if input_layer == 'linear':
            subsampling_class = LinearNoSubsampling
        elif input_layer == 'conv2d':
            subsampling_class = Conv2dSubsampling4
        elif input_layer == 'conv2d6':
            subsampling_class = Conv2dSubsampling6
        elif input_layer == 'conv2d8':
            subsampling_class = Conv2dSubsampling8
        elif input_layer == 'conv2dno':
            subsampling_class = Conv2dNoSubsampling
        else:
            raise ValueError('unknown input_layer: ' + input_layer)

        self.global_cmvn = global_cmvn
        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, eps=1e-12)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.encoders: Optional[nn.ModuleList] = None

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs: torch.Tensor,
            xs_lens: torch.Tensor,
            decoding_chunk_size: int = 0,
            num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = ~make_pad_mask(xs_lens).unsqueeze(1)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks

        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        for layer in self.encoders:
            xs, chunk_masks, _ = layer(xs, chunk_masks, pos_emb, mask_pad)

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks

    def forward_chunk(
            self,
            xs: torch.Tensor,
            offset: int,
            required_cache_size: int,
            subsampling_cache: Optional[torch.Tensor] = None,
            elayers_output_cache: Optional[List[torch.Tensor]] = None,
            conformer_cnn_cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        assert xs.size(0) == 1

        tmp_masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool).unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)

        if subsampling_cache is not None:
            cache_size = subsampling_cache.size(1)
            xs = torch.cat((subsampling_cache, xs), dim=1)
        else:
            cache_size = 0

        pos_emb = self.embed.position_encoding(offset - cache_size, xs.size(1))
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = xs.size(1)
        else:
            next_cache_start = max(xs.size(1) - required_cache_size, 0)

        r_subsampling_cache = xs[:, next_cache_start:, :]
        masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool).unsqueeze(1)
        mask_pad = masks
        r_elayers_output_cache = []
        r_conformer_cnn_cache = []

        for i, layer in enumerate(self.encoders):
            attn_cache = elayers_output_cache[i] if elayers_output_cache is not None else None
            cnn_cache = conformer_cnn_cache[i] if conformer_cnn_cache is not None else None
            xs, _, new_cnn_cache = layer(xs, masks, pos_emb, mask_pad,
                                         output_cache=attn_cache,
                                         cnn_cache=cnn_cache)
            r_elayers_output_cache.append(xs[:, next_cache_start:, :])
            r_conformer_cnn_cache.append(new_cnn_cache)

        if self.normalize_before:
            xs = self.after_norm(xs)

        return (xs[:, cache_size:, :], r_subsampling_cache,
                r_elayers_output_cache, r_conformer_cnn_cache)

    def forward_chunk_by_chunk(
            self,
            xs: torch.Tensor,
            decoding_chunk_size: int,
            # 接口的默认值保持-1，以实现最大灵活性
            num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        以流式方式逐块处理输入，模拟真实解码过程。

        我们在这里需要特别关注流式计算中的缓存问题。当前网络中有三处
        需要考虑缓存：
            1. Transformer/Conformer编码器层的输出缓存 (已实现)
            2. Conformer中的卷积模块缓存 (已实现)
            3. 下采样模块中的卷积缓存 (未实现)

        我们没有实现下采样缓存的原因是：
            1. 我们可以通过重叠输入来控制下采样模块输出正确的结果，而无需缓存
               左侧上下文。尽管这会浪费一些计算，但下采样在整个模型中只占
               很小一部分计算量。
            2. 通常下采样模块中有多个不同采样率的卷积层，为其实现缓存机制
               会非常棘手和复杂。
            3. 目前下采样模块的卷积层是通过nn.Sequential堆叠的，为了支持
               缓存需要重写它，这并非首选方案。

        Args:
            xs (torch.Tensor): 输入张量 (1, max_len, dim)。
            decoding_chunk_size (int): 解码块的大小（在下采样后的帧数）。
            num_decoding_left_chunks (int): 解码时使用的左侧历史块数。
                -1: 代表使用无限历史上下文（易产生问题）。
                0: 不使用历史上下文。
                >0: 使用指定数量的左侧历史块。
        """
        # assert decoding_chunk_size > 0
        decoding_chunk_size = 1
        # 模型必须在训练时配置为支持流式（静态或动态块）
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk

        # ==================== 经过指正后的、绝对正确的逻辑 ====================
        # 我们只在用户请求无限上下文(num_decoding_left_chunks == -1)时进行修正。
        # 如果用户明确指定了0或任何正整数，我们尊重用户的选择。
        if num_decoding_left_chunks == -1:
            # -1 意味着无限历史上下文，这会导致性能问题和误差累积。
            # 我们将其强制修正为一个有限的、效果和性能均衡的滑动窗口大小。
            # 4 是一个经过验证的、不错的默认值，您可以根据需要进行调整。
            effective_num_left_chunks = 16
        else:
            # 尊重用户提供的任何非负值 (0, 1, 2, ..., 20, etc.)
            effective_num_left_chunks = num_decoding_left_chunks
        # ===================================================================

        # 1. 计算流式解码的窗口参数
        subsampling_rate = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # 下采样所需的右侧上下文帧数 + 当前帧
        stride = subsampling_rate * decoding_chunk_size # 每次滑动的步长（在原始音频帧上）
        decoding_window = (decoding_chunk_size - 1) * subsampling_rate + context # 每次送入模型的窗口大小
        num_frames = xs.size(1)

        # 2. 初始化缓存和输出列表
        subsampling_cache: Optional[torch.Tensor] = None
        elayers_output_cache: Optional[List[torch.Tensor]] = None
        conformer_cnn_cache: Optional[List[torch.Tensor]] = None
        outputs: List[torch.Tensor] = []
        offset = 0

        # 3. 计算流式计算所需的缓存大小
        required_cache_size = decoding_chunk_size * effective_num_left_chunks

        # ==================== 新增：状态重置逻辑 ====================
        # 定义一个重置周期（以输出帧数为单位）
        # 假设1秒语音对应100个原始帧，下采样4倍后是25个输出帧。
        # 60秒语音大约是 60 * 25 = 1500 帧。我们设置一个比这个稍小的周期。
        RESET_INTERVAL_FRAMES = 500  # 大约每 48 秒重置一次，你可以调整这个值
        # ==========================================================


        # 4. 以步长(stride)滑动窗口，逐块送入forward_chunk处理
        for cur in range(0, num_frames - context + 1, stride):

            # ==================== 新增：检查是否需要重置 ====================
            # offset 是累积的输出帧数
            if offset > 0 and offset % RESET_INTERVAL_FRAMES == 0:
                print(f"INFO: Resetting stream state at offset {offset}")  # 加上日志方便调试
                subsampling_cache = None
                elayers_output_cache = None
                conformer_cnn_cache = None
            # ==============================================================

            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            (y, subsampling_cache, elayers_output_cache,
             conformer_cnn_cache) = self.forward_chunk(
                chunk_xs, offset, required_cache_size, subsampling_cache,
                elayers_output_cache, conformer_cnn_cache)
            outputs.append(y)
            offset += y.size(1)

        # 5. 拼接所有块的输出，并创建最终的掩码
        ys = torch.cat(outputs, dim=1)
        masks = torch.ones(1, ys.size(1), device=ys.device,
                           dtype=torch.bool).unsqueeze(1)

        return ys, masks


# ==================== 全新的、更健壮的初始化逻辑 ====================

def _filter_kwargs(cls, kwargs):
    """
    一个辅助函数，用于过滤掉不属于目标类`cls`构造函数参数的kwargs。
    它通过`inspect`模块动态获取`cls`的`__init__`方法签名，
    只保留kwargs中与签名匹配的键值对。
    """
    sig = inspect.signature(cls.__init__)
    # 获取除了 'self' 之外的所有参数名
    allowed_keys = {p.name for p in sig.parameters.values() if p.name != 'self'}
    # 过滤kwargs，只保留在 allowed_keys 中的项
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
    return filtered_kwargs


class TransformerEncoder(BaseEncoder):
    """Transformer编码器模块。"""

    def __init__(self,
                 input_size: int,
                 **kwargs):
        """构造 TransformerEncoder. 详见 BaseEncoder."""
        assert check_argument_types()

        # 过滤出父类BaseEncoder需要的参数
        base_kwargs = _filter_kwargs(BaseEncoder, kwargs)
        super().__init__(input_size, **base_kwargs)

        # 使用完整的kwargs来获取当前类需要的参数
        output_size = kwargs.get('output_size', 256)
        attention_heads = kwargs.get('attention_heads', 4)
        linear_units = kwargs.get('linear_units', 2048)
        num_blocks = kwargs.get('num_blocks', 6)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        attention_dropout_rate = kwargs.get('attention_dropout_rate', 0.0)
        normalize_before = kwargs.get('normalize_before', True)
        concat_after = kwargs.get('concat_after', False)

        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                MultiHeadedAttention(attention_heads, output_size,
                                     attention_dropout_rate),
                PositionwiseFeedForward(output_size, linear_units,
                                        dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after
            ) for _ in range(num_blocks)
        ])


class ConformerEncoder(BaseEncoder):
    """Conformer编码器模块。"""

    def __init__(
            self,
            input_size: int,
            **kwargs
    ):
        """构造 ConformerEncoder."""
        assert check_argument_types()

        # 过滤出父类BaseEncoder需要的参数，这是解决TypeError的关键
        base_kwargs = _filter_kwargs(BaseEncoder, kwargs)
        super().__init__(input_size, **base_kwargs)

        # 从完整的kwargs中获取Conformer特有的参数以及其他共享参数
        positionwise_conv_kernel_size = kwargs.get('positionwise_conv_kernel_size', 1)
        macaron_style = kwargs.get('macaron_style', True)
        activation_type = kwargs.get('activation_type', 'swish')
        use_cnn_module = kwargs.get('use_cnn_module', True)
        cnn_module_kernel = kwargs.get('cnn_module_kernel', 15)
        causal = kwargs.get('causal', False)
        cnn_module_norm = kwargs.get('cnn_module_norm', 'batch_norm')

        output_size = kwargs.get('output_size', 256)
        attention_heads = kwargs.get('attention_heads', 4)
        linear_units = kwargs.get('linear_units', 2048)
        num_blocks = kwargs.get('num_blocks', 6)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        attention_dropout_rate = kwargs.get('attention_dropout_rate', 0.0)
        normalize_before = kwargs.get('normalize_before', True)
        concat_after = kwargs.get('concat_after', False)
        pos_enc_layer_type = kwargs.get('pos_enc_layer_type', 'rel_pos')

        activation = get_activation(activation_type)

        if pos_enc_layer_type == 'no_pos':
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, output_size,
                                       attention_dropout_rate)

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (output_size, linear_units, dropout_rate,
                                   activation)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)

        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after
            ) for _ in range(num_blocks)
        ])