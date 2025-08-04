
#file name: /code/landmark2face_wy/models/DINet.py

import torch
from torch import nn
import torch.nn.functional as F
import math, cv2, numpy as np
from landmark2face_wy.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from landmark2face_wy.sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d
from einops import rearrange, repeat

# def make_coordinate_grid_3d_base(spatial_size, type, device):
#     """
#         generate 3D coordinate grid
#     """
#     d, h, w = spatial_size
#     x = torch.arange(w, device=device).type(type)
#     y = torch.arange(h, device=device).type(type)
#     z = torch.arange(d, device=device).type(type)
#     x = 2 * (x / (w - 1)) - 1
#     y = 2 * (y / (h - 1)) - 1
#     z = 2 * (z / (d - 1)) - 1
#     yy = y.view(1, -1, 1).repeat(d, 1, w)
#     xx = x.view(1, 1, -1).repeat(d, h, 1)
#     zz = z.view(-1, 1, 1).repeat(1, h, w)
#     meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
#     return (
#      meshed, zz)

# def make_coordinate_grid_3d_v2(spatial_size, type, device):
#     """
#     generate 3D coordinate grid (ONNX-compatible version)
#     """
#     d, h, w = spatial_size
#
#     # 核心修正：
#     # 1. 使用 feature_map.dtype 直接获取 torch.dtype 对象，而不是 feature_map.type() 返回的字符串。
#     # 2. 在 ONNX 追踪时, d, h, w 可能是 Tensor，所以我们不能直接在 torch.arange 中使用它们。
#     #    但在此特定模型和 PyTorch/ONNX 版本组合下，JIT追踪器可以处理它。
#     #    我们保持原始逻辑，但确保数据类型正确。
#     x = torch.arange(w, device=device).to(device)
#     y = torch.arange(h, device=device).to(device)
#     z = torch.arange(d, device=device).to(device)
#
#     # 归一化
#     x = (2.0 * x / (w - 1)) - 1.0
#     y = (2.0 * y / (h - 1)) - 1.0
#     z = (2.0 * z / (d - 1)) - 1.0
#
#     # 创建网格
#     yy = y.view(1, -1, 1).repeat(d, 1, w)
#     xx = x.view(1, 1, -1).repeat(d, h, 1)
#     zz = z.view(-1, 1, 1).repeat(1, h, w)
#
#     # 核心修正 2: 替换掉不兼容ONNX的原地操作 'unsqueeze_'
#     meshed = torch.cat([xx.unsqueeze(3), yy.unsqueeze(3)], 3)
#
#     return meshed, zz
#


def make_coordinate_grid_3d(spatial_size, dtype, device):
    """
    generate 3D coordinate grid (Final Production & ONNX Runtime Compatible Version)
    """
    d, h, w = spatial_size

    # -------------------- 最终核心修正点 --------------------
    # ONNX Runtime的'Range'算子不支持float16输入。
    # 因此，我们先用一个它支持的类型(如int64)创建arange，
    # 然后再将结果转换为模型所需的最终数据类型(dtype)。
    x = torch.arange(w, device=device, dtype=torch.int64).to(dtype)
    y = torch.arange(h, device=device, dtype=torch.int64).to(dtype)
    z = torch.arange(d, device=device, dtype=torch.int64).to(dtype)
    # --------------------------------------------------------

    # 归一化
    x = (2.0 * x / (w - 1)) - 1.0
    y = (2.0 * y / (h - 1)) - 1.0
    z = (2.0 * z / (d - 1)) - 1.0

    # 创建网格
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)

    meshed = torch.cat([xx.unsqueeze(3), yy.unsqueeze(3)], 3)

    return meshed, zz

class ResBlock1d(nn.Module):
    __doc__ = "\n        basic block\n    "

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(ResBlock1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv1d(in_features, out_features, 1)
        self.norm1 = BatchNorm1d(in_features)
        self.norm2 = BatchNorm1d(in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out


class ResBlock2d(nn.Module):
    __doc__ = "\n            basic block\n    "

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features, out_features, 1)
        self.norm1 = BatchNorm2d(in_features)
        self.norm2 = BatchNorm2d(in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out


class UpBlock2d(nn.Module):
    __doc__ = "\n            basic block\n    "

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock1d(nn.Module):
    __doc__ = "\n            basic block\n    "

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(DownBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding,
          stride=2)
        self.norm = BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class DownBlock2d(nn.Module):
    __doc__ = "\n            basic block\n    "

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding,
          stride=stride)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class SameBlock1d(nn.Module):
    __doc__ = "\n            basic block\n    "

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(SameBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
          padding=padding)
        self.norm = BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class SameBlock2d(nn.Module):
    __doc__ = "\n            basic block\n    "

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
          padding=padding)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class AdaAT(nn.Module):
    __doc__ = "\n       AdaAT operator\n    "

    def __init__(self, para_ch, feature_ch):
        super(AdaAT, self).__init__()
        self.para_ch = para_ch
        self.feature_ch = feature_ch
        self.commn_linear = nn.Sequential(nn.Linear(para_ch, para_ch), nn.ReLU())
        self.scale = nn.Sequential(nn.Linear(para_ch, feature_ch), nn.Sigmoid())
        self.rotation = nn.Sequential(nn.Linear(para_ch, feature_ch), nn.Tanh())
        self.translation = nn.Sequential(nn.Linear(para_ch, 2 * feature_ch), nn.Tanh())
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map, para_code):
        (batch, d, h, w) = (
            feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3))

        # para_code 的计算会继承输入的数据类型 (FP16 or FP32)
        para_code = self.commn_linear(para_code)
        scale = self.scale(para_code).unsqueeze(-1) * 2
        angle = self.rotation(para_code).unsqueeze(-1) * 3.14159

        # rotation_matrix 会是 FP16 或 FP32
        rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], -1)
        translation = self.translation(para_code).view(batch, self.feature_ch, 2)

        # -------------------- 核心修正点 --------------------
        # 调用 make_coordinate_grid_3d 时，直接传递 feature_map 的 dtype
        # 这样 grid_xy 和 grid_z 的类型就与 feature_map 和 rotation_matrix 完全一致
        (grid_xy, grid_z) = make_coordinate_grid_3d((d, h, w), dtype=feature_map.dtype, device=feature_map.device)
        # ----------------------------------------------------

        grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)

        scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)

        rotation_matrix = rearrange(rotation_matrix, "b d e -> b d () () e")
        rotation_matrix = repeat(rotation_matrix, "b d h w e -> b d (h h1) (w w1) e", h1=h, w1=w)

        grid_xy_expand = rearrange(grid_xy, "b d h w (t e) -> b (d h w) t e", e=1)
        rotation_matrix_expand = rearrange(rotation_matrix, "b d h w (t e) -> b (d h w) t e", t=2, e=2)

        # 现在，rotation_matrix_expand 和 grid_xy_expand 的 dtype 完全相同，matmul 不会再报错
        trans_grid = rearrange((torch.matmul(rotation_matrix_expand, grid_xy_expand)), "b (d h w) t e -> b d h w (t e)",
                               d=d,
                               h=h,
                               w=w) * scale + translation

        full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)

        # --- 保持ONNX兼容的 grid_sample 修复 ---
        B, D, H, W = feature_map.shape
        feature_map_reshaped = feature_map.view(B * D, 1, H, W)
        grid_reshaped = full_grid[..., :2].contiguous().view(B * D, H, W, 2)
        trans_feature_reshaped = F.grid_sample(feature_map_reshaped, grid_reshaped, align_corners=False,
                                               padding_mode='border')
        trans_feature = trans_feature_reshaped.view(B, D, H, W)

        return trans_feature

class DINet(nn.Module):

    def __init__(self, source_channel, ref_channel, audio_channel):
        super(DINet, self).__init__()
        self.source_in_conv = nn.Sequential(SameBlock2d(source_channel, 64, kernel_size=7, padding=3), DownBlock2d(64, 128, kernel_size=3, padding=1), DownBlock2d(128, 256, kernel_size=3, padding=1))
        self.ref_in_conv = nn.Sequential(SameBlock2d(ref_channel, 64, kernel_size=7, padding=3), DownBlock2d(64, 128, kernel_size=3, padding=1), DownBlock2d(128, 256, kernel_size=3, padding=1))
        self.trans_conv = nn.Sequential(SameBlock2d(512, 128, kernel_size=3, padding=1), SameBlock2d(128, 128, kernel_size=11, padding=5), SameBlock2d(128, 128, kernel_size=11, padding=5), DownBlock2d(128, 128, kernel_size=3, padding=1), SameBlock2d(128, 128, kernel_size=7, padding=3), SameBlock2d(128, 128, kernel_size=7, padding=3), DownBlock2d(128, 128, kernel_size=3, padding=1), SameBlock2d(128, 128, kernel_size=3, padding=1), DownBlock2d(128, 128, kernel_size=3, padding=1), SameBlock2d(128, 128, kernel_size=3, padding=1), DownBlock2d(128, 128, kernel_size=3, padding=1))
        self.audio_encoder = nn.Sequential(SameBlock1d(audio_channel, 128, kernel_size=5, padding=2), ResBlock1d(128, 128, 3, 1), DownBlock1d(128, 128, 3, 1), ResBlock1d(128, 128, 3, 1), DownBlock1d(128, 128, 3, 1), SameBlock1d(128, 128, kernel_size=3, padding=1))
        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(nn.Sequential(ResBlock2d(256, 256, 3, 1), ResBlock2d(256, 256, 3, 1), ResBlock2d(256, 256, 3, 1), ResBlock2d(256, 256, 3, 1)))

        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(256, 256)
        self.out_conv = nn.Sequential(SameBlock2d(512, 128, kernel_size=3, padding=1), UpBlock2d(128, 128, kernel_size=3, padding=1), ResBlock2d(128, 128, 3, 1), UpBlock2d(128, 128, kernel_size=3, padding=1), nn.Conv2d(128, 3, kernel_size=(7,
                                                                                                                                                                                                                                                7), padding=(3,
                                                                                                                                                                                                                                                             3)), nn.Sigmoid())
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, source_img, ref_img, audio_feature):
        source_in_feature = self.source_in_conv(source_img)
        ref_in_feature = self.ref_in_conv(ref_img)
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        audio_para = self.audio_encoder(audio_feature)
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        trans_para = torch.cat([img_para, audio_para], 1)
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        out = self.out_conv(merge_feature)
        return out

    def forward_analysis(self, source_img, ref_img, audio_feature):
        source_in_feature = self.source_in_conv(source_img)
        torch.save(source_in_feature, "source_in_feature.pt")
        ref_in_feature = self.ref_in_conv(ref_img)
        torch.save(ref_in_feature, "ref_in_feature.pt")
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        audio_para = self.audio_encoder(audio_feature)
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        trans_para = torch.cat([img_para, audio_para], 1)
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        torch.save(ref_trans_feature, "ref_trans_feature.pt")
        exit(0)
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        out = self.out_conv(merge_feature)
        return out


class DINetMouth(nn.Module):

    def __init__(self, source_channel, ref_channel, mouth_ref_channel, audio_channel):
        super(DINetMouth, self).__init__()
        self.source_in_conv = nn.Sequential(SameBlock2d(source_channel, 64, kernel_size=7, padding=3), DownBlock2d(64, 128, kernel_size=3, padding=1), DownBlock2d(128, 256, kernel_size=3, padding=1))
        self.ref_in_conv = nn.Sequential(SameBlock2d(ref_channel, 64, kernel_size=7, padding=3), DownBlock2d(64, 128, kernel_size=3, padding=1), DownBlock2d(128, 256, kernel_size=3, padding=1))
        self.ref_in_conv2 = nn.Sequential(SameBlock2d(mouth_ref_channel, 64, kernel_size=7, padding=3), DownBlock2d(64, 128, kernel_size=3, padding=1), DownBlock2d(128, 256, kernel_size=3, padding=1))
        self.trans_conv = nn.Sequential(SameBlock2d(768, 128, kernel_size=3, padding=1), SameBlock2d(128, 128, kernel_size=11, padding=5), SameBlock2d(128, 128, kernel_size=11, padding=5), DownBlock2d(128, 128, kernel_size=3, padding=1), SameBlock2d(128, 128, kernel_size=7, padding=3), SameBlock2d(128, 128, kernel_size=7, padding=3), DownBlock2d(128, 128, kernel_size=3, padding=1), SameBlock2d(128, 128, kernel_size=3, padding=1), DownBlock2d(128, 128, kernel_size=3, padding=1), SameBlock2d(128, 128, kernel_size=3, padding=1), DownBlock2d(128, 128, kernel_size=3, padding=1))
        self.audio_encoder = nn.Sequential(SameBlock1d(audio_channel, 128, kernel_size=5, padding=2), ResBlock1d(128, 128, 3, 1), DownBlock1d(128, 128, 3, 1), ResBlock1d(128, 128, 3, 1), DownBlock1d(128, 128, 3, 1), SameBlock1d(128, 128, kernel_size=3, padding=1))
        appearance_conv_list = list()
        appearance_conv_list.append(nn.Sequential(SameBlock2d(512, 256, kernel_size=5, padding=2), ResBlock2d(256, 256, 3, 1), ResBlock2d(256, 256, 3, 1), ResBlock2d(256, 256, 3, 1), ResBlock2d(256, 256, 3, 1)))
        appearance_conv_list.append(nn.Sequential(ResBlock2d(256, 256, 3, 1), ResBlock2d(256, 256, 3, 1), ResBlock2d(256, 256, 3, 1), ResBlock2d(256, 256, 3, 1)))
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(256, 256)
        self.out_conv = nn.Sequential(SameBlock2d(512, 128, kernel_size=3, padding=1), UpBlock2d(128, 128, kernel_size=3, padding=1), ResBlock2d(128, 128, 3, 1), UpBlock2d(128, 128, kernel_size=3, padding=1), nn.Conv2d(128, 3, kernel_size=(7,
                                                                                                                                                                                                                                                7), padding=(3,
                                                                                                                                                                                                                                                             3)), nn.Sigmoid())
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, source_img, ref_img, ref_mouth_img, audio_feature):
        source_in_feature = self.source_in_conv(source_img)
        ref_in_feature = self.ref_in_conv(ref_img)
        ref_mouth_in_feature = self.ref_in_conv2(ref_mouth_img)
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature, ref_mouth_in_feature], 1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        audio_para = self.audio_encoder(audio_feature)
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        trans_para = torch.cat([img_para, audio_para], 1)
        final_ref_infeature = torch.cat([ref_in_feature, ref_mouth_in_feature], 1)
        ref_trans_feature = self.appearance_conv_list[0](final_ref_infeature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        out = self.out_conv(merge_feature)
        return out

    def forward_analysis(self, source_img, ref_img, audio_feature):
        source_in_feature = self.source_in_conv(source_img)
        torch.save(source_in_feature, "source_in_feature.pt")
        ref_in_feature = self.ref_in_conv(ref_img)
        torch.save(ref_in_feature, "ref_in_feature.pt")
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        audio_para = self.audio_encoder(audio_feature)
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        trans_para = torch.cat([img_para, audio_para], 1)
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        torch.save(ref_trans_feature, "ref_trans_feature.pt")
        exit(0)
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        out = self.out_conv(merge_feature)
        return out


class DINetV1(nn.Module):

    def __init__(self, source_channel, ref_channel, audio_channel):
        super(DINetV1, self).__init__()
        self.source_in_conv = nn.ModuleList([
         nn.Sequential(SameBlock2d(source_channel, 64, kernel_size=7, padding=3), DownBlock2d(64, 128, kernel_size=3, padding=1)),
         nn.Sequential(SameBlock2d(128, 128, kernel_size=3, padding=1), DownBlock2d(128, 256, kernel_size=3, padding=1)),
         nn.Sequential(SameBlock2d(256, 256, kernel_size=3, padding=1), DownBlock2d(256, 512, kernel_size=3, padding=1))])
        self.ref_in_conv = nn.ModuleList([
         nn.Sequential(SameBlock2d(ref_channel, 64, kernel_size=7, padding=3), DownBlock2d(64, 128, kernel_size=3, padding=1)),
         nn.Sequential(SameBlock2d(128, 128, kernel_size=3, padding=1), DownBlock2d(128, 256, kernel_size=3, padding=1)),
         nn.Sequential(SameBlock2d(256, 256, kernel_size=3, padding=1), DownBlock2d(256, 512, kernel_size=3, padding=1))])
        self.trans_conv = nn.Sequential(SameBlock2d(1024, 256, kernel_size=3, padding=1), SameBlock2d(256, 256, kernel_size=11, padding=5), SameBlock2d(256, 256, kernel_size=11, padding=5), DownBlock2d(256, 256, kernel_size=3, padding=1), SameBlock2d(256, 256, kernel_size=7, padding=3), SameBlock2d(256, 256, kernel_size=7, padding=3), DownBlock2d(256, 256, kernel_size=3, padding=1), SameBlock2d(256, 256, kernel_size=5, padding=2), DownBlock2d(256, 256, kernel_size=5, padding=2), SameBlock2d(256, 256, kernel_size=3, padding=1), DownBlock2d(256, 256, kernel_size=3, padding=1))
        self.audio_encoder = nn.Sequential(SameBlock1d(audio_channel, 256, kernel_size=5, padding=2), ResBlock1d(256, 256, 3, 1), DownBlock1d(256, 256, 3, 1), ResBlock1d(256, 256, 3, 1), DownBlock1d(256, 256, 3, 1), SameBlock1d(256, 256, kernel_size=3, padding=1))
        appearance_conv_list = list()
        appearance_conv_list.append(nn.Sequential(SameBlock2d(512, 512, kernel_size=5, padding=2), ResBlock2d(512, 512, 3, 1), ResBlock2d(512, 512, 3, 1), ResBlock2d(512, 512, 3, 1), ResBlock2d(512, 512, 3, 1)))
        appearance_conv_list.append(nn.Sequential(ResBlock2d(512, 512, 3, 1), ResBlock2d(512, 512, 3, 1), ResBlock2d(512, 512, 3, 1), ResBlock2d(512, 512, 3, 1)))
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(512, 512)
        self.out_conv = nn.ModuleList([
         nn.Sequential(SameBlock2d(1024, 512, kernel_size=3, padding=1), UpBlock2d(512, 512, kernel_size=3, padding=1), ResBlock2d(512, 512, 3, 1)),
         nn.Sequential(UpBlock2d(768, 256, kernel_size=3, padding=1), ResBlock2d(256, 256, 3, 1)),
         nn.Sequential(UpBlock2d(384, 128, kernel_size=3, padding=1), ResBlock2d(128, 128, 3, 1), nn.Conv2d(128, 3, kernel_size=(7,
                                                                                                                        7), padding=(3,
                                                                                                                                     3)), nn.Sigmoid())])
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, source_img, ref_img, audio_feature):
        feats_source = []
        x = source_img
        for f in self.source_in_conv:
            x = f(x)
            feats_source.append(x)

        feats_ref = []
        x = ref_img
        count = 0
        for f in self.ref_in_conv:
            x = f(x)
            if count == 2:
                feats_ref.append(x)
            count += 1

        source_in_feature = feats_source.pop()
        ref_in_feature = feats_ref.pop()
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        audio_para = self.audio_encoder(audio_feature) * 1.5
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        trans_para = torch.cat([img_para, audio_para], 1)
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        out = self.out_conv[0](merge_feature)
        out = self.out_conv[1](torch.cat([out, feats_source.pop()], 1))
        out = self.out_conv[2](torch.cat([out, feats_source.pop()], 1))
        return out


class DINetMouth512(nn.Module):

    def __init__(self, source_channel, ref_channel, audio_channel, device=None):
        super(DINetMouth512, self).__init__()
        self.source_in_conv = nn.ModuleList([
         nn.Sequential(SameBlock2d(source_channel, 64, kernel_size=7, padding=3), DownBlock2d(64, 128, kernel_size=3, padding=1)),
         nn.Sequential(SameBlock2d(128, 128, kernel_size=3, padding=1), DownBlock2d(128, 256, kernel_size=3, padding=1)),
         nn.Sequential(SameBlock2d(256, 256, kernel_size=3, padding=1), DownBlock2d(256, 512, kernel_size=3, padding=1), DownBlock2d(512, 512, kernel_size=3, padding=1))])
        self.ref_in_conv = nn.ModuleList([
         nn.Sequential(SameBlock2d(ref_channel, 64, kernel_size=7, padding=3), DownBlock2d(64, 128, kernel_size=3, padding=1)),
         nn.Sequential(SameBlock2d(128, 128, kernel_size=3, padding=1), DownBlock2d(128, 256, kernel_size=3, padding=1)),
         nn.Sequential(SameBlock2d(256, 256, kernel_size=3, padding=1), DownBlock2d(256, 512, kernel_size=3, padding=1), DownBlock2d(512, 512, kernel_size=3, padding=1))])
        self.trans_conv = nn.Sequential(SameBlock2d(1024, 256, kernel_size=3, padding=1), SameBlock2d(256, 256, kernel_size=11, padding=5), SameBlock2d(256, 256, kernel_size=11, padding=5), DownBlock2d(256, 256, kernel_size=3, padding=1), SameBlock2d(256, 256, kernel_size=7, padding=3), SameBlock2d(256, 256, kernel_size=7, padding=3), DownBlock2d(256, 256, kernel_size=3, padding=1), SameBlock2d(256, 256, kernel_size=5, padding=2), DownBlock2d(256, 256, kernel_size=5, padding=2), SameBlock2d(256, 256, kernel_size=3, padding=1), DownBlock2d(256, 256, kernel_size=3, padding=1))
        self.audio_encoder = nn.Sequential(SameBlock1d(audio_channel, 256, kernel_size=5, padding=2), ResBlock1d(256, 256, 3, 1), DownBlock1d(256, 256, 3, 1), ResBlock1d(256, 256, 3, 1), DownBlock1d(256, 256, 3, 1), SameBlock1d(256, 256, kernel_size=3, padding=1))
        appearance_conv_list = list()
        appearance_conv_list.append(nn.Sequential(SameBlock2d(512, 512, kernel_size=5, padding=2), ResBlock2d(512, 512, 3, 1), ResBlock2d(512, 512, 3, 1), ResBlock2d(512, 512, 3, 1), ResBlock2d(512, 512, 3, 1)))
        appearance_conv_list.append(nn.Sequential(ResBlock2d(512, 512, 3, 1), ResBlock2d(512, 512, 3, 1), ResBlock2d(512, 512, 3, 1), ResBlock2d(512, 512, 3, 1)))
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(512, 512, device)
        self.out_conv = nn.ModuleList([
         nn.Sequential(SameBlock2d(1024, 512, kernel_size=3, padding=1), UpBlock2d(512, 512, kernel_size=3, padding=1), UpBlock2d(512, 512, kernel_size=3, padding=1), ResBlock2d(512, 512, 3, 1)),
         nn.Sequential(UpBlock2d(768, 256, kernel_size=3, padding=1), ResBlock2d(256, 256, 3, 1)),
         nn.Sequential(UpBlock2d(384, 128, kernel_size=3, padding=1), ResBlock2d(128, 128, 3, 1), nn.Conv2d(128, 3, kernel_size=(7,
                                                                                                                        7), padding=(3,
                                                                                                                                     3)), nn.Sigmoid())])
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, source_img, ref_img, audio_feature):
        feats_source = []
        x = source_img
        for f in self.source_in_conv:
            x = f(x)
            feats_source.append(x)

        feats_ref = []
        x = ref_img
        count = 0
        for f in self.ref_in_conv:
            x = f(x)
            if count == 2:
                feats_ref.append(x)
            count += 1

        source_in_feature = feats_source.pop()
        ref_in_feature = feats_ref.pop()
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        audio_para = self.audio_encoder(audio_feature) * 2
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        trans_para = torch.cat([img_para, audio_para], 1)
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        out = self.out_conv[0](merge_feature)
        out = self.out_conv[1](torch.cat([out, feats_source.pop()], 1))
        out = self.out_conv[2](torch.cat([out, feats_source.pop()], 1))
        return out


if __name__ == "__main__":
    dinet_model = DINetMouth512(3, 3, 256)
    dinet_model.cuda()
    ref_face = torch.randn(4, 3, 256, 256)
    audio_feature = torch.randn(4, 256, 20)
    print(dinet_model(ref_face.cuda(), ref_face.cuda(), audio_feature.cuda()).size())

