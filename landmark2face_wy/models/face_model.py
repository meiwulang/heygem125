# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/landmark2face_wy/models/face_model.py
# Compiled at: 2024-03-06 17:51:22
# Size of source mod 2**32: 5724 bytes
import functools, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from landmark2face_wy.util import flow_util
from .base_function import LayerNorm2d, ADAINHourglass, FineEncoder, FineDecoder

class FaceGenerator(nn.Module):

    def __init__(self, mapping_net, warpping_net, editing_net, common):
        super(FaceGenerator, self).__init__()
        self.mapping_net = MappingNet(**mapping_net)
        self.warpping_net = WarpingNet(**warpping_net, **common)
        self.editing_net = EditingNet(**editing_net, **common)

    def forward(self, input_image, driving_source, stage=None):
        if stage == "warp":
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
        else:
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
            output["fake_image"] = self.editing_net(input_image, output["warp_image"], descriptor)
        return output


class PirenderGenerator(nn.Module):

    def __init__(self, mapping_net, editing_net, common):
        super(PirenderGenerator, self).__init__()
        self.mapping_net = MappingNet(**mapping_net)
        self.editing_net = EditingNet(**editing_net, **common)

    def forward(self, ref_img, mask_img, driving_source):
        descriptor = self.mapping_net(driving_source)
        return self.editing_net(ref_img, mask_img, descriptor)


class MappingNet(nn.Module):

    def __init__(self, coeff_nc, descriptor_nc, layer):
        super(MappingNet, self).__init__()
        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)
        self.first = nn.Sequential(torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))
        for i in range(layer):
            net = nn.Sequential(nonlinearity, torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, "encoder" + str(i), net)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, "encoder" + str(i))
            out = model(out) + out[:, :, 3:-3]

        out = self.pooling(out)
        return out


class WarpingNet(nn.Module):

    def __init__(self, image_nc, descriptor_nc, base_nc, max_nc, encoder_layer, decoder_layer, use_spect):
        super(WarpingNet, self).__init__()
        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True)
        kwargs = {'nonlinearity':nonlinearity,  'use_spect':use_spect}
        self.descriptor_nc = descriptor_nc
        self.hourglass = ADAINHourglass(image_nc, (self.descriptor_nc), base_nc, 
         max_nc, encoder_layer, decoder_layer, **kwargs)
        self.flow_out = nn.Sequential(norm_layer(self.hourglass.output_nc), nonlinearity, nn.Conv2d((self.hourglass.output_nc), 2, kernel_size=7, stride=1, padding=3))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_image, descriptor):
        final_output = {}
        output = self.hourglass(input_image, descriptor)
        final_output["flow_field"] = self.flow_out(output)
        deformation = flow_util.convert_flow_to_deformation(final_output["flow_field"])
        final_output["warp_image"] = flow_util.warp_image(input_image, deformation)
        return final_output


class EditingNet(nn.Module):

    def __init__(self, image_nc, descriptor_nc, layer, base_nc, max_nc, num_res_blocks, use_spect):
        super(EditingNet, self).__init__()
        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True)
        kwargs = {'norm_layer':norm_layer,  'nonlinearity':nonlinearity,  'use_spect':use_spect}
        self.descriptor_nc = descriptor_nc
        self.encoder = FineEncoder((image_nc * 2), base_nc, max_nc, layer, **kwargs)
        self.decoder = FineDecoder(image_nc, (self.descriptor_nc), base_nc, max_nc, layer, num_res_blocks, **kwargs)

    def forward(self, input_image, warp_image, descriptor):
        x = torch.cat([input_image, warp_image], 1)
        x = self.encoder(x)
        gen_image = self.decoder(x, descriptor)
        return gen_image


if __name__ == "__main__":
    g = PirenderGenerator({'coeff_nc':64,  'descriptor_nc':256,  'layer':3}, {'layer':3, 
     'num_res_blocks':2,  'base_nc':64}, {
      'image_nc': 3, 'descriptor_nc': 256, 'max_nc': 256, 'use_spect': False})
    semantic_data = torch.randn(2, 64, 27)
    img = torch.randn(2, 3, 512, 512)
    img1 = torch.randn(2, 3, 512, 512)
    output = g(img, img1, semantic_data)
    print(output.size())

