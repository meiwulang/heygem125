
import time, torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from .base_function import ADAIN
from .face_model import *

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight'):
        if 'Conv' in classname or 'Linear' in classname:
            nn.init.normal_(m.weight.data, 0, 0.02)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1, 0.02)
            nn.init.constant_(m.bias.data, 0)

def get_norm_layer(norm_type='instance'):
    if norm_type == "batch":
        norm_layer = functools.partial((nn.BatchNorm2d), affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial((nn.InstanceNorm2d), affine=False)
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=4, n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3, norm="instance", gpu_ids=[], apex=False):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == "global":
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == "wenet":
        netG = GlobalGeneratorwenet(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == "globalaudio":
        netG = GlobalGeneratoraudio(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == "globalaudio2":
        netG = GlobalGeneratoraudio2(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == "global256to512audio":
        netG = Global256to512Generatoraudio(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == "AFRSmall":
        netG = AFRSmall(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == "AFR":
        netG = AFR(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == "local":
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == "encoder":
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    elif netG == "pirender":
        netG = PirenderGenerator({'coeff_nc':73,  'descriptor_nc':256,  'layer':3}, {'layer':3, 
         'num_res_blocks':2,  'base_nc':64}, {
          'image_nc': 3, 'descriptor_nc': 256, 'max_nc': 256, 'use_spect': False})
    elif netG == "pirenderhd":
        netG = GlobalGeneratorPirender(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == "pirenderhdv2":
        netG = GlobalGeneratorPirenderv2(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == "pirenderhdv3":
        netG = GlobalGeneratorPirenderv3(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    else:
        raise "generator not implemented!"
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        if not apex:
            netG.togpu_ids[0]
            netG = torch.nn.DataParallelnetGgpu_ids
        netG.applyweights_init
        return netG


def define_D(input_nc, ndf, n_layers_D, norm="instance", use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[], apex=False):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        if not apex:
            netD.togpu_ids[0]
            netD = torch.nn.DataParallelnetDgpu_ids
        netD.applyweights_init
        return netD


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print(net)
    print("Total number of parameters: %d" % num_params)


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = self.real_label_var is None or self.real_label_var.numel() != input.numel()
            if create_label:
                real_tensor = self.Tensorinput.size().fill_self.real_label
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = self.fake_label_var is None or self.fake_label_var.numel() != input.numel()
            if create_label:
                fake_tensor = self.Tensorinput.size().fill_self.fake_label
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)

            return loss
        target_tensor = self.get_target_tensor(input[-1], target_is_real)
        return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):

    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [0.03125,0.0625,0.125,0.25,1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


class LocalEnhancer(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        ngf_global = ngf * 2 ** n_local_enhancers
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global) - 3)]
        self.model = (nn.Sequential)(*model_global)
        for n in range(1, n_local_enhancers + 1):
            ngf_global = ngf * 2 ** (n_local_enhancers - n)
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
             norm_layer(ngf_global), nn.ReLU(True),
             nn.Conv2d(ngf_global, (ngf_global * 2), kernel_size=3, stride=2, padding=1),
             norm_layer(ngf_global * 2), nn.ReLU(True)]
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock((ngf_global * 2), padding_type=padding_type, norm_layer=norm_layer)]

            model_upsample += [
             nn.ConvTranspose2d((ngf_global * 2), ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
             norm_layer(ngf_global), nn.ReLU(True)]
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d3, nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                 nn.Tanh()]
            setattr(self, "model" + str(n) + "_1", (nn.Sequential)(*model_downsample))
            setattr(self, "model" + str(n) + "_2", (nn.Sequential)(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        input_downsampled = [
         input]
        for i in range(self.n_local_enhancers):
            input_downsampled.appendself.downsampleinput_downsampled[-1]

        output_prev = self.modelinput_downsampled[-1]
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, "model" + str(n_local_enhancers) + "_1")
            model_upsample = getattr(self, "model" + str(n_local_enhancers) + "_2")
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)

        return output_prev


class Conv2d(nn.Module):

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(nn.Conv2d(cin, cout, kernel_size, stride, padding), nn.BatchNorm2d(cout))
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AFR(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        assert n_blocks >= 0
        super(AFR, self).__init__()
        activation = nn.ReLU(True)
        ngf_1 = ngf
        model_face = [nn.ReflectionPad2d3, nn.Conv2d(input_nc, ngf_1, kernel_size=7, padding=0), norm_layer(ngf_1),
         activation]
        model_face += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
         norm_layer(128), activation]
        model_face += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
         norm_layer(256), activation]
        model_face += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
         norm_layer(256), activation]
        model_face += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
         norm_layer(256), activation]
        ngf_2 = ngf
        model_reference = [nn.ReflectionPad2d3, nn.Conv2d(input_nc, ngf_2, kernel_size=7, padding=0),
         norm_layer(ngf_2),
         activation]
        model_reference += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
         norm_layer(128), activation]
        model_reference += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
         norm_layer(256), activation]
        model_reference += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
         norm_layer(256), activation]
        model_reference += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
         norm_layer(256), activation]
        model_reference += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
         norm_layer(256), activation]
        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock((ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [
             nn.ConvTranspose2d((int(ngf * mult)), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
             norm_layer(int(ngf * mult / 2)), activation]

        model3 += [
         nn.ConvTranspose2d((int(ngf * mult / 2)), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
         norm_layer(int(ngf * mult / 2)), activation]
        model3 += [nn.ReflectionPad2d3, nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.face_encoder = (nn.Sequential)(*model_face)
        self.reference_encoder = (nn.Sequential)(*model_reference)
        self.audio_encoder = nn.Sequential(Conv2d(1, 64, kernel_size=3, stride=1, padding=(1,
                                                                                           3)), Conv2d(64, 128, kernel_size=3, stride=1, padding=1), Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128, 256, kernel_size=3, stride=2, padding=1), Conv2d(256, 256, kernel_size=3, stride=1, padding=1), Conv2d(256, 512, kernel_size=3, stride=2, padding=1), Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True))
        self.model2 = (nn.Sequential)(*model2)
        self.model3 = (nn.Sequential)(*model3)

    def forward(self, audio_feature, face_feature, referenc_feature):
        a = self.audio_encoderaudio_feature
        f = self.face_encoderface_feature
        r = self.reference_encoderreferenc_feature
        x = torch.cat((a, f, r), dim=1)
        x = self.model2x
        out = self.model3x
        return out


class AFRSmall(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        assert n_blocks >= 0
        super(AFRSmall, self).__init__()
        activation = nn.ReLU(True)
        ngf_1 = ngf // 2
        model_face = [nn.ReflectionPad2d3, nn.Conv2d(input_nc, ngf_1, kernel_size=7, padding=0), norm_layer(ngf_1),
         activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model_face += [nn.Conv2d((ngf_1 * mult), (ngf_1 * mult), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult), activation]
            else:
                model_face += [nn.Conv2d((ngf_1 * mult), (ngf_1 * mult * 2), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf_1 * mult * 2), activation]

        ngf_2 = ngf // 2
        model_reference = [nn.ReflectionPad2d3, nn.Conv2d(input_nc, ngf_2, kernel_size=7, padding=0),
         norm_layer(ngf_2),
         activation]
        model_reference += [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
         norm_layer(64), activation]
        model_reference += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
         norm_layer(128), activation]
        model_reference += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
         norm_layer(256), activation]
        model_reference += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
         norm_layer(256), activation]
        model_reference += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
         norm_layer(256), activation]
        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock((ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [
             nn.ConvTranspose2d((int(ngf * mult)), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
             norm_layer(int(ngf * mult / 2)), activation]

        model3 += [
         nn.ConvTranspose2d((int(ngf * mult / 2)), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
         norm_layer(int(ngf * mult / 2)), activation]
        model3 += [nn.ReflectionPad2d3, nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.audio_encoder = nn.Sequential(Conv2d(1, 64, kernel_size=3, stride=1, padding=(1,
                                                                                           3)), Conv2d(64, 128, kernel_size=3, stride=1, padding=1), Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128, 256, kernel_size=3, stride=2, padding=1), Conv2d(256, 256, kernel_size=3, stride=1, padding=1), Conv2d(256, 512, kernel_size=3, stride=2, padding=1), Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True))
        self.face_encoder = (nn.Sequential)(*model_face)
        self.reference_encoder = (nn.Sequential)(*model_reference)
        self.model2 = (nn.Sequential)(*model2)
        self.model3 = (nn.Sequential)(*model3)

    def forward(self, audio_feature, face_feature, referenc_feature):
        a = self.audio_encoderaudio_feature
        f = self.face_encoderface_feature
        r = self.reference_encoderreferenc_feature
        x = torch.cat((a, f, r), dim=1)
        x = self.model2x
        out = self.model3x
        return out


class Global256to512Generatoraudio(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        assert n_blocks >= 0
        super(Global256to512Generatoraudio, self).__init__()
        activation = nn.ReLU(True)
        model1 = [
         nn.ReflectionPad2d3, nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
         activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult), activation]
            else:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult * 2), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult * 2), activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock((ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [
             nn.ConvTranspose2d((int(ngf * mult)), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
             norm_layer(int(ngf * mult / 2)), activation]

        model3 += [
         nn.ConvTranspose2d((int(ngf * mult / 2)), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
         norm_layer(int(ngf * mult / 2)), activation]
        model3 += [nn.ReflectionPad2d3, nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model1 = (nn.Sequential)(*model1)
        self.model2 = (nn.Sequential)(*model2)
        self.model3 = (nn.Sequential)(*model3)
        self.audio_encoder = nn.Sequential(Conv2d(1, 64, kernel_size=3, stride=1, padding=(1,
                                                                                           3)), Conv2d(64, 128, kernel_size=3, stride=1, padding=1), Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128, 256, kernel_size=3, stride=2, padding=1), Conv2d(256, 256, kernel_size=3, stride=2, padding=1), Conv2d(256, 512, kernel_size=3, stride=1, padding=1), Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True))

    def forward(self, audio_feature, face_feature):
        audio_feature = self.audio_encoderaudio_feature
        x = self.model1face_feature
        x = torch.cat((x, audio_feature), dim=1)
        x = self.model2x
        out = self.model3x
        return out


class GlobalGeneratoraudio2(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        assert n_blocks >= 0
        super(GlobalGeneratoraudio2, self).__init__()
        activation = nn.ReLU(True)
        model1 = [
         nn.ReflectionPad2d3, nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
         activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult), activation]
            else:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult * 2), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult * 2), activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock((ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [
             nn.ConvTranspose2d((int(ngf * mult)), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
             norm_layer(int(ngf * mult / 2)), activation]

        model3 += [nn.ReflectionPad2d3, nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model1 = (nn.Sequential)(*model1)
        self.model2 = (nn.Sequential)(*model2)
        self.model3 = (nn.Sequential)(*model3)
        self.audio_encoder = [
         nn.Conv2d(1, ngf, kernel_size=7, padding=0), norm_layer(ngf),
         activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                self.audio_encoder += [nn.Conv2d((ngf * mult), (ngf * mult), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult), activation]
            else:
                self.audio_encoder += [nn.Conv2d((ngf * mult), (ngf * mult * 2), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult * 2), activation]

        self.audio_encoder = (nn.Sequential)(*self.audio_encoder)

    def forward(self, audio_feature, face_feature):
        x = self.model1face_feature
        audio_feature = self.audio_encoderaudio_feature
        x = torch.cat((x, audio_feature), dim=1)
        x = self.model2x
        out = self.model3x
        return out


class GlobalGeneratorwenet(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        assert n_blocks >= 0
        super(GlobalGeneratorwenet, self).__init__()
        activation = nn.ReLU(True)
        model1 = [
         nn.ReflectionPad2d3, nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
         activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult), activation]
            else:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult * 2), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult * 2), activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock((ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [
             nn.ConvTranspose2d((int(ngf * mult)), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
             norm_layer(int(ngf * mult / 2)), activation]

        model3 += [nn.ReflectionPad2d3, nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model1 = (nn.Sequential)(*model1)
        self.model2 = (nn.Sequential)(*model2)
        self.model3 = (nn.Sequential)(*model3)
        self.audio_encoder = nn.Sequential(Conv2d(1, 64, kernel_size=3, stride=(1,
                                                                                2), padding=(0,
                                                                                             0)), Conv2d(64, 128, kernel_size=3, stride=(1,
                                                                                                                                         2), padding=(0,
                                                                                                                                                      0)), Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128, 256, kernel_size=3, stride=(1,
                                                                                                                                                                                                                                                                                                                                             2), padding=1), Conv2d(256, 256, kernel_size=3, stride=1, padding=1), Conv2d(256, 256, kernel_size=3, stride=1, padding=(1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                      0)), nn.ConvTranspose2d(256, 512, kernel_size=3, stride=(2,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               1), padding=(1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            0), output_padding=(1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                0)), Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True))

    def forward(self, audio_feature, face_feature):
        x = self.model1face_feature
        x = torch.cat((x, audio_feature), dim=1)
        x = self.model2x
        out = self.model3x
        return out


class GlobalGeneratorPirender(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        assert n_blocks >= 0
        super(GlobalGeneratorPirender, self).__init__()
        activation = nn.ReLU(True)
        model1 = [
         nn.ReflectionPad2d3, nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
         activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult), activation]
            else:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult * 2), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult * 2), activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock((ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [
             nn.ConvTranspose2d((int(ngf * mult)), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
             norm_layer(int(ngf * mult / 2)), activation]

        model3 += [nn.ReflectionPad2d3, nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model1 = (nn.Sequential)(*model1)
        self.model2 = (nn.Sequential)(*model2)
        self.model3 = (nn.Sequential)(*model3)
        self.audio_encoder = nn.Sequential(Conv2d(1, 64, kernel_size=3, stride=(1,
                                                                                2), padding=(0,
                                                                                             0)), Conv2d(64, 128, kernel_size=3, stride=(1,
                                                                                                                                         2), padding=(0,
                                                                                                                                                      0)), Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128, 256, kernel_size=3, stride=(1,
                                                                                                                                                                                                                                                                                                                                             2), padding=1), Conv2d(256, 256, kernel_size=3, stride=1, padding=1), Conv2d(256, 256, kernel_size=3, stride=1, padding=(1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                      0)), nn.ConvTranspose2d(256, 512, kernel_size=3, stride=(2,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               1), padding=(1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            0), output_padding=(1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                0)), Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True))
        self.mapping_net = MappingNet(67, 256, 3)
        self.adain = ADAIN(512, 256)

    def forward(self, feature_3dmm, face_feature):
        feature_3dmm = self.mapping_netfeature_3dmm
        x = self.model1face_feature
        adain_x = self.adainxfeature_3dmm
        x = torch.cat((x, adain_x), dim=1)
        x = self.model2x
        out = self.model3x
        return out


class Conv1d(nn.Module):

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(nn.Conv1d(cin, cout, kernel_size, stride, padding), nn.BatchNorm1d(cout))
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class GlobalGeneratorPirenderv2(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        assert n_blocks >= 0
        super(GlobalGeneratorPirenderv2, self).__init__()
        activation = nn.ReLU(True)
        model1 = [
         nn.ReflectionPad2d3, nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
         activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult), activation]
            else:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult * 2), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult * 2), activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock((ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [
             nn.ConvTranspose2d((int(ngf * mult)), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
             norm_layer(int(ngf * mult / 2)), activation]

        model3 += [nn.ReflectionPad2d3, nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model1 = (nn.Sequential)(*model1)
        self.model2 = (nn.Sequential)(*model2)
        self.model3 = (nn.Sequential)(*model3)
        self.exp_3dmm_encoder = nn.Sequential(Conv1d(64, 128, kernel_size=3, stride=2, padding=1), Conv1d(128, 128, kernel_size=3, stride=2, padding=1), Conv1d(128, 128, kernel_size=3, stride=2, padding=1), Conv1d(128, 256, kernel_size=3, stride=2, padding=1), Conv1d(256, 256, kernel_size=3, stride=2, padding=1), Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True))
        self.adain = ADAIN(512, 256)

    def forward(self, feature_3dmm, face_feature):
        feature_3dmm = self.exp_3dmm_encoderfeature_3dmm
        x = self.model1face_feature
        adain_x = self.adainxfeature_3dmm
        x = torch.cat((x, adain_x), dim=1)
        x = self.model2x
        out = self.model3x
        return out


class GlobalGeneratorPirenderv3(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        assert n_blocks >= 0
        super(GlobalGeneratorPirenderv3, self).__init__()
        activation = nn.ReLU(True)
        model1 = [
         nn.ReflectionPad2d3, nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
         activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult), activation]
            else:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult * 2), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult * 2), activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock((ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [
             nn.ConvTranspose2d((int(ngf * mult)), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
             norm_layer(int(ngf * mult / 2)), activation]

        model3 += [nn.ReflectionPad2d3, nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model1 = (nn.Sequential)(*model1)
        self.model2 = (nn.Sequential)(*model2)
        self.model3 = (nn.Sequential)(*model3)
        self.exp_3dmm_encoder = nn.Sequential(Conv1d(323, 128, kernel_size=3, stride=2, padding=1), Conv1d(128, 128, kernel_size=3, stride=2, padding=1), Conv1d(128, 128, kernel_size=3, stride=2, padding=1), Conv1d(128, 256, kernel_size=3, stride=2, padding=1), Conv1d(256, 256, kernel_size=3, stride=2, padding=1), Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True))
        self.adain = ADAIN(512, 256)

    def forward(self, feature_3dmm, face_feature):
        for one in self.exp_3dmm_encoder:
            feature_3dmm = one(feature_3dmm)

        x = self.model1face_feature
        adain_x = self.adainxfeature_3dmm
        x = torch.cat((x, adain_x), dim=1)
        x = self.model2x
        out = self.model3x
        return out


class GlobalGeneratoraudio(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        assert n_blocks >= 0
        super(GlobalGeneratoraudio, self).__init__()
        activation = nn.ReLU(True)
        model1 = [
         nn.ReflectionPad2d3, nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
         activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult), activation]
            else:
                model1 += [nn.Conv2d((ngf * mult), (ngf * mult * 2), kernel_size=3, stride=2, padding=1),
                 norm_layer(ngf * mult * 2), activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock((ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [
             nn.ConvTranspose2d((int(ngf * mult)), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
             norm_layer(int(ngf * mult / 2)), activation]

        model3 += [nn.ReflectionPad2d3, nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model1 = (nn.Sequential)(*model1)
        self.model2 = (nn.Sequential)(*model2)
        self.model3 = (nn.Sequential)(*model3)
        self.audio_encoder = nn.Sequential(Conv2d(1, 64, kernel_size=3, stride=1, padding=(1,
                                                                                           3)), Conv2d(64, 128, kernel_size=3, stride=1, padding=1), Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128, 256, kernel_size=3, stride=2, padding=1), Conv2d(256, 512, kernel_size=3, stride=1, padding=1), Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True))

    def forward(self, audio_feature, face_feature):
        audio_feature = self.audio_encoderaudio_feature
        x = self.model1face_feature
        x = torch.cat((x, audio_feature), dim=1)
        x = self.model2x
        out = self.model3x
        return out


class GlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        assert n_blocks >= 0
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)
        model = [
         nn.ReflectionPad2d3, nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d((ngf * mult), (ngf * mult * 2), kernel_size=3, stride=2, padding=1),
             norm_layer(ngf * mult * 2), activation]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock((ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
             nn.ConvTranspose2d((ngf * mult), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
             norm_layer(int(ngf * mult / 2)), activation]

        model += [nn.ReflectionPad2d3, nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = (nn.Sequential)(*model)

    def forward(self, input):
        return self.modelinput


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d1]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d1]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
         norm_layer(dim),
         activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
         norm_layer(dim)]
        return (nn.Sequential)(*conv_block)

    def forward(self, x):
        out = x + self.conv_blockx
        return out


class Encoder(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc
        model = [
         nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
         norm_layer(ngf), nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d((ngf * mult), (ngf * mult * 2), kernel_size=3, stride=2, padding=1),
             norm_layer(ngf * mult * 2), nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
             nn.ConvTranspose2d((ngf * mult), (int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1),
             norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d3, nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = (nn.Sequential)(*model)

    def forward(self, input, inst):
        outputs = self.modelinput
        outputs_mean = outputs.clone()
        inst_list = np.uniqueinst.cpu().numpy().astypeint
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b + 1] == int(i)).nonzero()
                for j in range(self.output_nc):
                    output_ins = outputs[(indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3])]
                    mean_feat = torch.meanoutput_ins.expand_asoutput_ins
                    outputs_mean[(indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3])] = mean_feat

        return outputs_mean


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, "scale" + str(i) + "_layer" + str(j), getattr(netD, "model" + str(j)))

            else:
                setattr(self, "layer" + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [
             input]
            for i in range(len(model)):
                result.appendmodel[i](result[-1])

            return result[1:]
        return [
         model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, "scale" + str(num_D - 1 - i) + "_layer" + str(j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, "layer" + str(num_D - 1 - i))
            result.appendself.singleD_forwardmodelinput_downsampled
            if i != num_D - 1:
                input_downsampled = self.downsampleinput_downsampled
            return result


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
             [
              nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
              norm_layer(nf), nn.LeakyReLU(0.2, True)]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
         [
          nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
          norm_layer(nf),
          nn.LeakyReLU(0.2, True)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]
        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, "model" + str(n), (nn.Sequential)(*sequence[n]))

        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]

            self.model = (nn.Sequential)(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [
             input]
            for n in range(self.n_layers + 2):
                model = getattr(self, "model" + str(n))
                res.appendmodel(res[-1])

            return res[1:]
        return self.modelinput


from torchvision import models

class Vgg19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


if __name__ == "__main__":
    a = GlobalGeneratorPirender(6, 3)
    b = torch.ones(1, 6, 256, 256)
    c = torch.ones(1, 153, 27)
    d = a(c, b)
    print(d.shape)
