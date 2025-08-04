import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .face_model import *
from .networks_HD import *
from .DINet import DINetMouth, DINetV1, DINetMouth512

def get_norm_layer(norm_type='instance'):
    if norm_type == "batch":
        norm_layer = functools.partial((nn.BatchNorm2d), affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial((nn.InstanceNorm2d), affine=False, track_running_stats=False)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=(opt.lr_decay_iters), gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(opt.niter), eta_min=0)
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight"):
            if classname.find("Conv") != -1 or classname.find("Linear") != -1:
                if init_type == "normal":
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == "xavier":
                    init.xavier_normal_((m.weight.data), gain=init_gain)
                elif init_type == "kaiming":
                    init.kaiming_normal_((m.weight.data), a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    init.orthogonal_((m.weight.data), gain=init_gain)
                else:
                    raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
                if hasattr(m, "bias"):
                    if m.bias is not None:
                        init.constant_(m.bias.data, 0.0)
            if classname.find("BatchNorm2d") != -1:
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[], apex=False):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        if not apex:
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)
        init_weights(net, init_type, init_gain=init_gain)
        return net


def define_G(input_nc, output_nc, ngf, netG, norm="batch", use_dropout=False, init_type="normal", init_gain=0.02, gpu_ids=[], apex=False, n_blocks=9, input_size=512):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == "resnet_9blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == "resnet_9blocks_l2face":
        net = ResnetL2FaceGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif netG == "resnet_9blocks_l2face_512":
        net = ResnetL2Face512Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif netG == "resnet_9blocks_l2face_256audio":
        net = ResnetL2Face256AudioGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif netG == "resnet_9blocks_l2face_512audio":
        net = ResnetL2Face512AudioGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif netG == "resnet_9_gate_512audio":
        net = ResnetGateAudioGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif netG == "resnet_6blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=1)
    elif netG == "unet_128":
        net = UnetGenerator(input_nc, output_nc, 1, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == "unet_256":
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == "resnet_9_gate":
        net = ResnetGateGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == "resnet_9_gate_512SAE":
        net = ResnetGateSAEGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif netG == "wenet" or netG == "wenet_v2":
        net = define_G_gxx(6, 3, 64, "wenet", n_downsample_global=4, n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3, norm="instance", gpu_ids=[], input_size=input_size)
    elif netG == "dinet_ref_9channel_encode":
        net = DINetMouth(source_channel=3, ref_channel=3, mouth_ref_channel=6, audio_channel=256)
    elif netG == "dinet_v1":
        net = DINetV1(source_channel=3, ref_channel=3, audio_channel=256)
    elif netG == "dinet_512":
        net = DINetMouth512(source_channel=3, ref_channel=3, audio_channel=256)
    else:
        raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    return net


def define_D(input_nc, ndf, netD, n_layers_D=3, norm="batch", init_type="normal", init_gain=0.02, gpu_ids=[], apex=False):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netD == "basic":
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == "n_layers":
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == "pixel":
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError("Discriminator model name [%s] is not recognized" % net)
    return init_net(net, init_type, init_gain, gpu_ids, apex=apex)


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ('wgangp', ):
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ('lsgan', 'vanilla'):
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    if lambda_gp > 0.0:
        if type == "real":
            interpolatesv = real_data
        elif type == "fake":
            interpolatesv = fake_data
        elif type == "mixed":
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = (alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view)(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + (1 - alpha) * fake_data
        else:
            raise NotImplementedError("{} not implemented".format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv, grad_outputs=(torch.ones(disc_interpolates.size()).to(device)),
          create_graph=True,
          retain_graph=True,
          only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
        return (
         gradient_penalty, gradients)
    return (0.0, None)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type="reflect"):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3),
         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
         norm_layer(ngf),
         nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d((ngf * mult), (ngf * mult * 2), kernel_size=3, stride=2, padding=1, bias=use_bias),
             norm_layer(ngf * mult * 2),
             nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
             ResnetBlock((ngf * mult), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
             nn.ConvTranspose2d((ngf * mult), (int(ngf * mult / 2)), kernel_size=3,
               stride=2,
               padding=1,
               output_padding=1,
               bias=use_bias),
             norm_layer(int(ngf * mult / 2)),
             nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = (nn.Sequential)(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return (nn.Sequential)(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# 优化后的 UnetGenerator - 主要优化点
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        
        # 预计算所有层的通道数，避免重复计算
        channel_configs = []
        for i in range(num_downs):
            if i < 3:  # 前3层
                inner_channels = ngf * (2 ** min(i + 3, 3))  # 限制最大为 ngf * 8
            else:  # 后续层都是 ngf * 8
                inner_channels = ngf * 8
            channel_configs.append(inner_channels)
        
        # 从最内层开始构建
        unet_block = UnetSkipConnectionBlock(
            channel_configs[-1], channel_configs[-1], 
            input_nc=None, submodule=None, 
            norm_layer=norm_layer, innermost=True
        )
        
        # 构建中间层 - 优化循环
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                channel_configs[-(i+2)], channel_configs[-(i+1)], 
                input_nc=None, submodule=unet_block, 
                norm_layer=norm_layer, use_dropout=use_dropout
            )
        
        # 构建外层 - 使用预计算的通道配置
        layer_configs = [
            (ngf * 4, ngf * 8),
            (ngf * 2, ngf * 4), 
            (ngf, ngf * 2)
        ]
        
        for outer_nc, inner_nc in layer_configs:
            unet_block = UnetSkipConnectionBlock(
                outer_nc, inner_nc, input_nc=None, 
                submodule=unet_block, norm_layer=norm_layer
            )
        
        # 最外层
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, 
            submodule=unet_block, outermost=True, 
            norm_layer=norm_layer
        )

    def forward(self, input):
        return self.model(input)


# 优化后的 UnetSkipConnectionBlock - 减少计算开销
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        
        # 预计算bias设置，避免重复判断
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        if input_nc is None:
            input_nc = outer_nc
        
        # 使用更高效的激活函数和归一化层配置
        # 优化卷积层配置 - 减少参数传递开销
        conv_kwargs = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': use_bias}
        
        downconv = nn.Conv2d(input_nc, inner_nc, **conv_kwargs)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            # 使用 ModuleList 替代列表拼接，提高效率
            self.down_layers = nn.ModuleList([downconv])
            self.up_layers = nn.ModuleList([uprelu, upconv, nn.Tanh()])
            self.submodule = submodule
            self.model_type = 'outermost'
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, **conv_kwargs)
            self.down_layers = nn.ModuleList([downrelu, downconv])
            self.up_layers = nn.ModuleList([uprelu, upconv, upnorm])  
            self.submodule = None
            self.model_type = 'innermost'
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, **conv_kwargs)
            self.down_layers = nn.ModuleList([downrelu, downconv, downnorm])
            self.up_layers = nn.ModuleList([uprelu, upconv, upnorm])
            if use_dropout:
                self.up_layers.append(nn.Dropout(0.5))
            self.submodule = submodule
            self.model_type = 'middle'

    def forward(self, x):
        # 优化前向传播 - 减少中间变量创建
        if self.model_type == 'outermost':
            # 下采样
            for layer in self.down_layers:
                x = layer(x)
            # 子模块
            x = self.submodule(x)
            # 上采样
            for layer in self.up_layers:
                x = layer(x)
            return x
        elif self.model_type == 'innermost':
            # 下采样
            for layer in self.down_layers:
                x = layer(x)
            # 上采样
            for layer in self.up_layers:
                x = layer(x)
            return torch.cat([x, x], 1)  # 直接返回cat结果
        else:  # middle
            identity = x
            # 下采样
            for layer in self.down_layers:
                x = layer(x)
            # 子模块
            x = self.submodule(x)
            # 上采样
            for layer in self.up_layers:
                x = layer(x)
            return torch.cat([identity, x], 1)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
             nn.Conv2d((ndf * nf_mult_prev), (ndf * nf_mult), kernel_size=kw, stride=2, padding=padw, bias=use_bias),
             norm_layer(ndf * nf_mult),
             nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
         nn.Conv2d((ndf * nf_mult_prev), (ndf * nf_mult), kernel_size=kw, stride=1, padding=padw, bias=use_bias),
         norm_layer(ndf * nf_mult),
         nn.LeakyReLU(0.2, True)]
        sequence += [
         nn.Conv2d((ndf * nf_mult), 1, kernel_size=kw, stride=1, padding=1)]
        self.model = (nn.Sequential)(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d
        self.net = [
         nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
         nn.LeakyReLU(0.2, True),
         nn.Conv2d(ndf, (ndf * 2), kernel_size=1, stride=1, padding=0, bias=use_bias),
         norm_layer(ndf * 2),
         nn.LeakyReLU(0.2, True),
         nn.Conv2d((ndf * 2), 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        self.net = (nn.Sequential)(*self.net)

    def forward(self, input):
        return self.net(input)