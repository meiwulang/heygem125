# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/landmark2face_wy/models/l2faceaudio_model.py
# Compiled at: 2024-03-06 17:51:22
# Size of source mod 2**32: 4007 bytes
import torch
from .base_model import BaseModel
from . import networks

class L2FaceAudioModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.visual_names = ["real_A", "fake_B", "real_B", "mask_B"]
        if self.isTrain:
            self.loss_names = [
             "G_GAN", "G_L1", "D_real", "D_fake"]
            self.model_names = ["G", "D"]
        else:
            self.model_names = [
             "G"]
        self.netG = networks.define_G((opt.input_nc), (opt.output_nc), (opt.ngf), (opt.netG), (opt.norm), (not opt.no_dropout),
          (opt.init_type), (opt.init_gain), (self.gpu_ids), input_size=(opt.dataloader_size))
        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam((self.netG.parameters()), lr=(opt.lr), betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam((self.netD.parameters()), lr=(opt.lr), betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input["A"].to(self.device)
        self.A_label = input["A_label"].to(self.device)
        self.real_B = input["B"].to(self.device)
        self.B_label = input["B_label"].to(self.device)
        self.mask_B = input["mask_B"].to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.B_label, torch.cat((self.mask_B, self.real_A), 1))

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        lambda_GAN = 1
        lambda_L1 = 100
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * lambda_GAN
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def eval_(self):
        lambda_GAN = 1
        lambda_L1 = 100
        self.forward()
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * lambda_GAN
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * lambda_L1

