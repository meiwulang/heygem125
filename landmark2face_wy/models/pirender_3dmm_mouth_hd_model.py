# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/landmark2face_wy/models/pirender_3dmm_mouth_hd_model.py
# Compiled at: 2024-03-06 17:51:22
# Size of source mod 2**32: 8018 bytes
import torch
from .base_model import BaseModel
from . import networks_pix2pixHD as networks

class Pirender3dmmmouthhdModel(BaseModel):

    def name(self):
        return "Pirender3dmmmouthhdModel"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.resize_size = opt.resize_size
        (self.m_x1, self.m_x2, self.m_y1, self.m_y2) = (240 * self.resize_size // 512, 400 * self.resize_size // 512, 120 * self.resize_size // 512, -120 * self.resize_size // 512)
        self.visual_names = [
         "real_A", "fake_B", "real_B", "mask_B"]
        if self.isTrain:
            self.loss_names = [
             "G_GAN","G_L1","G_VGG","D_real","D_fake","G_mouthL1","G_mouthGAN","D_real_m",
             "D_fake_m"]
            self.model_names = ["G", "D"]
        else:
            self.model_names = [
             "G"]
        self.netG = networks.define_G(6, (opt.output_nc), (opt.ngf), (opt.netG), (opt.n_downsample_global),
          (opt.n_blocks_global), (opt.n_local_enhancers), (opt.n_blocks_local),
          (opt.norm), gpu_ids=(self.gpu_ids), apex=(opt.fp16))
        if self.isTrain:
            use_sigmoid = False
            self.netD = networks.define_D((opt.input_nc + opt.output_nc), (opt.ndf), (opt.n_layers_D), (opt.norm), use_sigmoid, (opt.num_D),
              (not opt.no_ganFeat_loss), gpu_ids=(self.gpu_ids), apex=(opt.fp16))
        if self.isTrain:
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionGAN = networks.GANLoss(use_lsgan=True, tensor=(torch.cuda.FloatTensor))
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam((self.netG.parameters()), lr=(opt.lr), betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam((self.netD.parameters()), lr=(opt.lr), betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if self.opt.fp16:
                import apex
                ((self.netG, self.netD), (self.optimizer_G, self.optimizer_D)) = apex.amp.initialize([
                 self.netG.to(self.device), self.netD.to(self.device)],
                  [self.optimizer_G, self.optimizer_D], opt_level="O1")
                if not opt.distributed:
                    self.netG = torch.nn.DataParallel((self.netG), device_ids=(opt.gpu_ids))
                    self.netD = torch.nn.DataParallel((self.netD), device_ids=(opt.gpu_ids))
                else:
                    self.netG = apex.parallel.DistributedDataParallel((self.netG), delay_allreduce=True)
                    self.netD = apex.parallel.DistributedDataParallel((self.netD), delay_allreduce=True)

    def set_input(self, input):
        self.real_A = input["A"].to(self.device).float()
        self.A_label = input["A_label"].to(self.device).float()
        self.real_B = input["B"].to(self.device).float()
        self.B_label = input["B_label"].to(self.device).float()
        self.mask_B = input["mask_B"].to(self.device).float()

    def forward(self):
        self.fake_B = self.netG(self.B_label, torch.cat((self.mask_B, self.real_A), 1))

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        if self.opt.fp16:
            import apex
            with apex.amp.scale_loss(self.loss_D, self.optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_D.backward()
        m_fake_AB = torch.cat((self.real_A[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2],
         self.fake_B[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2]), 1)
        m_pred_fake = self.netD(m_fake_AB.detach())
        self.loss_D_fake_m = self.criterionGAN(m_pred_fake, False)
        m_real_AB = torch.cat((self.real_A[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2],
         self.real_B[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2]), 1)
        m_pred_real = self.netD(m_real_AB)
        self.loss_D_real_m = self.criterionGAN(m_pred_real, True)
        self.loss_mouthD = (self.loss_D_fake_m + self.loss_D_real_m) * 0.5
        if self.opt.fp16:
            import apex
            with apex.amp.scale_loss(self.loss_mouthD, self.optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_mouthD.backward()

    def backward_G(self):
        lambda_GAN = 1
        lambda_L1 = 100
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * lambda_GAN
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * lambda_L1
        m_fake_AB = torch.cat((self.real_A[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2],
         self.fake_B[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2]), 1)
        m_pred_fake = self.netD(m_fake_AB)
        self.loss_G_mouthGAN = self.criterionGAN(m_pred_fake, True) * lambda_GAN * 2
        self.loss_G_mouthL1 = self.criterionL1(self.fake_B[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2], self.real_B[:, :, self.m_x1:self.m_x2,
         self.m_y1:self.m_y2]) * lambda_L1 * 2
        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * lambda_L1 / 6
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_VGG + self.loss_G_mouthL1 + self.loss_G_mouthGAN
        if self.opt.fp16:
            import apex
            with apex.amp.scale_loss(self.loss_G, self.optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
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
        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * lambda_L1 / 6
        self.loss_G_mouthL1 = self.criterionL1(self.fake_B[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2], self.real_B[:, :, self.m_x1:self.m_x2,
         self.m_y1:self.m_y2]) * lambda_L1 * 2

