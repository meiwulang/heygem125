# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/landmark2face_wy/options/base_options.py
# Compiled at: 2024-03-19 19:15:24
# Size of source mod 2**32: 13086 bytes
import argparse, os
from landmark2face_wy.util import util as util
import torch
import landmark2face_wy.models as models
import landmark2face_wy.data as data

class BaseOptions:
    __doc__ = "This class defines options used during both training and test time.\n\n    It also implements several helper functions such as parsing, printing, and saving the options.\n    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.\n    "

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser.add_argument("--dataroot", type=str, default="./data")
        parser.add_argument("--name", type=str, default="test", help="name of the experiment. It decides where to store samples and models")
        parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
        parser.add_argument("--checkpoints_dir", type=str, default="./landmark2face_wy/checkpoints", help="models are saved here")
        parser.add_argument("--model", type=str, default="pirender_3dmm_mouth_hd", help="chooses which model to use. [cycle_gan | pix2pix | test | colorization]")
        parser.add_argument("--input_nc", type=int, default=3, help="# of input image channels: 3 for RGB and 1 for grayscale")
        parser.add_argument("--output_nc", type=int, default=3, help="# of output image channels: 3 for RGB and 1 for grayscale")
        parser.add_argument("--ngf", type=int, default=64, help="# of gen filters in the last conv layer")
        parser.add_argument("--ndf", type=int, default=64, help="# of discrim filters in the first conv layer")
        parser.add_argument("--netD", type=str, default="basic", help="specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator")
        parser.add_argument("--netG", type=str, default="pirender", help="specify generator architecture [resnet_9blocks | resnet_8blocks | resnet_6blocks | unet_256 | unet_128]")
        parser.add_argument("--n_layers_D", type=int, default=3, help="only used if netD==n_layers")
        parser.add_argument("--n_blocks", type=int, default=9, help="only used if netG == resnet_9blocks_l2face/512")
        parser.add_argument("--norm", type=str, default="instance", help="instance normalization or batch normalization [instance | batch | none]")
        parser.add_argument("--init_type", type=str, default="normal", help="network initialization [normal | xavier | kaiming | orthogonal]")
        parser.add_argument("--init_gain", type=float, default=0.02, help="scaling factor for normal, xavier and orthogonal.")
        parser.add_argument("--no_dropout", action="store_true", default=True, help="no dropout for the generator")
        parser.add_argument("--num_D", type=int, default=2, help="number of discriminators to use")
        parser.add_argument("--no_ganFeat_loss", action="store_true", help="if specified, do *not* use discriminator feature matching loss")
        parser.add_argument("--n_downsample_global", type=int, default=4, help="number of downsampling layers in netG")
        parser.add_argument("--n_blocks_global", type=int, default=9, help="number of residual blocks in the global generator network")
        parser.add_argument("--n_blocks_local", type=int, default=3, help="number of residual blocks in the local enhancer network")
        parser.add_argument("--n_local_enhancers", type=int, default=1, help="number of local enhancers to use")
        parser.add_argument("--niter_fix_global", type=int, default=0, help="number of epochs that we only train the outmost local enhancer")
        parser.add_argument("--no_instance", action="store_true", help="if specified, do *not* add instance map as input")
        parser.add_argument("--instance_feat", action="store_true", help="if specified, add encoded instance features as input")
        parser.add_argument("--label_feat", action="store_true", help="if specified, add encoded label features as input")
        parser.add_argument("--feat_num", type=int, default=3, help="vector length for encoded features")
        parser.add_argument("--load_features", action="store_true", help="if specified, load precomputed feature maps")
        parser.add_argument("--n_downsample_E", type=int, default=2, help="# of downsampling layers in encoder")
        parser.add_argument("--nef", type=int, default=16, help="# of encoder filters in the first conv layer")
        parser.add_argument("--n_clusters", type=int, default=10, help="number of clusters for features")
        parser.add_argument("--dataset_mode", type=str, default="Facereala3dmm", help="chooses how datasets are loaded.")
        parser.add_argument("--img_size", type=int, default=256, help="input image size")
        parser.add_argument("--lan_size", type=int, default=1, help="input image size")
        parser.add_argument("--direction", type=str, default="AtoB", help="AtoB or BtoA")
        parser.add_argument("--serial_batches", action="store_true", help="if true, takes images in order to make batches, otherwise takes them randomly")
        parser.add_argument("--num_threads", default=8, type=int, help="# threads for loading data")
        parser.add_argument("--batch_size", type=int, default=16, help="input batch size")
        parser.add_argument("--load_size", type=int, default=286, help="scale images to this size")
        parser.add_argument("--crop_size", type=int, default=256, help="then crop to this size")
        parser.add_argument("--max_dataset_size", type=int, default=(float("inf")), help="Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.")
        parser.add_argument("--preprocess", type=str, default="none", help="scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]")
        parser.add_argument("--no_flip", action="store_true", help="if specified, do not flip the images for data augmentation")
        parser.add_argument("--display_winsize", type=int, default=256, help="display window size for both visdom and HTML")
        parser.add_argument("--epoch", type=str, default="latest", help="which epoch to load? set to latest to use latest cached model")
        parser.add_argument("--load_iter", type=int, default="0", help="which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]")
        parser.add_argument("--verbose", action="store_true", help="if specified, print more debugging information")
        parser.add_argument("--suffix", default="", type=str, help="customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}")
        parser.add_argument("--audio_feature", help="audio feature extraction mode: mfcc, deepSpeech, wenet, 3dmm", default="3dmm",
          type=str)
        parser.add_argument("--feature_path", help="dataset path", default="../AnnI_deep3dface_256_contains_id/", type=str)
        parser.add_argument("--fp16", action="store_true", default=True, help="train with AMP")
        parser.add_argument("--mfcc0_rate", type=float, default=0.2, help="mfcc drop rate for audio2face512audioMFCC0 dataset mode")
        parser.add_argument("--distributed", action="store_true", default=False)
        parser.add_argument("--local_rank", default=(-1), type=int, help="node rank for distributed training")
        parser.add_argument("--resize_size", type=int, default=512, help="resize image size")
        parser.add_argument("--perceptual_network", type=str, default="vgg19", help="perceptual loss network")
        parser.add_argument("--perceptual_layers", type=list, default=[
         "relu_1_1","relu_2_1","relu_3_1","relu_4_1","relu_5_1"],
          help="perceptual loss layers")
        parser.add_argument("--perceptual_num_scales", type=int, default=4, help="perceptual loss num_scales")
        parser.add_argument("--perceptual_use_style_loss", type=bool, default=True, help="perceptual loss use_style_loss")
        parser.add_argument("--weight_style_to_perceptual", type=float, default=250, help="perceptual loss style weight")
        parser.add_argument("--perceptual_weights", type=list, default=[4,4,4,4,4], help="perceptual loss weights")
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=(argparse.ArgumentDefaultsHelpFormatter))
            parser = self.initialize(parser)
        (opt, _) = parser.parse_known_args()
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        (opt, _) = parser.parse_known_args()
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ""
        message += "----------------- Options ---------------\n"
        for (k, v) in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)

        message += "----------------- End -------------------"
        print(message)
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        if opt.distributed:
            print("===================init distributed ==================")
            torch.distributed.init_process_group(backend="nccl")
            opt.gpu_ids = str(torch.distributed.get_rank())
        if opt.suffix:
            suffix = "_" + (opt.suffix.format)(**vars(opt)) if opt.suffix != "" else ""
            opt.name = opt.name + suffix
        self.print_options(opt)
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        self.opt = opt
        return self.opt

