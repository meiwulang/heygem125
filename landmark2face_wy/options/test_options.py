# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/landmark2face_wy/options/test_options.py
# Compiled at: 2024-04-01 10:28:20
# Size of source mod 2**32: 1407 bytes
from .base_options import BaseOptions
from y_utils.config import GlobalConfig

class TestOptions(BaseOptions):
    __doc__ = "This class includes test options.\n\n    It also includes shared options defined in BaseOptions.\n    "

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--ntest", type=int, default=(float("inf")), help="# of test examples.")
        parser.add_argument("--results_dir", type=str, default="./results/", help="saves results here.")
        parser.add_argument("--model_path", type=str, default="./landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth",
          help="saves results here.")
        parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of result images")
        parser.add_argument("--phase", type=str, default="test", help="train, val, test, etc")
        parser.add_argument("--eval", action="store_true", help="use eval mode during test time.")
        parser.add_argument("--num_test", type=int, default=50, help="how many test images to run")
        parser.add_argument("--test_muban", type=str)
        parser.add_argument("--test_audio_path", type=str)
        self.isTrain = False
        return parser

