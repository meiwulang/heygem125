# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/face_attr_detect/face_attr.py
# Compiled at: 2024-03-14 11:24:30
# Size of source mod 2**32: 862 bytes
import numpy as np
from cv2box import CVImage
from apstone import ModelBase
MODEL_ZOO = {"face_attr_mbnv3": {'model_path':"./face_attr_detect/face_attr_epoch_12_220318.onnx", 
                     'input_dynamic_shape':(1, 3, 512, 512)}}

class FaceAttr(ModelBase):

    def __init__(self, model_name='face_attr_mbnv3', provider='gpu'):
        super().__init__(MODEL_ZOO[model_name], provider)

    def forward(self, image_p_):
        blob = CVImage(image_p_).blob_innormal(512, input_mean=[132.38155592, 110.99284567, 102.62942472], input_std=[
         68.5106407, 61.65929394, 58.61700102])
        result = self.model.forward(blob, trans=False)[0]
        return np.around(result, 3)

    @staticmethod
    def show_label():
        print("female male front side clean occlusion super_hq hq blur nonhuman")

