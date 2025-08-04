# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/face_lib/face_restore/gfpgan_onnx/gfpgan_onnx_api.py
# Compiled at: 2024-04-01 10:28:12
# Size of source mod 2**32: 1459 bytes
from cv2box import CVImage, MyFpsCounter
from model_lib import ModelBase
MODEL_ZOO = {"GFPGANv1.4": {"model_path": "pretrain_models/face_lib/face_restore/gfpgan/GFPGANv1.4.onnx"}}

class GFPGAN(ModelBase):

    def __init__(self, model_type='GFPGANv1.4', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type
        self.input_std = self.input_mean = 127.5
        self.input_size = (512, 512)

    def forward(self, face_image):
        """
        Args:
            face_image: cv2 image 0-255 BGR
        Returns:
            BGR 512x512x3 0-1
        """
        image_in = CVImage(face_image).blob((self.input_size), (self.input_mean), (self.input_std), rgb=True)
        image_out = self.model.forward(image_in)
        output_face = ((image_out[0][0] + 1) / 2)[::-1].transpose(1, 2, 0).clip(0, 1)
        return output_face


if __name__ == "__main__":
    face_img_p = "resource/cropped_face/512.jpg"
    fa = GFPGAN(model_type="GFPGANv1.4", provider="gpu")
    with MyFpsCounter() as mfc:
        for i in range(10):
            face = fa.forward(face_img_p)

    CVImage(face, image_format="cv2").show()

