# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/model_lib/base_wrapper/onnx_model.py
# Compiled at: 2024-03-27 17:14:58
# Size of source mod 2**32: 5770 bytes
"""
todo: io_binding https://onnxruntime.ai/docs/api/python/api_summary.html
"""
import onnxruntime, numpy as np
from cv2box import MyFpsCounter
import os

def get_output_info(onnx_session):
    output_name = []
    output_shape = []
    for node in onnx_session.get_outputs():
        output_name.append(node.name)
        output_shape.append(node.shape)

    return (output_name, output_shape)


def get_input_info(onnx_session):
    input_name = []
    input_shape = []
    for node in onnx_session.get_inputs():
        input_name.append(node.name)
        input_shape.append(node.shape)

    return (input_name, input_shape)


def get_input_feed(input_name, image_tensor):
    """
    Args:
        input_name:
        image_tensor: [image tensor, ...]
    Returns:
    """
    input_feed = {}
    for (index, name) in enumerate(input_name):
        input_feed[name] = image_tensor[index]

    return input_feed


class ONNXModel:

    def __init__(self, onnx_path, provider='gpu', debug=False, input_dynamic_shape=None):
        self.provider = provider
        if self.provider == "gpu":
            self.providers = ("CUDAExecutionProvider",
             {"device_id": 0})
        elif self.provider == "trt":
            os.makedirs("./cache/trt", exist_ok=True)
            self.providers = (
             "TensorrtExecutionProvider",
             {'trt_engine_cache_enable':True, 
              'trt_engine_cache_path':"./cache/trt",  'trt_fp16_enable':False})
        elif self.provider == "trt16":
            os.makedirs("./cache/trt", exist_ok=True)
            self.providers = (
             "TensorrtExecutionProvider",
             {
               'trt_engine_cache_enable': True, 'trt_engine_cache_path': "./cache/trt", 'trt_fp16_enable': True,
               'trt_dla_enable': False})
        elif self.provider == "trt8":
            os.makedirs("./cache/trt", exist_ok=True)
            self.providers = (
             "TensorrtExecutionProvider",
             {'trt_engine_cache_enable':True, 
              'trt_int8_enable':True})
        else:
            self.providers = "CPUExecutionProvider"
        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, session_options, providers=[self.providers])
        (self.input_name, self.input_shape) = get_input_info(self.onnx_session)
        (self.output_name, self.output_shape) = get_output_info(self.onnx_session)
        self.input_dynamic_shape = input_dynamic_shape
        if self.input_dynamic_shape is not None:
            self.input_dynamic_shape = self.input_dynamic_shape if isinstance(self.input_dynamic_shape, list) else [
             self.input_dynamic_shape]
        if debug:
            print("onnx version: {}".format(onnxruntime.__version__))
            print("input_name:{}, shape:{}".format(self.input_name, self.input_shape))
            print("output_name:{}, shape:{}".format(self.output_name, self.output_shape))
        self.warm_up()

    def warm_up(self):
        if not self.input_dynamic_shape:
            try:
                self.forward([(np.random.rand)(*self.input_shape[i]).astype(np.float32) for i in range(len(self.input_shape))])
            except TypeError:
                print("Model may be dynamic, plz name the 'input_dynamic_shape' !")

        else:
            self.forward([(np.random.rand)(*self.input_dynamic_shape[i]).astype(np.float32) for i in range(len(self.input_shape))])
        print("Model warm up done !")

    def speed_test(self):
        if not self.input_dynamic_shape:
            input_tensor = [(np.random.rand)(*self.input_shape[i]).astype(np.float32) for i in range(len(self.input_shape))]
        else:
            input_tensor = [(np.random.rand)(*self.input_dynamic_shape[i]).astype(np.float32) for i in range(len(self.input_shape))]
        with MyFpsCounter("[{}] onnx 10 times".format(self.provider)) as mfc:
            for i in range(10):
                _ = self.forward(input_tensor)

    def forward(self, image_tensor_in, trans=False):
        """
        Args:
            image_tensor_in: image_tensor [image_tensor] [image_tensor_1, image_tensor_2]
            trans: apply trans for image_tensor or first image_tensor(list)
        Returns:
            model output
        """
        if not isinstance(image_tensor_in, list) or len(image_tensor_in) == 1:
            image_tensor_in = image_tensor_in[0] if isinstance(image_tensor_in, list) else image_tensor_in
            if trans:
                image_tensor_in = image_tensor_in.transpose(2, 0, 1)[np.newaxis, :]
            image_tensor_in = [
             np.ascontiguousarray(image_tensor_in)]
        else:
            if trans:
                image_tensor_in[0] = image_tensor_in[0].transpose(2, 0, 1)[np.newaxis, :]
            image_tensor_in = [np.ascontiguousarray(image_tensor) for image_tensor in image_tensor_in]
        input_feed = get_input_feed(self.input_name, image_tensor_in)
        return self.onnx_session.run((self.output_name), input_feed=input_feed)

    def batch_forward(self, bach_image_tensor, trans=False):
        if trans:
            bach_image_tensor = bach_image_tensor.transpose(0, 3, 1, 2)
        input_feed = get_input_feed(self.input_name, bach_image_tensor)
        return self.onnx_session.run((self.output_name), input_feed=input_feed)

