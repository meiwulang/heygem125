# File Name: /code/face_detect_utils/head_pose.py
# Python 3.8

import numpy as np
import cv2
import onnxruntime as ort
import os


class Headpose:
    """
    使用ONNX模型进行头部姿态估计的类。
    """

    def __init__(self, cpu=False, onnx_path=''):
        """
        初始化头部姿态估计器。
        :param cpu: 是否强制使用CPU。
        :param onnx_path: WHENet ONNX模型的路径。
        """
        # 预先计算用于角度计算的索引张量
        self.idx_tensor_yaw = [np.array(idx, dtype=np.float32) for idx in range(120)]
        self.idx_tensor = [np.array(idx, dtype=np.float32) for idx in range(66)]

        # 模型输入尺寸
        self.whenet_H = 224
        self.whenet_W = 224

        # 图像归一化的均值和标准差
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # 加载ONNX session
        try:
            # 根据cpu标志选择执行提供者
            providers = ['CPUExecutionProvider'] if cpu else ['CUDAExecutionProvider']
            self.whenet_session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            raise Exception(f"load head pose onnx failed: {e}")

        # 字节码显示了再次设置providers的冗余操作，忠实还原
        if cpu:
            self.whenet_session.set_providers(['CPUExecutionProvider'])
        else:
            self.whenet_session.set_providers(['CUDAExecutionProvider'])

        # 获取模型输入和输出信息
        self.whenet_input_name = self.whenet_session.get_inputs()[0].name
        self.whenet_output_names = [output.name for output in self.whenet_session.get_outputs()]
        self.whenet_output_shapes = [output.shape for output in self.whenet_session.get_outputs()]

        # 验证模型输出形状是否符合预期
        assert self.whenet_output_shapes[0] == [1, 120]
        assert self.whenet_output_shapes[1] == [1, 66]
        assert self.whenet_output_shapes[2] == [1, 66]

    def softmax(self, x):
        """
        计算softmax。
        """
        x -= np.max(x, axis=1, keepdims=True)
        a = np.exp(x)
        b = np.sum(np.exp(x), axis=1, keepdims=True)
        return a / b

    def get_head_pose(self, image):
        """
        从输入的图像中计算头部姿态。
        :param image: BGR格式的图像 (numpy array)。
        :return: 一个包含(pitch, roll, yaw)的元组。
        """
        # 1. 图像预处理
        croped_resized_frame = cv2.resize(image, (self.whenet_W, self.whenet_H))

        # BGR to RGB
        rgb = croped_resized_frame[..., ::-1]

        # 归一化
        rgb = (rgb / 255.0 - self.mean) / self.std

        # HWC to CHW
        chw = rgb.transpose((2, 0, 1))

        # Add batch dimension (CHW to NCHW)
        nchw = np.asarray(chw[np.newaxis, ...], dtype=np.float32)

        # 2. ONNX推理
        yaw, roll, pitch = self.whenet_session.run(
            output_names=self.whenet_output_names,
            input_feed={self.whenet_input_name: nchw}
        )

        # 3. 后处理，计算角度
        # Yaw
        yaw = self.softmax(yaw)
        yaw = np.sum(yaw * self.idx_tensor_yaw, axis=1) * 3 - 180

        # Pitch
        pitch = self.softmax(pitch)
        pitch = np.sum(pitch * self.idx_tensor, axis=1) * 3 - 99

        # Roll
        roll = self.softmax(roll)
        roll = np.sum(roll * self.idx_tensor, axis=1) * 3 - 99

        # 移除多余的维度
        yaw, pitch, roll = np.squeeze([yaw, pitch, roll])

        # 按照 (pitch, roll, yaw) 的顺序返回
        return (pitch, roll, yaw)


if __name__ == '__main__':
    # 实例化头部姿态估计器
    hp = Headpose(
        cpu=False,
        onnx_path='/home/zml/my_code/HeadPoseEstimation-WHENet-yolov4-onnx-openvino/saved_model_224x224/model_float32.onnx'
    )

    # 设置包含测试图片的目录
    base_dir = '/home/zml/dataset/项目/测试数据/测试模板/测试缩放/zhengyuqi/532_dlib_crop'

    # 遍历目录中的所有图片并进行姿态估计
    for img_name in sorted(os.listdir(base_dir)):
        img_path = os.path.join(base_dir, img_name)
        img = cv2.imread(img_path)

        # 获取并打印头部姿态角度
        pose = hp.get_head_pose(img)
        print(f"Image: {img_name}, Pose (Pitch, Roll, Yaw): {pose}")