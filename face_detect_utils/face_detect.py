# File Name: /data/zhangmaolin/code/face2face_train/face_detect_utils/face_detect.py
# Python 3.9

import numpy as np
import cv2
from .scrfd import SCRFD
import onnxruntime as ort
import os


class FaceDetect:
    """
    人脸检测类，使用SCRFD模型。
    """

    def __init__(self, mode='scrfd_500m', cpu=False, model_path='./resources/'):
        """
        初始化人脸检测器。
        :param mode: 使用的模型模式，'scrfd_500m' 或 'scrfd_10g'。
        :param cpu: 是否强制使用CPU。
        :param model_path: 存放ONNX模型的目录路径。
        """
        if 'scrfd' in mode:
            scrfd_model_path = ''
            if mode == 'scrfd_500m':
                scrfd_model_path = os.path.join(model_path, 'scrfd_500m_bnkps_shape640x640.onnx')
            elif mode == 'scrfd_10g':
                scrfd_model_path = os.path.join(model_path, 'scrfd_10g_bnkps.onnx')

            # 注意：这里的 `cpu` 参数传递方式是根据字节码 CALL_FUNCTION_KW 推断的
            # 实际的 scrfd 库可能使用不同的方式来指定设备
            self.det_model = SCRFD(model_file=scrfd_model_path)
            # 根据字节码，ctx_id=0 表示使用第一个可用的设备（GPU 0或CPU）
            self.det_model.prepare(ctx_id=0, input_size=(640, 640))

    def get_bboxes(self, image, thresh=0.5, max_num=0):
        """
        从图像中获取人脸边界框和关键点。
        :param image: 输入图像，可以是文件路径(str)或numpy数组(BGR格式)。
        :param thresh: 人脸检测的置信度阈值。
        :param max_num: 返回的最大人脸数量，0表示不限制。
        :return: (bboxes, kpss) 元组，bboxes是边界框，kpss是关键点。
        """
        if isinstance(image, str):
            # 如果输入是字符串，则假定为文件路径并读取
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            # 如果是numpy数组，则直接使用（假设已经是RGB格式或模型内部处理）
            # 字节码中有一个多余的检查，这里简化了逻辑
            pass

        # 使用 'max' 度量标准来选择人脸
        bboxes_, kpss_ = self.det_model.detect(image, thresh=thresh, max_num=max_num, metric='max')
        return bboxes_, kpss_


class pfpld:
    """
    PFPFLD模型类，用于人脸关键点检测。
    """

    def __init__(self, cpu=False, model_path='./resources'):
        """
        初始化PFPFLD关键点检测器。
        :param cpu: 是否强制使用CPU。
        :param model_path: 存放ONNX模型的目录路径。
        """
        onnx_path = f"{model_path}/pfpld_robust_sim_bs1_8003.onnx"
        try:
            providers = ['CPUExecutionProvider'] if cpu else ['CUDAExecutionProvider']
            self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            # 字节码中的异常处理比较奇特，这里使用更标准的模式
            raise Exception(f"load onnx failed: {e}")

        self.input_name = self.ort_session.get_inputs()[0].name

    def forward(self, input_image):
        """
        对单张人脸图像进行前向传播，获取68个关键点。
        :param input_image: 裁剪后的人脸图像 (numpy array)。
        :return: 68个关键点坐标 (numpy array, shape [68, 2])。
        """
        size = input_image.shape

        # 预处理
        img_resized = cv2.resize(input_image, (112, 112))
        img_tensor = (img_resized / 255.0).astype(np.float32)
        img_tensor = img_tensor.transpose((2, 0, 1))
        img_tensor = img_tensor[np.newaxis, :, :, :]  # 添加batch维度

        ort_inputs = {self.input_name: img_tensor}
        pred = self.ort_session.run(None, ort_inputs)

        # 后处理
        # pred[1] 包含98个关键点
        pred = convert98to68(pred[1])
        pred = pred.reshape(-1, 68, 2) * size[:2][::-1]

        return pred


def convert98to68(list_info):
    """
    将98点的人脸关键点转换为68点格式。
    """
    # list_info 可能是 shape [1, 196] 或更大的numpy数组
    points = list_info[0, :196]
    info_68 = []

    # 0-16: 턱선 (Jawline)
    # 字节码中第一段循环的索引计算为 j*4，这很奇怪，通常应为 j*2。
    # 但为了忠实于字节码，这里保留了它。这可能是一个错误或特殊的格式。
    # 如果出现问题，请尝试将下面的 j * 2 * 2 改为 j * 2。
    for j in range(17):
        x = points[j * 2 * 2 + 0]
        y = points[j * 2 * 2 + 1]
        info_68.append(x)
        info_68.append(y)

    # 17-21: 右眉 (Right eyebrow)
    for j in range(33, 38):
        x = points[j * 2]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)

    # 22-26: 左眉 (Left eyebrow)
    for j in range(42, 47):
        x = points[j * 2]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)

    # 27-35: 鼻子 (Nose)
    for j in range(51, 61):  # 实际上是10个点，但68点格式鼻子只有9个点，这里可能包含鼻翼
        x = points[j * 2]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)

    # 36-41: 右眼 (Right eye) - 通过计算得到
    point_38_x = (float(points[120]) + float(points[124])) / 2.0
    point_38_y = (float(points[121]) + float(points[125])) / 2.0
    point_39_x = (float(points[124]) + float(points[128])) / 2.0
    point_39_y = (float(points[125]) + float(points[129])) / 2.0
    point_41_x = (float(points[128]) + float(points[132])) / 2.0
    point_41_y = (float(points[129]) + float(points[133])) / 2.0
    point_42_x = (float(points[120]) + float(points[132])) / 2.0
    point_42_y = (float(points[121]) + float(points[133])) / 2.0

    # 42-47: 左眼 (Left eye) - 通过计算得到
    point_44_x = (float(points[136]) + float(points[140])) / 2.0
    point_44_y = (float(points[137]) + float(points[141])) / 2.0
    point_45_x = (float(points[140]) + float(points[144])) / 2.0
    point_45_y = (float(points[141]) + float(points[145])) / 2.0
    point_47_x = (float(points[144]) + float(points[148])) / 2.0
    point_47_y = (float(points[145]) + float(points[149])) / 2.0
    point_48_x = (float(points[136]) + float(points[148])) / 2.0
    point_48_y = (float(points[137]) + float(points[149])) / 2.0

    # 按顺序添加眼睛关键点
    # 右眼
    info_68.append(point_38_x)
    info_68.append(point_38_y)
    info_68.append(point_39_x)
    info_68.append(point_39_y)
    info_68.append(points[128])
    info_68.append(points[129])
    info_68.append(point_41_x)
    info_68.append(point_41_y)
    info_68.append(point_42_x)
    info_68.append(point_42_y)
    # 左眼
    info_68.append(points[136])
    info_68.append(points[137])
    info_68.append(point_44_x)
    info_68.append(point_44_y)
    info_68.append(point_45_x)
    info_68.append(point_45_y)
    info_68.append(points[144])
    info_68.append(points[145])
    info_68.append(point_47_x)
    info_68.append(point_47_y)
    info_68.append(point_48_x)
    info_68.append(point_48_y)

    # 48-67: 嘴巴 (Mouth)
    for j in range(76, 96):
        x = points[j * 2]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)

    # 字节码显示可能还有其他信息（如姿态角）被附加在后面
    if len(list_info) > 196:
        for j in range(len(list_info[196:])):
            info_68.append(list_info[196 + j])

    return np.array(info_68)