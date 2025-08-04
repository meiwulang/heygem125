#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: guiji
Created on Thu Jan 21 11:27:22 2021

此脚本定义了一个用于视频预处理的高度优化流程，主要针对人脸进行处理，
是唇形同步或人脸动画项目中的一个核心模块。

该版本在原始功能基础上进行了多项性能优化：
1.  **即时编译 (JIT)**: 使用 Numba 库对计算密集型函数进行JIT编译，大幅提升数值计算速度。
2.  **现代并发模型**: 采用 `concurrent.futures.ThreadPoolExecutor` 实现高效的批处理多线程。
3.  **向量化操作**: 充分利用 NumPy 的向量化特性，减少Python循环，提升数据处理效率。
4.  **智能数据结构**: 内部使用原生字典进行高速处理，仅在最后同步到共享字典以保持兼容性。

整体流程包括：人脸检测、头部姿态估计、边界框平滑、人脸裁剪和关键点提取。
"""

import re
from PIL import Image
from scipy import signal
import time, numpy as np, cv2
import multiprocessing.dummy as mp
import multiprocessing as real_mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
from numba import jit, prange
import warnings

# 忽略可能由某些库版本引起的警告
warnings.filterwarnings('ignore')

# 一个全局变量，可能用于存储视频处理相关的时间信息，但在此代码片段中未被使用。
video_time = []


# ==============================================================================
# JIT 编译的性能优化函数
# 使用 Numba 的 @jit 装饰器，将这些纯数值计算函数编译为原生机器码，
# 在循环中调用时可以获得数十倍甚至上百倍的性能提升。
# ==============================================================================

@jit(nopython=True, cache=True)
def fast_bbox_expansion(x1, y1, x2, y2, w, h, expand_ratio=0.1):
    """
    功能:
        (JIT编译) 快速计算边界框的扩展。为后续的人脸关键点检测提供更充足的上下文信息，
        避免人脸部分因边界框过紧而被裁切。

    实现关键点:
        - 使用`@jit(nopython=True)`装饰器，将此Python函数编译为无Python交互的机器码，以获得最大性能。
        - `cache=True`将编译结果保存到磁盘，避免了程序的重复编译。
        - 垂直方向上，通常只向下扩展(y2)，而不向上扩展(y1)，因为头部上方通常是头发，对人脸特征影响不大。

    可能的优化方向:
        - `expand_ratio`可以作为可配置参数传入，以适应不同的人脸姿态和场景。
        - 可以考虑非对称扩展，例如水平方向扩展比例大于垂直方向。

    入参:
        x1 (int/float): 原始边界框左上角x坐标。
        y1 (int/float): 原始边界框左上角y坐标。
        x2 (int/float): 原始边界框右下角x坐标。
        y2 (int/float): 原始边界框右下角y坐标。
        w (int): 原始图像宽度。
        h (int): 原始图像高度。
        expand_ratio (float): 扩展比例。

    返回值:
        tuple[int, int, int, int]: (new_x1, new_y1, new_x2, new_y2) 扩展后的新边界框坐标。
    """
    expansion_x = (x2 - x1) * expand_ratio
    expansion_y = (y2 - y1) * expand_ratio

    new_x1 = max(0, int(x1 - expansion_x))
    new_y1 = max(0, y1)
    new_x2 = min(w, int(x2 + expansion_x))
    new_y2 = min(h, int(y2 + expansion_y))

    return new_x1, new_y1, new_x2, new_y2


# ... (其他JIT函数的详细注释) ...
@jit(nopython=True, cache=True)
def fast_3dmm_bounds(xmin, ymin, w_rect, h_rect, img_w, img_h):
    """
    功能:
        (JIT编译) 快速计算用于3DMM头部姿态估计的裁剪边界。这个边界框的设计是为了
        创建一个接近标准化的头部视图，以便姿态估计模型能更准确地工作。

    实现关键点:
        - 计算基于关键点得到的矩形的中心点`x_c`和宽高比`wh_ratio`。
        - 所有的扩展和收缩都基于关键点的宽度`w_rect`和宽高比`wh_ratio`，这使得计算
          对人脸的尺度和比例具有一定的鲁棒性。
        - 比例系数(0.8, 0.35, 1.25)是经过调试的经验值，旨在捕捉完整的头部轮廓。

    可能的优化方向:
        - 这些经验比例系数可以作为模型的超参数进行调整和优化。
        - 对于极端姿态，可以设计一套动态调整比例系数的逻辑。

    入参:
        xmin (int/float): 由关键点计算出的边界框的左x坐标。
        ymin (int/float): 由关键点计算出的边界框的上y坐标。
        w_rect (int/float): 关键点边界框的宽度。
        h_rect (int/float): 关键点边界框的高度。
        img_w (int): 原始图像宽度。
        img_h (int): 原始图像高度。

    返回值:
        tuple[int, int, int, int]: (Xmin_3dmm, Xmax_3dmm, Ymin_3dmm, Ymax_3dmm) 用于头部姿态估计的裁剪区域坐标。
    """
    if h_rect == 0: h_rect = 1
    wh_ratio = w_rect / h_rect
    x_c = xmin + w_rect * 0.5

    height_factor = w_rect / wh_ratio
    half_width = height_factor * 0.8

    Xmin_3dmm = max(0, int(x_c - half_width))
    Xmax_3dmm = min(img_w, int(x_c + half_width))
    Ymin_3dmm = max(0, int(ymin - height_factor * 0.35))
    Ymax_3dmm = min(img_h, int(ymin + height_factor * 1.25))

    return Xmin_3dmm, Xmax_3dmm, Ymin_3dmm, Ymax_3dmm


@jit(nopython=True, cache=True)
def fast_crop_bounds(xmin, ymin, w, img_w, img_h, wh_ratio_factor):
    """
    功能:
        (JIT编译) 快速计算最终的人脸裁剪边界。这个边界框的目标是为下游的唇形同步模型
        提供一个内容稳定、大小一致的人脸图像。

    实现关键点:
        - 与`fast_3dmm_bounds`类似，但使用了不同的经验比例系数(0.75, 0.15, 1.35)。
        - `wh_ratio_factor` (等于1/wh) 被传入，用于将裁剪区域调整到期望的宽高比。
        - 最终的裁剪区域通常会聚焦于嘴部周围，同时包含足够的人脸上下文。

    可能的优化方向:
        - 同样，这些比例系数可以作为超参数进行优化。
        - 可以根据检测到的关键点（如鼻子和下巴的位置）动态调整裁剪中心，而不仅仅是`x_c`。

    入参:
        xmin, ymin, w: 由关键点计算出的边界框的左上角坐标和宽度。
        img_w, img_h: 原始图像的宽高。
        wh_ratio_factor (float): 预计算的目标宽高比的倒数 (1.0 / wh)。

    返回值:
        tuple[int, int, int, int]: (Xmin, Xmax, Ymin, Ymax) 最终用于裁剪的区域坐标。
    """
    x_c = xmin + w * 0.5
    height_factor = w * wh_ratio_factor
    half_width = height_factor * 0.75

    Xmin = max(0, int(x_c - half_width))
    Xmax = min(img_w, int(x_c + half_width))
    Ymin = max(0, int(ymin - height_factor * 0.15))
    Ymax = min(img_h, int(ymin + height_factor * 1.35))

    return Xmin, Xmax, Ymin, Ymax


@jit(nopython=True, cache=True, parallel=True)
def fast_landmark_transform(landmarks, xmin, ymin, scale_x, scale_y):
    """
    功能:
        (JIT编译, 并行) 快速将关键点坐标从原图坐标系变换到裁剪后的归一化坐标系。
        这是数据预处理的关键一步，确保关键点与裁剪后图像的对应关系。

    实现关键点:
        - `parallel=True`和`prange`的使用，让Numba自动将这个循环在多个CPU核心上并行执行，
          对于拥有68个或更多关键点的情况能提供额外加速。
        - 计算非常直接：先平移（减去裁剪框的左上角坐标），然后缩放（乘以缩放比例）。

    可能的优化方向:
        - 当前实现已经非常高效。进一步的优化可能需要切换到GPU（例如使用CuPy），但对于
          这个任务量来说，CPU并行通常已经足够。

    入参:
        landmarks (np.ndarray): 形状为 (N, 2) 的关键点数组，N是关键点数量。
        xmin (float): 裁剪框的左x坐标。
        ymin (float): 裁剪框的上y坐标。
        scale_x (float): 水平方向的缩放因子 (target_size / crop_width)。
        scale_y (float): 垂直方向的缩放因子 (target_size / crop_height)。

    返回值:
        np.ndarray: 形状为 (N, 2) 的变换后关键点数组，其坐标范围在 [0, target_size] 内。
    """
    result = np.zeros_like(landmarks)
    for i in prange(landmarks.shape[0]):
        result[i, 0] = (landmarks[i, 0] - xmin) * scale_x
        result[i, 1] = (landmarks[i, 1] - ymin) * scale_y
    return result


@jit(nopython=True, cache=True, parallel=True)
def fast_pose_check(head_poses, pose_threshold):
    """
    功能:
        (JIT编译, 并行) 快速检查头部姿态（俯仰、偏航、翻滚角）是否在可接受的阈值范围内。
        用于过滤掉侧脸过大或姿态不佳的帧，这些帧可能会导致模型预测失败或效果不佳。

    实现关键点:
        - `prange`在此处并行效果可能不明显（因为只有3次迭代），但无害。
        - 逻辑清晰：只要有一个角度超出范围，就立即返回`False`。

    可能的优化方向:
        - 可以设计更复杂的姿态评分机制，而不是简单的硬阈值判断。例如，一个姿态分数可以
          是三个角度偏离理想值（如0度）的加权和。

    入参:
        head_poses (np.ndarray): 形状为 (3,) 的数组，包含 [pitch, yaw, roll]。
        pose_threshold (np.ndarray): 形状为 (3, 2) 的数组，包含每个角度的[min, max]阈值。

    返回值:
        bool: 如果姿态在阈值内，返回`True`，否则返回`False`。
    """
    for i in prange(3):
        if not (pose_threshold[i, 0] < head_poses[i] < pose_threshold[i, 1]):
            return False
    return True


class op:
    """
    Operator (操作器) 类，封装了所有视频帧的人脸预处理操作。
    此版本经过高度优化，采用了JIT编译、批处理、多线程和向量化技术。
    """

    def __init__(self, caped_img2, wh, scrfd_detector, scrfd_predictor, hp, lm3d_std, img_size, driver_flag):
        """
        功能:
            初始化 op 类的实例。负责设置所有参数、加载模型引用、预计算常用值，
            并准备好处理所需的数据结构。

        实现关键点:
            - **双字典策略**: 内部使用高速的Python原生字典`self.data_dict`进行所有计算，
              以避免多进程/多线程共享字典`mp.Manager.dict`带来的巨大性能开销。
              `self.mp_dict`仅为保持外部API兼容性而存在。
            - **预计算**: 在构造函数中预先计算`self.wh_ratio_factor`和`self.target_size_float`等值，
              避免在处理循环中重复计算。
            - **批处理大小定义**: 动态计算一个合理的批处理大小`self.batch_size`，以平衡
              任务粒度和线程调度开销。

        可能的优化方向:
            - 模型引用(`scrfd_detector`等)可以考虑进行懒加载，只在第一次使用时加载，
              以减少初始化时间。
            - 可以增加一个配置对象(config object)来管理所有超参数（如阈值、比例因子），
              而不是将它们硬编码在代码中。

        入参:
            caped_img2 (dict): 输入的图像数据字典，键为帧索引(int)，值为包含`'imgs_data'`的字典。
            wh (float): 目标人脸区域的期望宽高比 (width / height)。
            scrfd_detector (object): 预加载的人脸检测模型实例。
            scrfd_predictor (object): 预加载的人脸关键点预测模型实例。
            hp (object): 预加载的头部姿态估计模型实例。
            lm3d_std (any): 3D标准人脸关键点数据（此代码中未使用，为兼容性保留）。
            img_size (int): 最终裁剪图像的目标边长尺寸（例如 256）。
            driver_flag (bool): 驱动视频标志。如果为True，会对最终图像进行BGR到RGB的颜色空间转换。
        """
        self.data_dict = dict(caped_img2)

        self.manager = mp.Manager
        self.mp_dict = self.manager().dict()
        self.mp_dict.update(self.data_dict)

        self.img_size = img_size
        self.target_size = self.img_size + int(self.img_size / 256) * 10

        self.wh = wh
        self.scrfd_detector = scrfd_detector
        self.scrfd_predictor = scrfd_predictor
        self.hp = hp
        self.pose_threshold = np.array([[-70, 50], [-100, 100], [-70, 70]], dtype=np.float64)
        self.driver_flag = driver_flag
        self.no_face = []

        self.wh_ratio_factor = 1.0 / self.wh if self.wh != 0 else 1.0
        self.target_size_float = float(self.target_size)

        self.batch_size = min(8, max(1, len(caped_img2) // (real_mp.cpu_count() or 1)))

    def smooth_optimized(self):
        """
        功能:
            对所有帧的边界框序列进行时间上的平滑处理。此函数的目的是消除由人脸检测器
            在逐帧处理时产生的边界框抖动，使得最终的裁剪视频更加稳定、无跳动感。

        实现关键点:
            - **数据填充**: 首先，遍历所有帧，将边界框数据填充到一个NumPy数组`bbx_smooth`中。
              对于没有检测到人脸的帧，使用前一帧的数据进行填充，以保证序列的连续性。
            - **卷积平滑**: 使用`scipy.signal.convolve2d`和一个5帧的移动平均核(`conv_core`)
              对边界框坐标序列进行卷积操作，这本质上是一个低通滤波器，能有效滤除高频抖动。
            - **智能抖动修正**: 这是此算法的核心。平滑虽然能去抖动，但也可能“抹平”真实的
              快速运动，导致边界框滞后于人脸。此步骤通过计算平滑前后值的差异，找出差异
              过大的点（`diff_mask`），在这些点上恢复使用原始的、未平滑的数据。这就在
              “消除抖动”和“保留真实运动”之间取得了很好的平衡。
            - **二次平滑**: 在修正后的数据上再次进行卷积平滑，目的是为了平滑掉上一步骤中
              因恢复原始值而可能引入的新的小跳变。
            - **向量化**: 整个过程高度向量化，使用NumPy操作代替Python循环，性能极高。

        可能的优化方向:
            - **卡尔曼滤波 (Kalman Filter)**: 对于边界框平滑任务，卡尔曼滤波器是一个理论上
              更优的模型。它可以对边界框的位置、速度甚至加速度进行建模和预测，能够更好地
              处理遮挡和快速运动，但实现也更复杂。
            - **自适应卷积核**: 可以根据检测到的运动幅度动态调整卷积核的大小。运动快时用小核
              （响应快），静止时用大核（平滑效果好）。

        返回值:
            None: 此函数直接修改`self.data_dict`中各帧的`'bounding_box_p'`字段，无返回值。
        """
        keylist = np.array(sorted(self.data_dict.keys()))
        if len(keylist) == 0: return

        bbx_smooth = np.zeros((len(keylist), 4), dtype=np.float32)

        for i, key in enumerate(keylist):
            bbox = self.data_dict[key].get("bounding_box_p", np.array([]))
            if bbox.size == 4:
                bbx_smooth[i, :] = bbox
            elif i > 0:
                bbx_smooth[i, :] = bbx_smooth[i - 1, :]

        conv_core = np.full((5, 1), 0.2, dtype=np.float32)

        try:
            bbx_smooth2 = signal.convolve2d(bbx_smooth, conv_core, boundary="symm", mode="same")
            diff_values = np.sum(np.abs(bbx_smooth2 - bbx_smooth), axis=1)
            diff_mask = diff_values > 12.0
            bbx_smooth3 = np.where(diff_mask[:, np.newaxis], bbx_smooth, bbx_smooth2)
            bbx_smooth4 = signal.convolve2d(bbx_smooth3, conv_core, boundary="symm", mode="same")

            for i, key in enumerate(keylist):
                self.data_dict[key]["bounding_box_p"] = bbx_smooth4[i, :].astype(np.int32)
        except Exception as e:
            print(f"Smoothing failed: {e}. Skipping smoothing.")
            for i, key in enumerate(keylist):
                self.data_dict[key]["bounding_box_p"] = bbx_smooth[i, :].astype(np.int32)

    def flow_optimized(self):
        """
        功能:
            编排整个高度优化的主处理流程。它负责任务的分解、并行执行和结果的同步，
            是整个预处理工作的“总指挥”。

        实现关键点:
            - **三阶段流程**: 流程被清晰地划分为三个阶段：
              1. 并行人脸检测与姿态过滤。
              2. 串行边界框平滑（因为平滑需要所有帧的数据）。
              3. 并行最终裁剪与特征提取。
            - **批处理与并发**: 使用`ThreadPoolExecutor`和`as_completed`来高效地管理
              线程池。将所有帧分成多个批次（`key_batches`），每个线程处理一个批次，
              大大减少了任务调度的开销。`as_completed`可以确保一旦有任何一个批次
              完成，主线程就可以立即处理其结果，而无需等待其他批次。
            - **错误处理**: 在`as_completed`循环中对`future.result()`进行`try...except`
              包裹，即使某个批次的处理失败，也不会中断整个流程，增强了程序的鲁棒性。

        可能的优化方向:
            - **流水线并行 (Pipeline Parallelism)**: 对于非常长的视频，可以实现流水线
              并行。例如，当第一批次的检测完成后，可以立即开始对其进行平滑和裁剪，
              而不需要等待所有帧的检测都完成。这需要更复杂的同步机制。
            - **混合精度训练/推理**: 如果模型支持，可以在推理时使用FP16混合精度，
              这在支持Tensor Core的NVIDIA GPU上能带来显著的速度提升。
            - **模型蒸馏**: 如果推理速度是极致瓶颈，可以考虑使用知识蒸馏技术，
              训练一个更小、更快的学生模型来替代现有的`scrfd`等模型。

        返回值:
            None: 此函数 orchestrates the entire process and modifies `self.data_dict`
            in place. It does not return anything.
        """
        keys = list(self.data_dict.keys())
        if not keys: return

        key_batches = [keys[i:i + self.batch_size] for i in range(0, len(keys), self.batch_size)]
        max_workers = min(len(key_batches), (real_mp.cpu_count() or 1) * 2)

        # 阶段一: 并行执行人脸检测
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(self.process_face_detection_batch, batch): batch for batch in
                               key_batches}
            for future in as_completed(future_to_batch):
                try:
                    self.data_dict.update(future.result())
                except Exception as e:
                    print(f"A face detection batch failed: {e}")

        # 阶段二: 单线程执行边界框平滑
        self.smooth_optimized()

        # 阶段三: 并行执行人脸裁剪
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(self.process_face_crop_batch, batch): batch for batch in key_batches}
            for future in as_completed(future_to_batch):
                try:
                    self.data_dict.update(future.result())
                except Exception as e:
                    print(f"A face crop batch failed: {e}")

        # 最终同步
        self.sync_data_to_mp_dict()

    def get_max_face_vectorized(self, face_boxes):
        """(向量化) 从多个人脸框中选择面积最大的一个。"""
        if face_boxes.shape[0] == 1:
            return face_boxes[0].astype(np.int32)

        areas = (face_boxes[:, 2] - face_boxes[:, 0]) * (face_boxes[:, 3] - face_boxes[:, 1])
        return face_boxes[np.argmax(areas)].astype(np.int32)

    def process_face_detection_batch(self, idx_batch):
        """(批处理) 对一个批次的帧进行人脸检测和姿态验证。"""
        results = {}
        for idx in idx_batch:
            try:
                # 注意：这里所有对类属性的访问都需要加上 self.
                loc_dict = self.data_dict[idx].copy()
                img = loc_dict['imgs_data']
                h, w = img.shape[:2]

                face_boxes, _ = self.scrfd_detector.get_bboxes(img)

                if face_boxes.shape[0] > 0:
                    x1, y1, x2, y2, score = self.get_max_face_vectorized(face_boxes)
                    x1, y1, x2, y2 = fast_bbox_expansion(x1, y1, x2, y2, w, h)

                    face_img = img[y1:y2, x1:x2]
                    if face_img.size == 0:
                        loc_dict['bounding_box_p'] = np.array([], dtype=np.int32)
                    else:
                        pots = self.scrfd_predictor.forward(face_img)[0]
                        landmarks = pots.astype(np.int32) + np.array([x1, y1])

                        xmin, ymin, w_rect, h_rect = cv2.boundingRect(landmarks)

                        Xmin_3dmm, Xmax_3dmm, Ymin_3dmm, Ymax_3dmm = fast_3dmm_bounds(xmin, ymin, w_rect, h_rect, w, h)

                        head_pose_crop = img[Ymin_3dmm:Ymax_3dmm, Xmin_3dmm:Xmax_3dmm]
                        if head_pose_crop.size > 0:
                            head_poses = self.hp.get_head_pose(head_pose_crop)
                            if fast_pose_check(head_poses, self.pose_threshold):
                                loc_dict['bounding_box_p'] = np.array([y1, y2, x1, x2], dtype=np.int32)
                            else:
                                loc_dict['bounding_box_p'] = np.array([], dtype=np.int32)
                        else:
                            loc_dict['bounding_box_p'] = np.array([], dtype=np.int32)
                else:
                    loc_dict['bounding_box_p'] = np.array([], dtype=np.int32)

                results[idx] = loc_dict
            except Exception as e:
                print(f"Error processing face detection for idx {idx}: {e}")
                results[idx] = self.data_dict[idx]
        return results

    def process_face_crop_batch(self, idx_batch):
        """(批处理) 使用平滑后的边界框，对一个批次的帧进行最终的裁剪和特征提取。"""
        results = {}
        for idx in idx_batch:
            try:
                loc_dict = self.data_dict[idx].copy()
                img = loc_dict["imgs_data"]
                dets = loc_dict.get("bounding_box_p", np.array([]))

                if dets.size == 0 or np.max(dets) == 0:
                    self.no_face.append(idx)
                    dets = np.array([0, 100, 0, 100], dtype=np.int32)
                    loc_dict["bounding_box_p"] = np.zeros(4, dtype=np.int32)

                y1, y2, x1, x2 = dets.astype(np.int32)

                face_region = img[y1:y2, x1:x2]
                if face_region.size > 0:
                    face_landmarks = self.scrfd_predictor.forward(face_region)[0]
                    landmarks = face_landmarks + np.array([x1, y1])
                    loc_dict["landmarks"] = landmarks

                    xmin, ymin, w, h = cv2.boundingRect(landmarks.astype(np.int32))

                    Xmin, Xmax, Ymin, Ymax = fast_crop_bounds(xmin, ymin, w, img.shape[1], img.shape[0],
                                                              self.wh_ratio_factor)
                    loc_dict["bounding_box"] = np.array([Ymin, Ymax, Xmin, Xmax], dtype=np.int32)

                    if (Xmax - Xmin) > 0 and (Ymax - Ymin) > 0:
                        scale_x = self.target_size_float / (Xmax - Xmin)
                        scale_y = self.target_size_float / (Ymax - Ymin)
                        lm_crop = fast_landmark_transform(landmarks, Xmin, Ymin, scale_x, scale_y)

                        img_crop = img[Ymin:Ymax, Xmin:Xmax]

                        interpolation = cv2.INTER_CUBIC
                        resized_img = cv2.resize(img_crop, (self.target_size, self.target_size),
                                                 interpolation=interpolation)

                        if self.driver_flag:
                            loc_dict["crop_img"] = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                        else:
                            loc_dict["crop_img"] = resized_img

                        loc_dict["crop_lm"] = lm_crop

                results[idx] = loc_dict
            except Exception as e:
                print(f"Error processing face crop for idx {idx}: {e}")
                results[idx] = self.data_dict[idx]
        return results

    # ... 其他兼容性方法 ...
    def sync_data_to_mp_dict(self):
        """将内部高速字典的结果同步回共享字典，以保持API兼容性。"""
        self.mp_dict.clear()
        self.mp_dict.update(self.data_dict)

    def show(self):
        """(兼容) 打印共享字典内容，用于调试。"""
        for idx in sorted(self.mp_dict.keys()):
            print(f"Frame {idx}: {self.mp_dict[idx]}")

    def smooth_(self):
        """(兼容) 调用优化的平滑方法。"""
        self.smooth_optimized()
        self.sync_data_to_mp_dict()

    def flow(self):
        """(兼容) 调用优化的主流程方法。"""
        self.flow_optimized()

    def get_results(self):
        """获取最终处理结果的便捷方法。"""
        return dict(self.data_dict)