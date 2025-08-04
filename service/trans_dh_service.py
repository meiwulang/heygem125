# trans_dh_service.py

"""
@project : face2face_train
@author  : huyi
@file   : trans_dh_service.py
@ide    : PyCharm
@time   : 2023-12-06 14:47:11
"""
import gc
import multiprocessing
import os
import subprocess
import threading
import time
import traceback
from enum import Enum
from multiprocessing import Process, set_start_method, Queue
from queue import Empty, Full

import cv2
import librosa
import torch
# 从自定义的 cv2box 工具库中导入 CVImage 类，可能封装了图像处理操作
from cv2box import CVImage
# 从 cv2box 中导入多线程/多进程处理工具，如 Linker, Queue, CVVideoWriterThread
from cv2box.cv_gears import Linker, Queue, CVVideoWriterThread
# 从人脸检测工具库中导入 FaceDetect 和 pfpld 模型
from face_detect_utils.face_detect import FaceDetect, pfpld
# 从人脸检测工具库中导入头部姿态估计模型
from face_detect_utils.head_pose import Headpose
# 从人脸库中导入一个包含5个关键点的人脸检测对齐类
from face_lib.face_detect_and_align import FaceDetect5Landmarks
from face_lib.face_detect_and_align import face_align_utils
# 从人脸库中导入 GFPGAN 模型，用于人脸修复和增强
from face_lib.face_restore import GFPGAN
# 从自定义工具库中导入自定义异常类
from h_utils.custom import CustomError
# 从自定义工具库中导入文件下载函数
from h_utils.request_utils import download_file
# 从自定义工具库中导入文件清理函数
from h_utils.sweep_bot import sweep
# 从数字人项目库中导入核心模型接口
from landmark2face_wy.digitalhuman_interface import DigitalHumanModel
# 导入3DMM和音频预处理操作
from preprocess_audio_and_3dmm import op
# 从 Wenet 相关库中导入特征提取和模型加载函数
from wenet.compute_ctc_att_bnf import get_weget, load_ppg_model
# 从自定义工具库中导入全局配置、日志和许可证检查
from y_utils.config import GlobalConfig
from y_utils.logger import logger
from y_utils.lcr import check_lc
# 导入服务注册相关函数
from .server import register_host, repost_host
import os
import cv2
import numpy as np
import hashlib
import pickle
import json
import concurrent.futures
from dataclasses import dataclass, field, asdict
import time as a_time # 避免与 time 模块重名
from typing import Optional # 用于类型提示，表示字段可以是可选的
from oss_utils import get_oss_manager

DUMMY_AUDIO_PATH = "dummy_silent_audio.wav"

@dataclass
class Task:
    """
    定义一个进入系统的任务对象。
    这个对象将被放入不同的请求队列中，代表一个具体的工作单元。
    """

    # --- 任务必要参数 ---
    # 这些参数在提交任何类型的任务时都必须提供，尽管在某些场景下可能是占位符。

    task_id: str
    """任务的唯一标识符，由API层生成，用于跟踪整个任务生命周期。"""

    audio_url: str
    """
    音频文件的来源URL或本地路径。
    - 在“合成任务”中，这是驱动口型的音频。
    - 在“预处理任务”中，此字段可以是一个无意义的占位符（如 "dummy.wav"），因为该任务不处理音频。
    """

    video_url: str
    """
    视频文件的来源URL或本地路径。
    - 在“预处理任务”中，这是需要被处理的原始视频。
    - 在“合成任务”中，这将是经过预处理后，保存在服务器上的本地视频文件路径。
    """

    # --- 任务路由与状态传递 ---

    task_type: str = "synthesis"
    """
    【新增核心字段】任务的类型，用于将任务路由到正确的CPU工作池。
    - 'preprocess': 表示这是一个视频预处理任务，将被发送到 `preprocess_request_queue`。
    - 'synthesis': 表示这是一个音频驱动的视频合成任务，将被发送到 `synthesis_request_queue`。
    - 默认值为 'synthesis'，可以为现有逻辑提供一定的向后兼容性。
    """

    model_id: Optional[str] = None
    """
    【新增核心字段】模型的唯一标识符，通常是视频内容的哈希值。
    - 在“预处理任务”的生命周期中，此字段在任务开始时为 None，在处理过程中被计算并填充。
    - 在提交“合成任务”时，此字段必须由API层提供，用于指定使用哪个预处理好的视频缓存。
    """

    # --- 任务可选参数 ---
    # 这些参数为任务提供了额外的配置选项，都有默认值。

    watermark_switch: int = 0
    """水印开关。0: 关闭, 1: 开启。"""

    digital_auth: int = 0
    """数字人标识开关。0: 关闭, 1: 开启。"""

    chaofen: int = 0
    """超分（分辨率增强）开关。0: 关闭, 1: 开启。"""

    pn: int = 0
    """乒乓（Ping-Pong）模式开关。0: 关闭 (循环播放), 1: 开启 (来回播放)。"""

    # --- 任务元数据 ---
    # 这些字段由系统在任务处理过程中动态填充，用于跟踪状态。

    status: str = "pending"
    """任务的当前状态，例如 "pending", "preprocessing_cpu", "processing_gpu", "success", "error"。"""

    progress: int = 0
    """任务的完成进度，以百分比表示 (0-100)。"""

    result_path: str = ""
    """
    任务的最终结果路径。
    - 对于“合成任务”，这是最终生成的视频文件的URL或路径。
    - 对于“预处理任务”，这里可以用来存储最终生成的 `model_id`。
    """

    error_message: str = ""
    """如果任务失败，这里会记录详细的错误信息。"""

    submit_time: float = field(default_factory=a_time.time)
    """任务被提交到系统的时间戳，用于计算总耗时等。"""



class CacheManager:
    """
    【新增】预处理结果缓存管理类
    负责为视频生成唯一ID，并处理缓存的读取、写入和校验。
    """

    def __init__(self, cache_root='./data'):
        self.cache_root = cache_root
        if not os.path.exists(self.cache_root):
            os.makedirs(self.cache_root)
            logger.info(f"缓存根目录 '{self.cache_root}' 已创建。")

    def get_video_id(self, video_path):
        """使用SHA256哈希算法为视频文件生成唯一ID。"""
        hasher = hashlib.sha256()
        with open(video_path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_cache_dir(self, video_id):
        """获取指定视频ID的缓存目录路径。"""
        return os.path.join(self.cache_root, video_id)

    def check_cache_valid(self, cache_dir):
        """检查缓存目录及其内容的完整性。"""
        if not os.path.isdir(cache_dir):
            return False

        required_files = ['face_data_dict.pkl', 'spline_masks.npz', 'face_coords.npz', 'metadata.json']
        for filename in required_files:
            if not os.path.exists(os.path.join(cache_dir, filename)):
                logger.warning(f"缓存校验失败: 在 '{cache_dir}' 中找不到文件 '{filename}'。")
                return False

        logger.info(f"缓存目录 '{cache_dir}' 校验通过。")
        return True

    def load_cache(self, cache_dir):
        """从缓存目录加载所有预处理数据。"""
        try:
            with open(os.path.join(cache_dir, 'face_data_dict.pkl'), 'rb') as f:
                face_data_dict_all = pickle.load(f)

            spline_masks_data = np.load(os.path.join(cache_dir, 'spline_masks.npz'))
            spline_masks_all = [spline_masks_data[key] for key in
                                sorted(spline_masks_data.keys(), key=lambda x: int(x.split('_')[1]))]

            face_coords_data = np.load(os.path.join(cache_dir, 'face_coords.npz'))
            face_coords_all = [face_coords_data[key] for key in
                               sorted(face_coords_data.keys(), key=lambda x: int(x.split('_')[1]))]

            with open(os.path.join(cache_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)

            logger.info(f"成功从 '{cache_dir}' 加载缓存。")
            return face_data_dict_all, spline_masks_all, face_coords_all, metadata
        except Exception as e:
            logger.error(f"从 '{cache_dir}' 加载缓存时出错: {e}")
            return None, None, None, None

    def save_cache(self, cache_dir, face_data_dict, spline_masks, face_coords, metadata):
        """将所有预处理数据原子化地保存到缓存目录。"""
        try:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            with open(os.path.join(cache_dir, 'face_data_dict.pkl'), 'wb') as f:
                pickle.dump(face_data_dict, f)

            np.savez_compressed(os.path.join(cache_dir, 'spline_masks.npz'),
                                **{f'arr_{i}': arr for i, arr in enumerate(spline_masks)})
            np.savez_compressed(os.path.join(cache_dir, 'face_coords.npz'),
                                **{f'arr_{i}': arr for i, arr in enumerate(face_coords)})

            with open(os.path.join(cache_dir, 'metadata.json'), 'w') as f:
                # 【修正】正确调用 json.dump，将 metadata_to_save 写入文件对象 f
                json.dump(metadata, f, indent=4)

            logger.info(f"预处理结果已成功缓存到 '{cache_dir}'。")
        except Exception as e:
            logger.error(f"保存缓存到 '{cache_dir}' 时出错: {e}")





def save_debug_image(work_id, frame_idx, step_name, image_data):
    """
    【新增核心工具】根据全局开关，保存调试过程中的图像。
    """
    cfg = GlobalConfig.instance()
    if not cfg.enable_debug_save:
        return

    try:
        # 确保图像数据是 BGR 格式的 uint8
        if image_data.dtype != np.uint8:
            # 处理蒙版 (单通道, float)
            if image_data.ndim == 2 and image_data.dtype == np.float32:
                image_data = (image_data * 255).astype(np.uint8)
            # 处理模型输出 (RGB, float/byte)
            else:
                if image_data.max() <= 1.0:
                    image_data = (image_data * 255)
                image_data = image_data.astype(np.uint8)
                if image_data.shape[2] == 3 and step_name.startswith(('03_', '04_')): # 假设模型输出为RGB
                     image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

        # 创建目录
        debug_dir = os.path.join(cfg.debug_output_dir, str(work_id), f'frame_{frame_idx:05d}')
        os.makedirs(debug_dir, exist_ok=True)

        # 保存文件
        file_path = os.path.join(debug_dir, f'{step_name}.png')
        cv2.imwrite(file_path, image_data)

    except Exception as e:
        # 打印错误但不要中断主流程
        print(f"警告: 无法保存调试图像 {work_id}/{frame_idx}/{step_name}. 错误: {e}")

def feature_extraction_wenet(audio_file, fps, wenet_model, mfccnorm=True, section=560000):
    """
    使用 Wenet 模型从音频文件中提取声学特征（如PPG）。
    它将音频与视频帧率对齐，为每一视频帧生成一个对应的特征窗口。

    Args:
        audio_file (str or np.array): 音频文件路径或已加载的波形数据。
        fps (int): 目标视频的帧率。
        wenet_model: 预加载的 Wenet 模型。
        mfccnorm (bool): 是否对MFCC进行归一化。
        section (int): 处理音频的分段大小。

    Returns:
        list: 包含每一帧对应声学特征的列表。
    """
    get_weget_start = time.time()

    rate = 16000  # 音频采样率固定为 16000 Hz
    win_size = 20  # 特征窗口大小
    if type(audio_file) == str:
        # 如果输入是文件路径，使用 librosa 加载音频
        sig, rate = librosa.load(audio_file, sr=16000, duration=None)
    else:
        # 否则，直接使用输入的波形数据
        sig = audio_file

    time_duration = len(sig) / rate  # 计算音频总时长（秒）
    # 根据视频帧率和音频时长，计算总帧数
    cnts = range(int(time_duration * fps))
    indexs = []  # 存储每一帧的特征
    # 调用 Wenet 提取整个音频的特征
    f_wenet_all = get_weget(audio_file, wenet_model, section)

    logger.info(f"【性能日志】Wenet特征提取(get_weget)完成, 耗时: {time.time() - get_weget_start:.4f} 秒")

    # 遍历每一视频帧，为其匹配一个音频特征窗口
    for cnt in cnts:
        # 计算当前帧在整个特征序列中的大致位置
        c_count = int(cnt / cnts[-1] * (f_wenet_all.shape[0] - 20)) + win_size // 2
        # 提取以 c_count 为中心的、大小为 win_size 的窗口
        indexs.append(f_wenet_all[c_count - win_size // 2:c_count + win_size // 2, ...])
    return indexs


def get_aud_feat1(wav_fragment, fps, wenet_model):
    """
    对 `feature_extraction_wenet` 的简单封装。
    """
    return feature_extraction_wenet(wav_fragment, fps, wenet_model)


def warp_imgs(imgs_data):
    """
    将图像列表转换为一个以索引为键、包含图像和索引的字典。
    这个结构可能是为了方便在多进程/多线程中追踪数据。

    Args:
        imgs_data (list): 图像帧列表。

    Returns:
        dict: 格式为 {0: {'imgs_data': frame0, 'idx': 0}, 1: ...} 的字典。
    """
    caped_img2 = {idx: {'imgs_data': it, 'idx': idx} for it, idx in zip(imgs_data, range(len(imgs_data)))}
    return caped_img2





def get_face_mask(mask_shape=(512, 512)):
    """
    生成一个柔边的椭圆形人脸蒙版。
    这个蒙版用于在融合生成的人脸和原始背景时，使得边缘过渡更自然。

    Args:
        mask_shape (tuple): 蒙版的尺寸。

    Returns:
        np.array: 一个单通道的浮点型蒙版图像，值域在 [0, 1]。
    """
    mask = np.zeros(mask_shape, dtype=np.float32)
    # 在黑色背景上画一个白色的实心椭圆
    cv2.ellipse(mask, (256, 256), (160, 220), 90, 0, 360, (255, 255, 255), -1)
    # 将边缘部分设置为0，创建一个边界
    thres = 20
    mask[:thres, :] = 0
    mask[-thres:, :] = 0
    mask[:, :thres] = 0
    mask[:, -thres:] = 0
    # 使用高斯模糊（这里是 stackBlur）使椭圆边缘变得平滑
    mask = cv2.stackBlur(mask, (201, 201))
    # 归一化到 [0, 1]
    mask = mask / 255.0
    # 调整尺寸并增加一个通道维度，以匹配图像格式
    mask = cv2.resize(mask, mask_shape)
    return mask[..., np.newaxis]


def get_single_face(bboxes, kpss, image, crop_size, mode='mtcnn_512', apply_roi=True):
    """
    从一张图像中检测到的多个人脸里，选出最主要的一个，并进行裁剪和对齐。

    Args:
        bboxes (list): 人脸边界框列表。
        kpss (list): 人脸关键点列表。
        image (np.array): 原始图像。
        crop_size (int): 裁剪后的目标尺寸。
        mode (str): 对齐模式。
        apply_roi (bool): 是否应用感兴趣区域（ROI）裁剪。

    Returns:
        tuple: (对齐后的人脸图像, 逆变换矩阵, ROI边界框)
    """
    if mode not in ('default', 'mtcnn_512', 'mtcnn_256', 'arcface_512', 'arcface', 'default_95'):
        raise AssertionError
    if bboxes.shape[0] == 0:
        return (None, None, None)

    # 根据检测分数选出最佳人脸
    det_score = bboxes[..., 4]
    best_index = np.argmax(det_score)

    new_kpss = None
    if kpss is not None:
        new_kpss = kpss[best_index]

    if apply_roi:
        # 应用ROI并进行标准化裁剪
        roi, roi_box, roi_kpss = face_align_utils.apply_roi_func(image, bboxes[best_index], new_kpss)
        align_img, mat_rev = face_align_utils.norm_crop(roi, roi_kpss, crop_size, mode=mode)
        return align_img, mat_rev, roi_box
    else:
        # 直接进行标准化裁剪
        align_img, M = face_align_utils.norm_crop(image, new_kpss, crop_size, mode=mode)
        return align_img, M, None


# --- 全局变量和初始化 ---
# get_firstface_frame 和 need_chaofen_flag 可能是用来缓存单次任务中是否需要超分的决策
get_firstface_frame = False
need_chaofen_flag = False
# face_mask 预先生成一个通用的椭圆形人脸蒙版，用于后续的图像融合
face_mask = get_face_mask(mask_shape=(512, 512))


def chaofen_src(frame_list, gfpgan, fd, frame_id, face_blur_detect, code):
    """
    人脸超分辨率处理函数。
    - 首先判断整个视频序列是否需要进行超分处理。
    - 如果需要，则对每一帧进行人脸检测、裁剪，然后用 GFPGAN 模型进行增强，最后贴回原图。

    Args:
        frame_list (list): 待处理的视频帧列表。
        gfpgan: GFPGAN 模型实例。
        fd: 人脸检测器实例。
        ...

    Returns:
        list: 处理（或未处理）后的视频帧列表。
    """
    global need_chaofen_flag, get_firstface_frame
    s_chao = time.time()

    # 决定是否需要超分 (仅在任务第一次调用时执行)
    if frame_id == 4 or not get_firstface_frame:
        s_blur_detect = time.time()
        chaofen_flag = False
        firstface_frame = False
        for frame in frame_list:
            if frame.shape[0] >= 3840 or frame.shape[1] >= 3840:
                chaofen_flag = False
                firstface_frame = True
                logger.info(f'[{code}] -> video frame shape is 4k, skip chaofen')
                break  # 4K视频不进行超分

            # 检测人脸并判断模糊度
            bboxes_scrfd, kpss_scrfd = fd.get_bboxes(frame)
            if len(bboxes_scrfd) == 0:
                continue

            face_image_, mat_rev_, roi_box_ = get_single_face(bboxes_scrfd, kpss_scrfd, frame, 512, 'mtcnn_512', True)
            face_attr_res = face_blur_detect.forward(face_image_)
            blur_threshold = face_attr_res[0][-2]
            logger.info(f'[{code}] -> frame_id:[{frame_id}] 模糊置信度:[{blur_threshold}]')
            if blur_threshold > GlobalConfig.instance().blur_threshold:
                logger.info(f'[{code}] -> need chaofen .')
                chaofen_flag = True
            else:
                chaofen_flag = False
            firstface_frame = True
            logger.info(f'[{code}] -> 前置超分决策完成, 耗时: {time.time() - s_blur_detect:.4f}s')
            break  # 仅根据第一帧或前几帧做决定


        need_chaofen_flag = chaofen_flag
        get_firstface_frame = firstface_frame

    # 如果不需要超分，直接返回原列表
    if not need_chaofen_flag:
        return frame_list

    # 执行超分流程
    new_frame_list = []
    for i in range(len(frame_list)):
        frame = frame_list[i]
        bboxes_scrfd, kpss_scrfd = fd.get_bboxes(frame)
        if len(bboxes_scrfd) == 0:
            new_frame_list.append(frame)
            continue

        face_image_, mat_rev_, roi_box_ = get_single_face(bboxes_scrfd, kpss_scrfd, frame, 512, 'mtcnn_512', True)
        # 使用 GFPGAN 进行人脸修复
        face_restore_out_ = gfpgan.forward(face_image_)
        # 将修复后的人脸贴回原图
        restore_roi = CVImage(None).recover_from_reverse_matrix(
            face_restore_out_, frame, roi_box_[1:3], roi_box_[0:2], mat_rev_, img_fg_mask=face_mask)
        frame[roi_box_[1]:roi_box_[3], roi_box_[0]:roi_box_[2]] = restore_roi
        new_frame_list.append(frame)

    torch.cuda.empty_cache()
    logger.info(f'[{code}] ->(可能是前置) 超分_src  cost:{time.time() - s_chao}s')
    return new_frame_list




class FaceDetectThread(Linker):
    """流水线第一步：人脸检测"""

    def __init__(self, queue_list):
        super().__init__(queue_list, fps_counter=True)
        self.fd = FaceDetect5Landmarks(mode='scrfd_500m')

    def forward_func(self, something_in):
        frame = something_in
        # 使用 scrfd_500m 模型检测人脸，设置最小边界框为64
        bboxes_scrfd, kpss_scrfd = self.fd.get_bboxes(frame, min_bbox_size=64)
        if len(bboxes_scrfd) == 0:
            return [frame, None, None, None]
        # 从可能的多个人脸中，裁剪出最主要的一个
        face_image_, mat_rev_, roi_box_ = self.fd.get_single_face(bboxes_scrfd, kpss_scrfd, frame, crop_size=512,
                                                                  mode='mtcnn_512', apply_roi=True)
        # 将原图、裁剪的人脸、逆变换矩阵、人脸框传递给下一阶段
        return [frame, face_image_, mat_rev_, roi_box_]


class FaceRestoreThread(Linker):
    """流水线第二步：人脸修复 (超分)"""

    def __init__(self, queue_list):
        super().__init__(queue_list, fps_counter=True)
        self.gfp = GFPGAN(model_type='GFPGANv1.4', provider='gpu')

    def forward_func(self, something_in):
        src_face_image_ = something_in[1]  # 获取上一阶段传来的人脸图像
        if src_face_image_ is None:
            # 如果没有检测到人脸，则直接传递 None
            return [None] + something_in
        # 使用 GFPGAN 进行修复
        face_restore_out_ = self.gfp.forward(src_face_image_)
        torch.cuda.empty_cache()
        # 将修复后的人脸和原始数据一起传递下去
        return [face_restore_out_] + something_in


class FaceParseThread(Linker):
    """流水线第三步：生成并传递人脸蒙版"""

    def __init__(self, queue_list):
        super().__init__(queue_list, fps_counter=True)
        # 在初始化时就生成好一个通用的蒙版，避免重复计算
        self.face_mask_ = self.get_face_mask(mask_shape=(512, 512))

    def get_face_mask(self, mask_shape):
        mask = np.zeros(mask_shape, dtype=np.float32)
        cv2.ellipse(mask, (256, 256), (160, 220), 90, 0, 360, (255, 255, 255), -1)
        thres = 20
        mask[:thres, :] = 0
        mask[-thres:, :] = 0
        mask[:, :thres] = 0
        mask[:, -thres:] = 0
        mask = cv2.stackBlur(mask, (201, 201))
        mask = mask / 255.0
        mask = cv2.resize(mask, mask_shape)
        return mask[..., np.newaxis]

    def forward_func(self, something_in):
        # 如果上一阶段没有人脸，直接传递
        if something_in[0] is None:
            return something_in + [None]
        # 将预先生成的蒙版附加到数据流中
        return something_in + [self.face_mask_]


class FaceReverseThread(Linker):
    """流水线第四步：将修复后的人脸贴回原图"""

    def __init__(self, queue_list):
        super().__init__(queue_list, fps_counter=True)
        self.counter = 0
        self.start_time = time.time()

    def forward_func(self, something_in):
        face_restore_out, src_img_in, _, mat_rev, roi_box, face_mask_ = something_in

        # 如果有人脸数据，则执行贴回操作
        if face_restore_out is not None:
            # 使用封装好的 CVImage 工具类进行逆向变换和融合
            restore_roi = CVImage(None).recover_from_reverse_matrix(
                face_restore_out, src_img_in,
                roi_box[1:3], roi_box[0:2],  # 裁剪区域的 y 和 x 坐标
                mat_rev, img_fg_mask=face_mask_
            )
            # 将融合后的 ROI 区域写回原图
            src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = restore_roi

        # 最终只返回处理完成的完整帧
        return [src_img_in]


def write_video_chaofen(output_imgs_queue, temp_dir, result_dir, work_id, audio_path, result_queue, width, height, fps,
                        watermark_switch, digital_auth):
    """
    【消费者进程】带有后置超分功能的视频写入进程。
    它内部创建了一个由 `FaceDetectThread`, `FaceRestoreThread` 等组成的
    多线程流水线，对接收到的每一帧进行超分处理，然后再写入视频。
    """
    output_mp4 = os.path.join(temp_dir, f'{work_id}-t.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result_path = os.path.join(result_dir, f'{work_id}-r.mp4')
    video_write = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))

    try:
        # 初始化超分流水线
        q0, q1, q2, q3, q4 = Queue(2), Queue(2), Queue(2), Queue(2), Queue(2)
        fdt = FaceDetectThread([q0, q1])
        frt = FaceRestoreThread([q1, q2])
        fpt = FaceParseThread([q2, q3])
        fret = FaceReverseThread([q3, q4])
        cvvwt = CVVideoWriterThread(video_write, [q4])  # 最终写入器
        threads_list = [fdt, frt, fpt, fret, cvvwt]
        [thread_.start() for thread_ in threads_list]

        while True:
            state, reason, value_ = output_imgs_queue.get()
            if isinstance(state, bool):
                if state:
                    logger.info(f'[{work_id}]视频帧队列处理已结束')
                    q0.put(None)  # 发送结束信号给流水线
                    [thread_.join() for thread_ in threads_list]
                    break
                else:
                    logger.error(f'[{work_id}]任务视频帧队列 -> 异常原因:[{reason}]')
                    q0.put(None)
                    [thread_.join() for thread_ in threads_list]
                    raise CustomError(reason)
            for result_img in value_:
                q0.put(result_img)  # 将帧放入流水线

        video_write.release()

        # 使用 ffmpeg 添加音频和水印
        command = ''
        s_ffmpeg = time.time()
        if watermark_switch == 1 and digital_auth == 1:
            command = f'ffmpeg -y -i {audio_path} -i {output_mp4} -i {GlobalConfig.instance().watermark_path} -i {GlobalConfig.instance().digital_auth_path} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10,overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {result_path}'
            logger.info(f'command:{command}')
        elif watermark_switch == 1:
            command = f'ffmpeg -y -i {audio_path} -i {output_mp4} -i {GlobalConfig.instance().watermark_path} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10" -c:a aac -crf 15 -strict -2 {result_path}'
        elif digital_auth == 1:
            command = f'ffmpeg -y -i {audio_path} -i {output_mp4} -i {GlobalConfig.instance().digital_auth_path} -filter_complex "overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {result_path}'
        else:
            command = f'ffmpeg -loglevel warning -y -i {audio_path} -i {output_mp4} -c:a aac -c:v libx264 -crf 15 -strict -2 {result_path}'

        subprocess.call(command, shell=True)
        logger.info(f"【性能日志】[{work_id}] (后置超分) FFMPEG 音视频合成完成，耗时: {time.time() - s_ffmpeg:.4f} 秒")
        print('###### write over')
        result_queue.put([True, result_path])

    except Exception as e:
        logger.error(f'[{work_id}]视频帧队列处理异常结束，异常原因:[{e.__str__()}]')
        result_queue.put([False, f'[{work_id}]视频帧队列处理异常结束，异常原因:[{e.__str__()}]'])

    logger.info('后处理进程结束')


def video_synthesis(output_imgs_queue):
    """
    一个简单的视频合成函数，用于实时显示处理结果，主要用于调试。
    """
    img_id = 0
    st = time.time()
    while output_imgs_queue.empty():
        pass
    et = time.time()
    print('表情迁移首次出现耗时======================:', et - st)

    while True:
        output_imgs = output_imgs_queue.get()
        for img in output_imgs:
            time.sleep(0.03125)  # 模拟32fps播放
            cv2.imshow('output_imgs', img)
            cv2.waitKey(1)
        st = time.time()




class Status(Enum):
    """
    定义任务状态的枚举类。
    """
    run = 1
    success = 2
    error = 3


def init_wh_process(in_queue, out_queue, gpu_id):
    """
    【后台进程】用于计算驱动视频中人脸的平均宽高比(w/h)。
    这个比例(wh)是数字人模型的一个重要参数。
    """

    try:
        logger.info(f'>>> init_wh_process: 接收到分配的 gpu_id: {gpu_id}')
        torch.cuda.set_device(gpu_id)
        current_device = torch.cuda.current_device()
        logger.info(f'>>> init_wh_process: 成功设置设备，当前实际设备为: cuda:{current_device}')
    except Exception as e:
        logger.error(f'>>> init_wh_process: 设置设备 cuda:{gpu_id} 失败! 错误: {e}', exc_info=True)

    face_detector = FaceDetect(cpu=False,
                               model_path='face_detect_utils/resources/')
    plfd = pfpld(cpu=False,
                 model_path='face_detect_utils/resources/')
    logger.info(f'>>> init_wh_process进程在 [cuda:{gpu_id}] 上启动')
    while True:
        try:
            code, driver_path = in_queue.get()
            s = time.time()
            wh_list = []
            cap = cv2.VideoCapture(driver_path)
            count = 0
            has_multi_face = False
            try:
                # 只处理视频的前100帧来计算平均值
                while cap.isOpened() and count < 100:
                    ret, frame = cap.read()
                    if not ret: break

                    bboxes, kpss = [], None
                    try:
                        bboxes, kpss = face_detector.get_bboxes(frame)
                    except Exception as e:
                        logger.error(f'[{code}]init_wh exception: {e}')

                    if len(bboxes) > 0:
                        if len(bboxes) > 1: has_multi_face = True

                        # 选择分数最高的人脸框
                        bbox = bboxes[0]
                        x1, y1, x2, y2, score = bbox.astype(int)

                        # 对人脸框进行轻微放大
                        x1 = max(x1 - int((x2 - x1) * 0.1), 0)
                        x2 = x2 + int((x2 - x1) * 0.1)
                        y2 = y2 + int((y2 - y1) * 0.1)
                        y1 = max(y1, 0)

                        # 裁剪出人脸区域
                        face_img = frame[y1:y2, x1:x2]

                        # 使用 pfpld 模型预测关键点
                        pots = plfd.forward(face_img)[0]
                        landmarks = np.array([[x + x1, y + y1] for x, y in pots.astype(np.int32)])

                        # 根据关键点计算精确的边界框
                        xmin, ymin, w, h = cv2.boundingRect(np.array(landmarks))

                        # 计算宽高比并存入列表
                        wh_list.append(w / h)

                    count += 1
            finally:
                cap.release()

            # 计算平均 wh 值
            wh = 0 if len(wh_list) == 0 else np.mean(np.array(wh_list))
            logger.info(f'[{code}]init_wh result :[{wh}]， cost: {time.time() - s} s')
            torch.cuda.empty_cache()
            out_queue.put([code, wh, has_multi_face])
        except Exception as e:
            print(traceback.format_exc())
            out_queue.put([f'init_wh，失败原因:[{e.args}]', '', False])
            torch.cuda.empty_cache()


def init_wh(code, drivered_path):
    """
    【废弃】`init_wh`的单进程版本，功能与`init_wh_process`类似，但在主流程中被多进程版本替代。
    """
    s = time.time()
    face_detector = FaceDetect(cpu=False, model_path='face_detect_utils/resources/')
    plfd = pfpld(cpu=False, model_path='face_detect_utils/resources/')
    wh_list = []
    cap = cv2.VideoCapture(drivered_path)
    count = 0
    try:
        while cap.isOpened() and count < 100:
            ret, frame = cap.read()
            if not ret: break
            try:
                bboxes, kpss = face_detector.get_bboxes(frame)
            except Exception as e:
                logger.error(f'[{code}]init_wh exception: {e}')

            if len(bboxes) > 0:
                bbox = bboxes[0]
                x1, y1, x2, y2, score = bbox.astype(int)
                x1 = max(x1 - int((x2 - x1) * 0.1), 0)
                x2 = x2 + int((x2 - x1) * 0.1)
                y2 = y2 + int((y2 - y1) * 0.1)
                y1 = max(y1, 0)
                face_img = frame[y1:y2, x1:x2]
                pots = plfd.forward(face_img)[0]
                landmarks = np.array([[x + x1, y + y1] for x, y in pots.astype(np.int32)])
                xmin, ymin, w, h = cv2.boundingRect(np.array(landmarks))
                wh_list.append(w / h)
            count += 1
    except Exception as e1:
        logger.error(f'[{code}]init_wh exception: {e1}')
    finally:
        cap.release()

    wh = 0 if len(wh_list) == 0 else np.mean(np.array(wh_list))
    logger.info(f'[{code}]init_wh result :[{wh}]， cost: {time.time() - s} s')
    torch.cuda.empty_cache()
    return wh


def get_video_info(video_file):
    """
    使用 OpenCV 获取视频的基本信息。

    Returns:
        tuple: (fps, width, height, fourcc)
    """
    cap = cv2.VideoCapture(video_file)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    return fps, width, height, fourcc


# 【新增函数】
def format_video(code, video_path, fourcc):
    """【新】只处理视频的格式化，不涉及音频。"""
    # 根据原始视频编码决定 ffmpeg 命令
    if fourcc in [cv2.VideoWriter_fourcc(*'H264'), cv2.VideoWriter_fourcc(*'avc1'), cv2.VideoWriter_fourcc(*'h264')]:
        # 如果已经是H264，只复制视频流，不重新编码，-an 表示移除所有音频
        ffmpeg_command = 'ffmpeg -loglevel warning -i "%s" -crf 15 -vcodec copy -an -y "%s"'
    else:
        # 其他格式则重新编码为libx264
        ffmpeg_command = 'ffmpeg -loglevel warning -i "%s" -c:v libx264 -crf 15 -an -y "%s"'

    video_format_path = os.path.join(GlobalConfig.instance().temp_dir, f'{code}_format.mp4')
    # 在路径两边加上引号，防止路径中包含空格导致命令失败
    os.system(ffmpeg_command % (video_path, video_format_path))
    if not os.path.exists(video_format_path):
        raise Exception(f'格式化视频失败: {video_path}')
    return video_format_path

# 【新增函数】
def format_audio(code, audio_path):
    """【新】只处理音频的格式化，转换为16kHz单声道WAV。"""
    audio_format_path = os.path.join(GlobalConfig.instance().temp_dir, f'{code}_format.wav')
    # -vn 表示移除所有视频
    ffmpeg_command_audio = 'ffmpeg -loglevel warning -i "%s" -ac 1 -ar 16000 -acodec pcm_s16le -vn -y "%s"'
    # 在路径两边加上引号
    os.system(ffmpeg_command_audio % (audio_path, audio_format_path))
    if not os.path.exists(audio_format_path):
        raise Exception(f'格式化音频失败: {audio_path}')
    return audio_format_path

def build_ffmpeg_command(cfg, task, audio_p, output_p, result_p):
    """
    一个辅助函数，根据任务配置构建最终的FFMPEG命令行字符串。
    """
    ffmpeg_command = ''
    watermark_path = cfg.watermark_path
    auth_path = cfg.digital_auth_path

    use_watermark = task.watermark_switch == 1 and watermark_path and os.path.exists(watermark_path)
    use_auth = task.digital_auth == 1 and auth_path and os.path.exists(auth_path)

    # 根据检查结果构建命令
    if use_watermark and use_auth:
        logger.info(f'[{task.task_id}] 任务需要水印和数字人标识')
        watermark_p = f'"{watermark_path}"'
        auth_p = f'"{auth_path}"'
        ffmpeg_command = f'ffmpeg -y -i {audio_p} -i {output_p} -i {watermark_p} -i {auth_p} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10,overlay=(main_w-overlay_w)-10:10" -c:a aac -crf {cfg.ffmpeg_crf} -strict -2 {result_p}'
    elif use_watermark:
        logger.info(f'[{task.task_id}] 任务需要水印')
        watermark_p = f'"{watermark_path}"'
        ffmpeg_command = f'ffmpeg -y -i {audio_p} -i {output_p} -i {watermark_p} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10" -c:a aac -crf {cfg.ffmpeg_crf} -strict -2 {result_p}'
    elif use_auth:
        logger.info(f'[{task.task_id}] 任务需要数字人标识')
        auth_p = f'"{auth_path}"'
        ffmpeg_command = f'ffmpeg -y -i {audio_p} -i {output_p} -i {auth_p} -filter_complex "overlay=(main_w-overlay_w)-10:10" -c:a aac -crf {cfg.ffmpeg_crf} -strict -2 {result_p}'
    else:
        # 默认的基础合并命令
        ffmpeg_command = f'ffmpeg -loglevel warning -y -i {audio_p} -i {output_p} -c:a aac -c:v libx264 -crf {cfg.ffmpeg_crf} -strict -2 {result_p}'

    return ffmpeg_command

def upload_to_oss_if_configured(local_result_path, task_id, status, task_status_dict):
    """
    一个辅助函数，检查OSS是否已配置。如果已配置，则尝试上传文件；
    否则，直接返回本地文件路径。
    """
    # 调用函数获取OSS管理器单例实例
    oss_manager = get_oss_manager()  # 确保 get_oss_manager 已被导入

    # 检查实例是否成功获取 (核心逻辑)
    if oss_manager:
        # 如果 oss_manager 有效，执行上传逻辑
        logger.info(f"[{task_id}] 开始上传到阿里云OSS: {local_result_path}")
        status.update({'status': 'uploading_to_oss', 'progress': 98})
        task_status_dict[task_id] = status

        success, oss_url = oss_manager.upload_video(local_result_path, task_id)

        if success:
            logger.info(f"[{task_id}] 文件成功上传到OSS: {oss_url}")
            return oss_url  # 上传成功，返回OSS URL
        else:
            logger.error(f"[{task_id}] 文件上传到OSS失败！将返回本地路径。")
            return local_result_path  # 上传失败，返回本地路径
    else:
        # 如果 oss_manager 为 None，跳过上传，直接使用本地路径
        logger.warning(f"[{task_id}] OSS未配置或初始化失败，跳过上传步骤。")
        return local_result_path  # OSS未配置，返回本地路径


def download_files(task: Task):
    """一个独立的工具函数，用于下载任务所需的文件。"""
    temp_dir = GlobalConfig.instance().temp_dir
    os.makedirs(temp_dir, exist_ok=True)          # ← 新增：目录不存在就创建

    try:
        if task.audio_url.startswith(('http:', 'https:')):
            _tmp_audio_path = os.path.join(temp_dir, f'{task.task_id}.wav')
            download_file(task.audio_url, _tmp_audio_path)
        else:
            _tmp_audio_path = task.audio_url
    except Exception as e:
        raise CustomError(f'[{task.task_id}] Audio download failed: {e}')

    try:
        if task.video_url.startswith(('http:', 'https:')):
            _tmp_video_path = os.path.join(temp_dir, f'{task.task_id}.mp4')
            download_file(task.video_url, _tmp_video_path)
        else:
            _tmp_video_path = task.video_url
    except Exception as e:
        raise CustomError(f'[{task.task_id}] Video download failed: {e}')

    return _tmp_audio_path, _tmp_video_path

#预处理进程
def _preprocessing_worker(video_path,
                          num_base_frames,
                          wh,
                          code,
                          cache_dir,
                          project_root_path,
                          gpu_id):

    """
    预处理工作进程 - 使用分批处理机制。

    职责:
    1. 接收视频路径和物理帧数。
    2. 分批次对视频的每一个物理帧执行3DMM、标准化等操作。
    3. 在这个连续的、无跳变的物理帧序列上进行健壮的漏检填充。
    4. 将只与物理帧相关的元数据存入缓存。
    """
    try:
        torch.cuda.set_device(gpu_id)

        # --- 【核心修正】在子进程中重新配置路径 ---
        from y_utils.config import GlobalConfig
        # 告诉 GlobalConfig 实例在当前进程中应该使用哪个根目录
        GlobalConfig.instance().force_set_project_root(project_root_path)
        cfg = GlobalConfig.instance()
        # --- 修正结束 ---

        logger.info(f"[{code}] [Preprocessing-Worker] 进程在 [cuda:{gpu_id}] 上启动。 PID: {os.getpid()}")
        logger.info(f"[{code}] [Preprocessing-Worker] Target: {num_base_frames} physical frames from {video_path}")

        # --- 新增：定义处理批次大小 ---
        # 这个值可以根据您的GPU显存大小进行调整，例如 32, 64 或 128
        BATCH_SIZE = GlobalConfig.instance().batch_size
        logger.info(f"[{code}] [Preprocessing-Worker] Using batch size: {BATCH_SIZE}")

        # 1. 在此进程内部，加载所有仅用于预处理的模型
        scrfd_detector = FaceDetect(cpu=False, model_path=cfg.face_detect_resources)
        scrfd_predictor = pfpld(cpu=False, model_path=cfg.face_detect_resources)
        hp = Headpose(cpu=False, onnx_path=cfg.head_pose_model)

        # --- 步骤A: 一次性加载所有物理帧到内存 (CPU RAM) ---
        # 这一步仍然保留，因为后续的漏检填充需要全局信息。如果CPU内存也成为瓶颈，则需进一步改造为逐批读帧。
        logger.info(f'[{code}] [Preprocessing-Worker] Loading all {num_base_frames} physical frames into memory...')
        physical_frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened() and len(physical_frames) < num_base_frames:
            ret, frame = cap.read()
            if not ret: break
            physical_frames.append(frame)
        cap.release()

        if len(physical_frames) != num_base_frames:
            logger.warning(
                f"[{code}] [Preprocessing-Worker] Video read mismatch. Expected {num_base_frames}, but got {len(physical_frames)} frames.")
            if not physical_frames:
                raise ValueError(f"Could not read any frames from video file: {video_path}")

        # --- 步骤B: 【核心改造】分批次执行3DMM特征提取 ---
        logger.info(f'[{code}] [Preprocessing-Worker] Starting BATCHED 3DMM feature extraction...')
        s_3dmm = time.time()

        # 用于收集所有批次的结果
        face_data_dict_all = {}

        # 按 BATCH_SIZE 步长循环
        for i in range(0, len(physical_frames), BATCH_SIZE):
            # 获取当前批次的帧
            batch_frames = physical_frames[i:i + BATCH_SIZE]

            logger.info(
                f"[{code}] [Preprocessing-Worker] Processing batch {i // BATCH_SIZE + 1}, frames {i} to {i + len(batch_frames) - 1}")

            # 1. 为当前批次准备数据
            caped_drivered_img_batch = warp_imgs(batch_frames)

            # 2. 对当前批次执行op
            drivered_op = op(caped_drivered_img_batch, wh, scrfd_detector, scrfd_predictor, hp, None, 256, False)
            drivered_op.flow()

            # 3. 收集结果，注意要调整字典的键，使其对应全局帧索引
            for batch_idx, frame_data in drivered_op.mp_dict.items():
                global_idx = i + batch_idx
                face_data_dict_all[global_idx] = frame_data

            # 4. 【关键】清理显存，为下一个批次做准备
            del drivered_op, caped_drivered_img_batch
            torch.cuda.empty_cache()

        logger.info(
            f"[{code}] [Preprocessing-Worker] BATCHED 3DMM processing completed. Cost: {time.time() - s_3dmm:.4f}s")

        # --- 步骤C: 在物理帧序列上执行健壮的漏检填充 ---
        logger.info(
            f'[{code}] [Preprocessing-Worker] Checking and filling missed detections on physical frame sequence...')

        bad_indices = {i for i, data in face_data_dict_all.items() if len(data.get('bounding_box_p', [])) != 4}

        if bad_indices:
            logger.warning(
                f'[{code}] [Preprocessing-Worker] Found {len(bad_indices)} missed detections. Starting smart fill...')
            good_indices = sorted(list(set(face_data_dict_all.keys()) - bad_indices))

            if not good_indices:
                raise RuntimeError(f"No valid faces detected in the entire video: {video_path}")

            # 使用我们之前讨论的“最近邻”智能填充算法
            import bisect
            for i in sorted(list(bad_indices)):
                insertion_point = bisect.bisect_left(good_indices, i)
                if insertion_point == 0:
                    best_good_idx = good_indices[0]
                elif insertion_point == len(good_indices):
                    best_good_idx = good_indices[-1]
                else:
                    prev_good_idx = good_indices[insertion_point - 1]
                    next_good_idx = good_indices[insertion_point]
                    best_good_idx = prev_good_idx if (i - prev_good_idx) <= (next_good_idx - i) else next_good_idx

                face_data_dict_all[i] = face_data_dict_all[best_good_idx].copy()

        logger.info(f'[{code}] [Preprocessing-Worker] Missed detection filling completed.')

        # --- 步骤D: 标准化与蒙版生成 ---
        # (这部分逻辑与之前完全一样，作用于每一个物理帧)
        logger.info(f'[{code}] [Preprocessing-Worker] Normalizing images/landmarks and generating spline masks...')
        STD_SIZE = 256
        spline_masks_all = []
        for i in range(len(physical_frames)):
            original_crop_img = face_data_dict_all[i]["crop_img"]
            original_lm = face_data_dict_all[i]["crop_lm"]
            original_h, original_w, _ = original_crop_img.shape
            std_crop_img = cv2.resize(original_crop_img, (STD_SIZE, STD_SIZE))
            scale_w, scale_h = STD_SIZE / original_w, STD_SIZE / original_h
            std_lm = original_lm.copy()
            std_lm[:, 0] *= scale_w
            std_lm[:, 1] *= scale_h
            std_mask = DigitalHumanModel.create_spline_mask(landmarks=std_lm,
                                                            img_shape=(STD_SIZE, STD_SIZE, 3),
                                                            profile='mouth_and_chin',
                                                            feather_kernel_size=35,
                                                            dilation_size=19,
                                                            horizontal_erosion_ratio=0.15)

            face_data_dict_all[i]['crop_img'] = std_crop_img
            face_data_dict_all[i]['crop_lm'] = std_lm
            spline_masks_all.append(std_mask)

        # --- 步骤E: 保存只包含物理帧元数据的“基础缓存” ---
        logger.info(f'[{code}] [Preprocessing-Worker] Saving base metadata to cache...')
        face_coords_all = [face_data_dict_all[i]['bounding_box'] for i in range(len(physical_frames))]
        metadata_to_save = {'wh': wh, 'timestamp': time.time(), 'num_base_frames': len(physical_frames)}

        cache_manager = CacheManager()
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)
        cache_manager.save_cache(cache_dir, face_data_dict_all, spline_masks_all, face_coords_all, metadata_to_save)

        logger.info(f"[{code}] [Preprocessing-Worker] Process finished successfully. Base cache saved to {cache_dir}.")

    except Exception as e:
        logger.error(f"[{code}] [Preprocessing-Worker] An unrecoverable error occurred: {e}", exc_info=True)
        # 将错误信息写入文件，以便主进程可以检测到失败
        error_file_path = os.path.join(os.path.dirname(cache_dir), f"{os.path.basename(cache_dir)}.error")
        with open(error_file_path, 'w', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        raise

#预处理工作进程
def preprocess_worker_loop(request_queue,
                           task_status_dict,
                           preprocess_results_dict,
                           init_wh_queue,
                           init_wh_queue_output,
                           gpu_id):
    """
    【“原材料加工线” - 视频预处理工作循环】

    这是一个在独立进程中运行的守护循环。它的唯一职责是：
    1. 从“预处理任务队列”中获取任务。
    2. 对任务中指定的视频进行下载、格式化、计算哈希（model_id）。
    3. 检查视频缓存是否存在，如果不存在，则启动一个一次性的、隔离的子进程 (`_preprocessing_worker`) 来执行CPU和GPU密集的预处理计算。
    4. 任务成功后，将预处理结果（特别是 model_id 和本地视频路径）注册到共享的 `preprocess_results_dict` 中，供后续的合成任务查询和使用。
    """
    cfg = GlobalConfig.instance()
    process_id = os.getpid()
    logger.info(f"[Preprocess-Worker-{process_id} on cuda:{gpu_id}] 进程已启动，准备接收视频预处理任务。")

    # 注意：这个工作循环内部可以预加载一些仅用于预处理的模型，但更推荐的做法是在
    # 一次性的 _preprocessing_worker 子进程中加载，以保持此守护进程的轻量和稳定。
    # 此处我们遵循在子进程中加载模型的最佳实践。

    while True:
        current_task = None
        try:
            # 1. 从专属的预处理请求队列中阻塞式地获取任务
            current_task = request_queue.get()

            # 收到 "毒丸" (None) 信号，优雅地退出循环
            if current_task is None:
                logger.info(f"[Preprocess-Worker-{process_id}] 收到退出信号，进程将终止。")
                break

            task_id = current_task.task_id
            logger.info(f"[{task_id}] [Preprocess-Worker] 领到新任务，开始处理视频: {current_task.video_url}")

            # 2. 更新任务初始状态
            status = task_status_dict[task_id]
            status.update({'status': 'preprocessing_downloading', 'progress': 5})
            task_status_dict[task_id] = status

            # 3. 下载并格式化视频文件
            # 预处理任务不关心音频，所以 audio_url 是一个占位符
            _, tmp_video_path = download_files(current_task)
            fps, _, _, fourcc = get_video_info(tmp_video_path)
            # 格式化视频，得到一个标准化的本地视频文件路径，这是后续所有操作的基础 只调用 format_video，不再需要提供占位音频
            local_video_path = format_video(task_id, tmp_video_path, fourcc)

            status.update({'status': 'preprocessing_hashing', 'progress': 15})
            task_status_dict[task_id] = status

            # 4. 计算视频内容的唯一标识符 (model_id)
            cache_manager = CacheManager(cache_root=cfg.cache_dir)
            model_id = cache_manager.get_video_id(local_video_path)
            cache_dir = cache_manager.get_cache_dir(model_id)
            logger.info(f"[{task_id}] 视频哈希 (model_id) 计算完成: {model_id}")

            # 5. 检查缓存，如果不存在，则启动重量级的预处理子进程
            if not cache_manager.check_cache_valid(cache_dir):
                logger.info(f'[{task_id}] 视频缓存不存在或无效，将启动独立的预处理子进程...')
                status.update({'status': 'preprocessing_3dmm', 'progress': 20})
                task_status_dict[task_id] = status

                # a. 获取 'wh' (宽高比) 参数
                init_wh_queue.put([task_id, local_video_path])
                wh_output = init_wh_queue_output.get(timeout=60)  # 等待wh计算结果
                if not wh_output or wh_output[0] != task_id:
                    raise CustomError(f"任务 {task_id} 的'wh'值计算错乱。")
                wh = wh_output[1]

                # 【方案B核心】在方案B中，_preprocessing_worker 只处理物理帧
                # 我们需要先获取物理帧数
                cap = cv2.VideoCapture(local_video_path)
                num_base_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                # b. 启动一次性的、隔离的子进程来完成所有繁重工作
                preproc_process = multiprocessing.Process(
                    target=_preprocessing_worker,
                    # 注意：这里的参数列表与您选择的 _preprocessing_worker 版本对应
                    # 这是方案B的参数列表
                    args=(local_video_path,
                          num_base_frames,
                          wh,
                          task_id,
                          cache_dir,
                          cfg.project_root,
                          gpu_id)
                )
                preproc_process.start()
                preproc_process.join(timeout=600)  # 等待子进程完成，设置超时

                # c. 检查子进程的执行结果
                if preproc_process.is_alive():
                    preproc_process.terminate()
                    raise CustomError(f"预处理子进程超时（超过600秒）。")
                if preproc_process.exitcode != 0:
                    # 可以在 _preprocessing_worker 中设计更详细的错误传递机制
                    raise CustomError(f"预处理子进程执行失败，退出码: {preproc_process.exitcode}。")

                logger.info(f'[{task_id}] 独立的预处理子进程已成功完成。')
            else:
                logger.info(f'[{task_id}] 发现有效的视频缓存，跳过重量级预处理步骤。')

            # 6. 【关键步骤】将成功的预处理结果注册到共享字典中
            # 无论缓存是否存在，只要流程成功走到这里，就应该确保注册信息
            preprocess_results_dict[model_id] = {
                'status': 'completed',
                'local_video_path': local_video_path,  # 供合成任务查找和使用
                'original_video_url': current_task.video_url,
                'created_at': time.time()
            }
            logger.info(f"模型ID: {model_id} 已成功注册到系统中。")

            # 7. 更新最终任务状态为成功
            status.update({'status': 'success', 'progress': 100, 'result_path': model_id})
            task_status_dict[task_id] = status
            logger.info(f"[{task_id}] 视频预处理任务圆满完成。")

        except Exception as e:
            # 统一的异常处理
            if current_task:
                task_id_in_exception = current_task.task_id
                error_message = f"视频预处理失败: {str(e)}"
                logger.error(f"[{task_id_in_exception}] [Preprocess-Worker] {error_message}", exc_info=True)

                # 更新任务状态为失败
                status = task_status_dict.get(task_id_in_exception, {})
                status.update({'status': 'error', 'error_message': error_message, 'progress': 0})
                task_status_dict[task_id_in_exception] = status
            else:
                logger.error(f"[Preprocess-Worker-{process_id}] 在获取任务前发生未知错误。", exc_info=True)

#合成准备工作进程
def synthesis_prep_worker_loop(request_queue, ready_queue, task_status_dict):
    """
    【“产品组装准备线” - 合成任务准备工作循环】

    这是一个在独立进程中运行的守护循环。它的职责是：
    1. 从“合成任务队列”中获取任务。
    2. 对任务中指定的音频进行下载、格式化。
    3. 加载 Wenet 模型，提取音频的声学特征（如 PPG）。
    4. 将包含（任务对象、音频特征、视频元信息、缓存目录路径等）所有合成所需信息的
       “轻量级交接包” (preprocessed_package)，放入 `ready_queue`。
    5. 下游的 `frame_dispatcher_loop` 将从 `ready_queue` 中消费这些物料包。
    """
    cfg = GlobalConfig.instance()
    process_id = os.getpid()
    logger.info(f"[SynthPrep-Worker-{process_id}] 进程已启动，准备接收合成任务。")

    # 在进程启动时，一次性加载 Wenet 模型到内存中。
    # 这是一个纯CPU操作，适合放在这里。
    wenet_model = None
    try:
        device = 'cpu'  # 音频处理通常在CPU上进行
        logger.info(f"[SynthPrep-Worker-{process_id}] 正在加载 Wenet 模型到设备 '{device}'...")
        wenet_model = load_ppg_model(
            cfg.wenet_config_path,
            cfg.wenet_model_path,
            device
        )
        logger.info(f"[SynthPrep-Worker-{process_id}] Wenet 模型加载成功，工作进程准备就绪。")
    except Exception as e:
        logger.error(f"[SynthPrep-Worker-{process_id}] 初始化 Wenet 模型失败，进程无法工作: {e}", exc_info=True)
        return  # 如果模型加载失败，此worker将无法工作，直接退出。

    while True:
        current_task = None
        try:
            # 1. 从专属的合成请求队列中阻塞式地获取任务
            current_task = request_queue.get()

            # 收到 "毒丸" (None) 信号，优雅地退出循环
            if current_task is None:
                logger.info(f"[SynthPrep-Worker-{process_id}] 收到退出信号，进程将终止。")
                break

            task_id = current_task.task_id
            logger.info(f"[{task_id}] [SynthPrep-Worker] 领到新任务，开始准备合成数据...")

            # 2. 更新任务初始状态
            status = task_status_dict[task_id]
            status.update({'status': 'synthesis_preparing_audio', 'progress': 5})
            task_status_dict[task_id] = status

            # 3. 下载和格式化音频文件
            # 对于合成任务，video_url 已经是本地路径，由API层在提交时传入，此处无需处理。
            # 我们只需要处理 audio_url。
            tmp_audio_path, _ = download_files(current_task)
            #只调用 format_audio，不再需要提供占位视频
            local_audio_path = format_audio(task_id, tmp_audio_path)

            status.update({'status': 'synthesis_extracting_feature', 'progress': 15})
            task_status_dict[task_id] = status

            # 4. 提取音频特征
            # 使用任务中的本地视频路径来获取正确的帧率(fps)
            fps, width, height, fourcc = get_video_info(current_task.video_url)
            audio_wenet_feature = get_aud_feat1(local_audio_path, fps=fps, wenet_model=wenet_model)
            total_frames = len(audio_wenet_feature)
            logger.info(f"[{task_id}] 音频特征提取完成，目标视频总帧数: {total_frames}")

            # 5. 获取视频的物理帧数 (用于后续的PN模式等计算)
            cap = cv2.VideoCapture(current_task.video_url)
            num_base_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # 6. 【关键步骤】准备并打包“轻量级交接包”
            # 根据 model_id 推算出缓存目录路径
            cache_manager = CacheManager(cache_root=cfg.cache_dir)
            cache_dir = cache_manager.get_cache_dir(current_task.model_id)

            preprocessed_package = {
                'task': current_task,
                'cache_dir': cache_dir,
                'audio_features': audio_wenet_feature,
                'video_url': current_task.video_url,  # 这是本地路径
                'num_base_frames': num_base_frames,
                'total_frames': total_frames,
                'video_info': (fps, width, height, fourcc),
                'tmp_audio_path': local_audio_path  # 使用格式化后的wav路径，供ffmpeg最后合并
            }

            # 7. 将准备好的物料包放入 ready_queue，供下游的帧分发器(frame_dispatcher_loop)消费
            ready_queue.put(preprocessed_package)

            # 8. 更新任务状态，表示已成功分发给下一阶段
            status.update({'status': 'waiting_dispatch', 'progress': 45})
            task_status_dict[task_id] = status
            logger.info(f"[{task_id}] 合成数据准备完成，已提交至帧分发队列。")

        except Exception as e:
            # 统一的异常处理
            if current_task:
                task_id_in_exception = current_task.task_id
                error_message = f"合成任务准备失败: {str(e)}"
                logger.error(f"[{task_id_in_exception}] [SynthPrep-Worker] {error_message}", exc_info=True)

                # 更新任务状态为失败
                status = task_status_dict.get(task_id_in_exception, {})
                status.update({'status': 'error', 'error_message': error_message, 'progress': 0})
                task_status_dict[task_id_in_exception] = status
            else:
                logger.error(f"[SynthPrep-Worker-{process_id}] 在获取任务前发生未知错误。", exc_info=True)

#帧分发工作进程
def frame_dispatcher_loop(ready_queue, chunk_queue, task_status_dict, batch_size):
    """
    [方案B - 最终版] 帧分发器进程的主循环 (智能数据合成中心)。

    这是一个在独立进程中运行的守护循环。它的职责如下：
    1. 从 `ready_queue` 获取由 `synthesis_prep_worker` 准备好的“轻量级施工图纸”。
    2. 【按需加载】根据图纸中的引用（cache_dir, video_url），从磁盘加载当前任务所需的全部
       “基础资源”（预处理缓存数据 + 物理视频帧）到本进程的内存中。
    3. 根据任务的详细参数（如音频长度、PN模式），实时地、分块地从内存中的基础资源
       合成GPU所需的数据包。
    4. 将合成好的、包含实际数据的小数据块（chunk）放入 `chunk_queue`，供GPU工作者消费。
    5. 一个任务的所有数据块分发完毕后，主动释放内存，并等待下一个任务。
    """
    cfg = GlobalConfig.instance()
    process_id = os.getpid()
    logger.info(f"[FrameDispatcher-{process_id}] 进程已启动，等待就绪的合成任务。")

    while True:
        current_task_id = "None"
        preprocessed_package = None
        # 定义需要手动清理的大数据变量
        base_frames = None
        base_face_data_dict = None
        base_spline_masks = None
        base_face_coords = None

        try:
            # 1. 等待并获取一个“轻量级施工图纸”
            preprocessed_package = ready_queue.get()

            # 收到“毒丸”信号，优雅地退出循环
            if preprocessed_package is None:
                logger.info(f"[FrameDispatcher-{process_id}] 收到退出信号，正在通知下游并关闭。")
                # 将毒丸信号继续传递给所有GPU workers，确保它们也能正常退出
                chunk_queue.put(None)
                break

            # 2. 解包所有指令和元信息
            task = preprocessed_package['task']
            current_task_id = task.task_id
            cache_dir = preprocessed_package['cache_dir']
            audio_wenet_feature = preprocessed_package['audio_features']
            video_url = preprocessed_package['video_url']
            num_base_frames = preprocessed_package['num_base_frames']
            total_frames = preprocessed_package['total_frames']
            pn_mode = (task.pn == 1)

            logger.info(f"[{current_task_id}] [FrameDispatcher] 任务PN模式: {pn_mode} (从 task.pn={task.pn} 读取)")
            # 更新任务状态
            status = task_status_dict[current_task_id]
            status.update({'status': 'dispatching_loading_data', 'progress': 46})
            task_status_dict[current_task_id] = status

            # 3. 【核心逻辑：按需加载任务所需资源到内存】
            # a. 从磁盘加载预处理缓存数据
            cache_manager = CacheManager(cache_root=cfg.cache_dir)
            base_face_data_dict, base_spline_masks, base_face_coords, _ = cache_manager.load_cache(cache_dir)
            if base_face_data_dict is None:
                raise CustomError(f"帧分发器未能从缓存目录 {cache_dir} 加载元数据。")

            # b. 从磁盘加载物理视频帧
            base_frames = []
            cap = cv2.VideoCapture(video_url)
            # 确保只读取num_base_frames数量的帧
            while len(base_frames) < num_base_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                base_frames.append(frame)
            cap.release()

            if len(base_frames) != num_base_frames:
                raise ValueError(
                    f"视频 '{video_url}' 帧数不匹配。期望 {num_base_frames} 帧, 实际读到 {len(base_frames)} 帧。")

            logger.info(
                f"[{current_task_id}] [FrameDispatcher] 基础资源加载完成。内存中已准备好 {len(base_frames)} 视频帧及相关缓存数据。")

            # 4. 发送“任务初始化”信号，通知下游的 video_writer_manager 准备写入
            init_package = {
                'type': 'task_init',
                'task_id': current_task_id,
                'data': {
                    'task': task,
                    'video_info': preprocessed_package['video_info'],
                    'tmp_audio_path': preprocessed_package['tmp_audio_path'],
                    'total_frames': total_frames
                }
            }
            chunk_queue.put(init_package)

            # 5. 进入分块循环，从内存中的“基础资源”实时合成并发送数据块
            logger.info(f"[{current_task_id}] [FrameDispatcher] 开始向GPU分发 {total_frames} 帧的数据块...")
            status.update({'status': 'dispatching_frames', 'progress': 48})
            task_status_dict[current_task_id] = status

            for i in range(0, total_frames, batch_size):
                start_idx = i
                end_idx = min(i + batch_size, total_frames)
                this_batch_size = end_idx - start_idx
                if this_batch_size <= 0:
                    continue

                # a. 实时合成当前批次所需的所有数据
                frames_batch, face_dict_batch_list, spline_masks_batch, face_coords_batch = [], [], [], []

                for frame_index in range(start_idx, end_idx):
                    # 【核心索引计算】根据是否为PN模式，计算出当前帧应该使用的物理帧索引

                    final_idx = 0

                    if num_base_frames > 1:
                        if pn_mode:
                            # 一个完整的来回周期长度是 2 * (N-1)
                            # 例如 N=4, 序列是 0,1,2,3,2,1 (长度6)
                            cycle_len = (num_base_frames - 1) * 2

                            idx_in_cycle = frame_index % cycle_len

                            if idx_in_cycle < num_base_frames:
                                # 处于正向播放部分 (0, 1, ..., N-1)
                                final_idx = idx_in_cycle
                            else:
                                # 处于反向播放部分
                                # 例如 N=4, cycle_len=6. 当 idx_in_cycle=4, 对应序列中的第5个元素，应为2.
                                # final_idx = cycle_len - idx_in_cycle
                                # N=4, idx=4 -> 6-4=2.
                                # N=4, idx=5 -> 6-5=1.
                                final_idx = cycle_len - idx_in_cycle
                        else:  # 非PN模式，即循环播放
                            final_idx = frame_index % num_base_frames

                    elif num_base_frames == 1:
                        # 如果视频只有一帧，无论如何都只能用第一帧
                        final_idx = 0

                    # 从内存中的“基础资源”中按需取用
                    frames_batch.append(base_frames[final_idx])
                    face_dict_batch_list.append(base_face_data_dict[final_idx])
                    spline_masks_batch.append(base_spline_masks[final_idx])
                    face_coords_batch.append(base_face_coords[final_idx][:4])  # 确保只取4个坐标

                # 将列表形式的人脸字典转换为推理所需的格式
                face_dict_batch = {k: v for k, v in enumerate(face_dict_batch_list)}
                audio_batch = audio_wenet_feature[start_idx:end_idx]

                # b. 打包成数据块并放入队列
                chunk_package = {
                    'type': 'data_chunk',
                    'task_id': current_task_id,
                    'start_frame_index': start_idx,
                    'chunk_data': (audio_batch, face_dict_batch, frames_batch, face_coords_batch, spline_masks_batch)
                }
                chunk_queue.put(chunk_package)

            # 6. 发送“数据块已发完”信号，通知 video_writer_manager 准备结束任务
            end_of_chunks_package = {'type': 'end_of_chunks', 'task_id': current_task_id}
            chunk_queue.put(end_of_chunks_package)

            logger.info(f"[{current_task_id}] [FrameDispatcher] 所有数据块已成功分发。")

        except Exception as e:
            # 统一的异常处理
            error_message = f"帧分发器在处理任务 {current_task_id} 时发生严重错误: {e}"
            logger.error(error_message, exc_info=True)
            if current_task_id != "None" and current_task_id in task_status_dict:
                status = task_status_dict[current_task_id]
                status.update({'status': 'error', 'error_message': error_message})
                task_status_dict[current_task_id] = status

        finally:
            # 【关键】无论成功还是失败，都确保在处理完一个任务后，主动释放大块内存
            # 这对于保持服务长期运行的稳定性至关重要
            if base_frames is not None:
                del base_frames
                base_frames = None
            if base_face_data_dict is not None:
                del base_face_data_dict
                base_face_data_dict = None
            if base_spline_masks is not None:
                del base_spline_masks
                base_spline_masks = None
            if base_face_coords is not None:
                del base_face_coords
                base_face_coords = None

            # 调用垃圾回收器，帮助系统更快地回收内存
            gc.collect()

    logger.info(f"[FrameDispatcher-{process_id}] 进程已关闭。")

#核心gpu工作进程
def gpu_worker_loop(gpu_id, chunk_queue, writer_command_queue, task_status_dict, batch_size):
    """
     GPU工作进程的主循环 (计算执行器)。

    职责:
    1. 绑定到指定的GPU并加载推理模型。
    2. 不断地从 chunk_queue 获取“数据块”。
    3. 如果是“数据块”，则执行GPU密集型的推理和合成。
    4. 如果是“初始化信号”，则直接转发给Video Writer Manager。
    5. 将处理结果（完成的帧块）或信号放入 writer_command_queue。
    """

    # --- 初始化阶段 (进程启动时执行一次) ---
    process_id = os.getpid()
    logger.info(f"[GPU-Worker-{gpu_id} / PID:{process_id}] Process started, binding to device cuda:{gpu_id}.")

    digital_human_model = None
    try:
        # 1. 绑定GPU
        torch.cuda.set_device(gpu_id)

        # 2. 加载此GPU专属的 DigitalHumanModel
        logger.info(f"[GPU-Worker-{gpu_id}] Loading DigitalHumanModel onto device cuda:{gpu_id}...")
        digital_human_model = DigitalHumanModel(
            blend_dynamic=GlobalConfig.instance().blend_dynamic,
            chaofen_before=0,
            gpu_id=gpu_id
        )
        logger.info(f"[GPU-Worker-{gpu_id}] Model loaded successfully. Worker is ready.")
    except Exception as e:
        logger.error(f"[GPU-Worker-{gpu_id}] Initialization failed: {e}", exc_info=True)
        return  # 初始化失败，进程退出

    # --- 主循环 (无限循环) ---
    while True:
        try:
            # 1. 等待并获取一个“包”（可能是数据块，也可能是信号）
            package = chunk_queue.get()

            if package is None:
                logger.info(f"[GPU-Worker-{gpu_id}] Received poison pill. Forwarding and shutting down.")
                # 将毒丸传递给下一个消费者（writer manager）
                writer_command_queue.put(None)
                break

            # 2. 【核心工作流】根据包的类型进行分流
            package_type = package.get('type')
            task_id = package.get('task_id')

            if package_type == 'data_chunk':
                # a. 如果是数据块，直接开始计算！
                start_frame_index = package['start_frame_index']
                batch_data = package['chunk_data'] # chunk_data is now batch_data

                # 获取当前块的大小，用于日志输出
                chunk_size = len(batch_data[2])  # original_frames_bgr
                end_frame_index = start_frame_index + chunk_size - 1

                logger.info(
                    f"[{task_id}] [GPU-{gpu_id}] Processing chunk for frames: {start_frame_index} - {end_frame_index} (Size: {chunk_size})")

                # 更新任务状态，表示GPU已开始处理
                if task_id in task_status_dict:
                    status = task_status_dict[task_id]
                    if status['status'] != 'processing_gpu':
                        status['status'] = 'processing_gpu'
                        status['progress'] = 50
                        task_status_dict[task_id] = status

                # [新增] 开始计时
                batch_start_time = time.time()

                # b. 调用推理函数
                #    注意：这里不再需要内部的批处理预取线程，因为块已经足够小
                final_frames_chunk = digital_human_model.inference_notraining_gpu_composite(
                    *batch_data,  # 解包元组作为参数
                    params=[task_id, start_frame_index]
                )

                # [新增] 结束计时并计算耗时
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                fps_batch = chunk_size / batch_duration if batch_duration > 0 else float('inf')

                # [新增] 打印带有任务ID的性能日志
                logger.info(
                    f"[{task_id}] [GPU-{gpu_id}] "
                    f"Batch frames {start_frame_index}-{end_frame_index} processed. "
                    f"Total time: {batch_duration:.4f}s, "
                    f"FPS: {fps_batch:.2f}"
                )

                # c. 将处理好的 batch 发送给写入管理器
                writer_command_queue.put({
                    'task_id': task_id,
                    'type': 'write_chunk',
                    # 【关键新增】附上这个数据块的起始帧索引
                    'start_frame_index': start_frame_index,
                    'data': final_frames_chunk
                })
                # logger.debug(f"[{task_id}] [GPU-{gpu_id}] Sent chunk {start_frame_index} to writer.")

            elif package_type == 'task_init':
                # d. 如果是初始化信号，我们不处理，直接“转发”给下一站
                logger.info(f"[{task_id}] [GPU-{gpu_id}] Forwarding 'task_init' signal to Video Writer.")
                writer_command_queue.put(package)

            elif package_type == 'end_of_chunks':
                # 【新增】收到了“数据块已发完”的信号
                logger.info(f"[{task_id}] [GPU-{gpu_id}] Received 'end_of_chunks' signal. Notifying writer.")
                # 直接将这个结束信号转发给 writer manager
                writer_command_queue.put(package)

        except Exception as e:
            # 异常处理
            task_id_in_exception = "unknown"
            if 'task_id' in locals() and task_id:
                task_id_in_exception = task_id
            logger.error(f"[GPU-Worker-{gpu_id}] Error processing a chunk for task {task_id_in_exception}: {e}",
                         exc_info=True)
            # 可以在这里向 writer_command_queue 发送一个任务失败信号，以便上层能感知到
            # writer_command_queue.put({'task_id': task_id_in_exception, 'command': 'error', 'data': str(e)})
        finally:
            # [新增] 无论是否异常，都在每次循环后尝试释放内存
            if 'package' in locals(): del package
            if 'chunk_data' in locals(): del batch_data
            if 'final_frames_chunk' in locals(): del final_frames_chunk
            # gc.collect() # 通常不需要，但可以作为最后的手段
    logger.info(f"[GPU-Worker-{gpu_id}] Shutting down.")

#视频写入工作进程
def video_writer_manager(writer_command_queue, result_queue, task_status_dict):
    """
    一个独立的、极其健壮的视频写入管理进程。

    该进程的核心职责:
    1. 接收上游GPU工作者发来的（可能乱序的）已处理帧数据块。
    2. 使用一个重排序缓冲区(`buffer`)，确保帧按正确的顺序被写入视频文件。
    3. 监听一个结束信号(`end_of_chunks`)，并在所有帧都写入完毕后，才执行最终的视频合成。
    4. 健壮地处理FFMPEG合成、OSS上传等收尾工作，并能优雅地处理其中可能发生的错误，而不会使自身崩溃。
    """
    # =============================================================
    #                  块 1: 初始化与辅助函数
    # =============================================================
    process_id = os.getpid()
    logger.info(f"[VideoWriterManager-{process_id}] 进程已启动，准备接收写入指令。")

    # 核心数据结构: 一个字典，用于管理所有正在并行进行的写入任务
    active_writers = {}

    def try_write_buffered_chunks(task_id):
        """
        核心辅助函数：检查并按顺序写入所有已到达的、连续的帧数据块。
        """
        if task_id not in active_writers:
            return

        writer_info = active_writers[task_id]
        buffer = writer_info['buffer']
        next_frame_to_write = writer_info['next_frame_to_write']
        writer = writer_info['writer']

        # 只要下一个期望的帧块在缓冲区里，就持续写入
        while next_frame_to_write in buffer:
            package = buffer.pop(next_frame_to_write)
            chunk_data = package['data']

            for frame in chunk_data:
                writer.write(frame)

            # 更新计数器，指向下一个期望的帧块的起始帧
            next_frame_to_write += len(chunk_data)
            # --- 【新增】更新最后一个块写入的时间 ---
            writer_info['last_chunk_write_time'] = time.time()

        # 将更新后的计数器写回任务信息中
        writer_info['next_frame_to_write'] = next_frame_to_write

    # =============================================================
    #                块 2: 任务终结与清理逻辑
    # =============================================================
    def finalize_task(task_id):
        """
        核心辅助函数：检查一个任务是否已满足所有结束条件，并执行最终处理。
        """
        if task_id not in active_writers:
            return

        writer_info = active_writers[task_id]
        # 预先定义一个变量，用于标记任务是否应该被清理
        should_cleanup = False

        # --- 条件1: 任务成功完成 ---
        # 必须同时满足：a) 已收到结束信号 b) 所有帧都已写入
        if writer_info.get('end_signal_received') and writer_info['next_frame_to_write'] >= writer_info['total_frames']:

            # --- 【新增】计算并打印 T10 耗时 ---
            first_arrival = writer_info.get('first_chunk_arrival_time')
            last_write = writer_info.get('last_chunk_write_time')
            if first_arrival and last_write:
                total_write_duration = last_write - first_arrival
                logger.info(
                    f"[{task_id}] [VideoWriterManager] T10: 视频写入总耗时 (从首个块到达至末个块写完): {total_write_duration:.4f}s")
            # --- 新增结束 ---

            logger.info(f"[{task_id}] 所有 {writer_info['total_frames']} 帧已写入。开始最终处理...")

            # 【优化】将所有可能抛出异常的收尾工作，都包裹在 try...except 中
            # 这可以防止因 FFMPEG 失败或网络问题导致整个管理进程崩溃。
            try:
                # 释放写入器，以便FFMPEG可以访问该文件
                writer_info['writer'].release()

                status = task_status_dict.get(task_id, {})
                status.update({'status': 'finalizing_video', 'progress': 95})
                task_status_dict[task_id] = status

                # --- 您提供的 FFMPEG 和 OSS 逻辑 ---
                cfg = GlobalConfig.instance()
                task = writer_info['task_object']

                # 构建 FFMPEG 命令... (此部分逻辑正确，保持不变)
                audio_p = f'"{os.path.normpath(writer_info["audio_path"])}"'
                output_p = f'"{os.path.normpath(writer_info["output_path"])}"'
                result_p = f'"{os.path.normpath(writer_info["result_path"])}"'
                ffmpeg_command = build_ffmpeg_command(cfg, task, audio_p, output_p, result_p)  # 进一步封装

                local_result_path = writer_info['result_path']
                output_directory = os.path.dirname(local_result_path)
                os.makedirs(output_directory, exist_ok=True)

                logger.info(f"[{task_id}] 执行 FFMPEG 命令: {ffmpeg_command}")
                subprocess.call(ffmpeg_command, shell=True)
                logger.info(f"[{task_id}] FFMPEG 处理完成。")

                # 检查输出文件是否存在，如果不存在则抛出异常
                if not os.path.exists(local_result_path):
                    raise FileNotFoundError(f"FFMPEG合成结束后，未找到结果文件: {local_result_path}")

                # 上传到 OSS...
                final_result_url = upload_to_oss_if_configured(local_result_path, task_id, status, task_status_dict)

                # 报告最终成功状态
                status['status'] = 'success'
                status['progress'] = 100
                status['result_path'] = final_result_url
                task_status_dict[task_id] = status
                result_queue.put(task_status_dict[task_id])

            except Exception as finalization_error:
                # 【健壮性优化】捕获收尾阶段的任何错误，并将其报告为任务失败
                error_message = f"在最终处理阶段失败: {finalization_error}"
                logger.error(f"[{task_id}] {error_message}", exc_info=True)
                status = task_status_dict.get(task_id, {})
                status['status'] = 'error'
                status['error_message'] = error_message
                task_status_dict[task_id] = status

            should_cleanup = True  # 标记任务（无论成功或失败）都需要清理

        # --- 统一的清理逻辑 ---
        if should_cleanup:
            if task_id in active_writers:
                # 释放资源
                if writer_info.get('writer'):
                    # 确保在 FFMPEG 之前已经 release 过了，这里是双重保险
                    if writer_info['writer'].isOpened():
                        writer_info['writer'].release()
                # 从活动字典中删除
                del active_writers[task_id]
                logger.info(f"[{task_id}] 任务已终结并清理。")

    # =============================================================
    #                       块 3: 主事件循环
    # =============================================================
    while True:
        command_package = None
        try:
            command_package = writer_command_queue.get()

            if command_package is None:
                break

            command = command_package.get('type')
            task_id = command_package.get('task_id')

            if command == 'task_init' and task_id:
                # 初始化新任务
                init_data = command_package['data']
                task = init_data['task']
                video_info = init_data['video_info']
                tmp_audio_path = init_data['tmp_audio_path']
                total_frames = init_data['total_frames']

                cfg = GlobalConfig.instance()
                output_mp4_path = os.path.normpath(os.path.join(cfg.temp_dir, f'{task_id}-t.mp4'))
                final_result_path = os.path.normpath(os.path.join(cfg.result_dir, f'{task_id}-r.mp4'))

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_mp4_path, fourcc, video_info[0], (video_info[1], video_info[2]))

                active_writers[task_id] = {
                    'writer': writer, 'output_path': output_mp4_path,
                    'result_path': final_result_path, 'audio_path': tmp_audio_path,
                    'task_object': task, 'total_frames': total_frames,
                    'buffer': {}, 'next_frame_to_write': 0,
                    'end_signal_received': False,  # 新增的状态标志

                    # --- 【新增】用于计时的字段 ---
                    'first_chunk_arrival_time': None,  # 记录第一个块到达的时间
                    'last_chunk_write_time': None  # 记录最后一个块写入完成的时间
                }
                logger.info(f"[{task_id}] 写入器已为 {total_frames} 帧创建。")

            elif command == 'write_chunk' and task_id in active_writers:
                # 收到一个数据块
                start_index = command_package['start_frame_index']
                writer_info = active_writers[task_id]

                # --- 【新增】记录第一个块到达的时间 ---
                if writer_info['first_chunk_arrival_time'] is None:
                    writer_info['first_chunk_arrival_time'] = time.time()

                writer_info['buffer'][start_index] = command_package
                writer_info['buffer'][command_package['start_frame_index']] = command_package

                try_write_buffered_chunks(task_id)
                finalize_task(task_id)  # 每次写入后都检查是否可以结束

            elif command == 'end_of_chunks' and task_id in active_writers:
                # 收到结束信号
                logger.info(f"[{task_id}] 收到数据结束信号，将等待所有数据块写入。")
                writer_info = active_writers[task_id]
                writer_info['end_signal_received'] = True

                finalize_task(task_id)  # 收到信号后也检查一次

        except Exception as e:
            # 捕获主循环中的意外错误
            task_id_in_exception = command_package.get('task_id', 'unknown') if command_package else 'unknown'
            error_message = f"VideoWriterManager在处理任务 {task_id_in_exception} 时发生意外错误: {e}"
            logger.error(error_message, exc_info=True)
            if task_id_in_exception in task_status_dict:
                status = task_status_dict.get(task_id_in_exception, {})
                status['status'] = 'error'
                status['error_message'] = f"视频写入或合成失败: {str(e)}"
                task_status_dict[task_id_in_exception] = status
            if task_id_in_exception in active_writers:
                active_writers[task_id_in_exception]['writer'].release()
                del active_writers[task_id_in_exception]

    # =============================================================
    #                     块 4: 进程善后处理
    # =============================================================
    for task_id, writer_info in list(active_writers.items()):
        logger.warning(f"[{task_id}] 在关闭时强制释放未完成的视频写入器。")
        writer_info['writer'].release()
        del active_writers[task_id]

    logger.info(f"[VideoWriterManager-{process_id}] 进程已关闭。")






class TransDhTask(object):

    def __init__(self, *args, **kwargs):
        """
        [最终架构] 初始化服务框架的所有核心组件。
        这个构造函数负责建立整个多进程架构的“神经网络”——即所有的通信队列和共享状态字典。
        """
        cfg = GlobalConfig.instance()
        logger.info('TransDhTask Service Framework Init: 正在构建多进程通信管道...')

        try:
            # 强制使用 'spawn' 启动方法，以避免在不同操作系统（特别是macOS和Windows）上出现fork相关的问题
            set_start_method('spawn', force=True)
        except RuntimeError:
            # 如果已经设置过，会抛出 RuntimeError，可以安全地忽略
            pass

        # 从全局配置中获取批处理大小
        self.batch_size = int(GlobalConfig.instance().batch_size)

        # 创建一个 Manager 对象，用于创建可以在进程间共享的数据结构
        manager = multiprocessing.Manager()

        logger.info("正在创建所有多进程队列和共享字典...")

        # --- 1.【关键改造】为两条CPU生产线创建独立的请求队列 ---
        self.preprocess_request_queue = manager.Queue(maxsize=100)
        """接收视频预处理（训练）任务的队列。"""

        self.synthesis_request_queue = manager.Queue(maxsize=100)
        """接收音频驱动合成任务的队列。"""

        # --- 2. 后续流程的队列 ---
        # "准备就绪"队列的容量由 合成准备进程数 和 一个因子 决定
        ready_queue_size = cfg.num_synthesis_prep_workers * cfg.ready_queue_factor
        self.ready_queue = manager.Queue(maxsize=ready_queue_size)
        """存放已准备好、等待被分发成帧块的合成任务“物料包”。"""

        # GPU流水线相关队列的容量由 GPU数量 和 一个因子 决定
        gpu_pipeline_queue_size = cfg.num_gpus * cfg.gpu_pipeline_queue_factor
        self.chunk_queue = manager.Queue(maxsize=gpu_pipeline_queue_size)
        """存放已经被切分好的、包含具体帧数据的“数据块”，等待GPU处理。"""

        self.writer_command_queue = manager.Queue(maxsize=gpu_pipeline_queue_size)
        """存放GPU处理完成的帧块，或发送给写入器的控制指令。"""

        # 最终结果队列
        self.result_queue = manager.Queue(maxsize=100)  # (写入器 -> 主进程/API层)
        """存放已完成任务的最终结果信息。"""

        # --- 3.【关键改造】为所有工作进程创建独立的管理列表 ---
        self.preprocess_workers = []
        """存放所有“视频预处理”工作进程对象的列表。"""

        self.synthesis_prep_workers = []
        """存放所有“合成准备”工作进程对象的列表。"""

        self.gpu_workers = []
        """存放所有GPU工作进程对象的列表。"""

        self.utility_workers = []
        """存放所有辅助工具进程（如帧分发器、视频写入器、wh计算器）对象的列表。"""

        # --- 4. 共享状态与结果字典 ---
        self.task_status_dict = manager.dict()
        """
        一个全局共享的字典，用于实时跟踪所有任务（无论类型）的状态。
        Key: task_id, Value: 包含状态、进度、错误信息等的字典。
        """

        self.preprocess_results_dict = manager.dict()
        """
        【新增】一个至关重要的共享字典，用于注册和查询已完成的预处理结果。
        Key: model_id (视频哈希), 
        Value: 包含 {'status': 'completed', 'local_video_path': '...'} 等信息的字典。
        这是连接“训练”和“合成”两个阶段的桥梁。
        """

        # --- 5. 辅助进程队列保持不变 ---
        self.init_wh_queue = manager.Queue(2)
        self.init_wh_queue_output = manager.Queue(2)

        logger.info("所有通信管道和共享状态字典已成功创建。服务框架基础结构搭建完毕。")

    def submit_preprocess_task(self, task: Task):
        """
        【公开方法】向“视频预处理”生产线提交一个新任务。

        这个方法是线程安全和进程安全的，因为它只与Manager创建的队列和字典交互。
        它将任务对象放入预处理队列，并初始化任务的实时状态。

        Args:
            task (Task): 一个已经实例化的 Task 对象，其 task_type 应为 'preprocess'。

        Returns:
            tuple[bool, str]: 一个元组，第一个元素表示提交是否成功，第二个是相关信息。
        """
        try:
            # 确保提交的任务类型是正确的
            if task.task_type != 'preprocess':
                warning_msg = f"任务 {task.task_id} 类型错误：期望 'preprocess'，但得到 '{task.task_type}'。任务未提交。"
                logger.warning(warning_msg)
                return False, warning_msg

            logger.info(f"接收到新的预处理任务，ID: {task.task_id}。正在提交至预处理队列...")

            # 1. 将任务对象放入专用的预处理请求队列
            self.preprocess_request_queue.put(task)

            # 2. 在全局任务状态字典中，初始化该任务的状态
            # asdict 是一个方便的函数，可以将 dataclass 对象转换为字典
            self.task_status_dict[task.task_id] = asdict(task)

            logger.info(f"任务 {task.task_id} 已成功提交。")
            return True, f"预处理任务 {task.task_id} 已成功提交。"

        except Exception as e:
            # 捕获可能在放入队列或字典时发生的罕见异常
            error_message = f"提交预处理任务 {getattr(task, 'task_id', 'Unknown')} 时发生严重错误: {e}"
            logger.error(error_message, exc_info=True)
            return False, str(e)

    def submit_synthesis_task(self, task: Task):
        """
        【公开方法】向“视频合成”生产线提交一个新任务。

        这个方法同样是线程安全和进程安全的。
        它将任务对象放入合成队列，并初始化任务的实时状态。

        Args:
            task (Task): 一个已经实例化的 Task 对象，其 task_type 应为 'synthesis'。

        Returns:
            tuple[bool, str]: 一个元组，第一个元素表示提交是否成功，第二个是相关信息。
        """
        try:
            # 确保提交的任务类型是正确的
            if task.task_type != 'synthesis':
                warning_msg = f"任务 {task.task_id} 类型错误：期望 'synthesis'，但得到 '{task.task_type}'。任务未提交。"
                logger.warning(warning_msg)
                return False, warning_msg

            logger.info(f"接收到新的合成任务，ID: {task.task_id}。正在提交至合成准备队列...")

            # 1. 将任务对象放入专用的合成请求队列
            self.synthesis_request_queue.put(task)

            # 2. 在全局任务状态字典中，初始化该任务的状态
            self.task_status_dict[task.task_id] = asdict(task)

            logger.info(f"任务 {task.task_id} 已成功提交。")
            return True, f"合成任务 {task.task_id} 已成功提交。"

        except Exception as e:
            # 捕获可能在放入队列或字典时发生的罕见异常
            error_message = f"提交合成任务 {getattr(task, 'task_id', 'Unknown')} 时发生严重错误: {e}"
            logger.error(error_message, exc_info=True)
            return False, str(e)


    #启动整个多进程服务框架
    def start_service(self, num_preprocess_workers=1, num_synthesis_prep_workers=2, num_gpus=1):
        """
        [最终架构] 启动整个多进程服务框架。

        此方法负责创建并启动所有类型的工作进程，包括：
        - 辅助工具进程 (WH计算器, 帧分发器, 视频写入器)
        - 视频预处理工作池 (CPU密集型)
        - 合成准备工作池 (CPU密集型)
        - GPU工作池 (GPU密集型)

        Args:
            num_preprocess_workers (int): 要启动的“视频预处理”工作进程的数量。
            num_synthesis_prep_workers (int): 要启动的“合成准备”工作进程的数量。
            num_gpus (int): 要启动的GPU工作进程的数量。
        """
        logger.info("=============================================")
        logger.info("      正在启动 TransDhTask 服务框架      ")
        logger.info("=============================================")

        # --- 1. 启动所有辅助工具进程 (Utility Workers) ---
        # 这些进程通常是单例，作为流水线中的关键节点。
        logger.info("正在启动辅助工具进程 (WH Calculator, Dispatcher, Writer)...")

        # 动态分配GPU给wh_calc_proc
        if num_gpus > 0:
            wh_gpu_id = num_gpus - 1
            logger.info(f"  - 分配 WH-Calculator 到 cuda:{wh_gpu_id}")
            wh_calc_proc = Process(target=init_wh_process,
                                   args=(self.init_wh_queue,
                                         self.init_wh_queue_output,
                                         wh_gpu_id),
                                   daemon=True)
            self.utility_workers.append(wh_calc_proc)

        dispatcher_proc = Process(target=frame_dispatcher_loop,
                                  args=(self.ready_queue, self.chunk_queue, self.task_status_dict, self.batch_size),
                                  daemon=True)
        self.utility_workers.append(dispatcher_proc)

        writer_proc = Process(target=video_writer_manager,
                              args=(self.writer_command_queue, self.result_queue, self.task_status_dict), daemon=True)
        self.utility_workers.append(writer_proc)

        for p in self.utility_workers:
            p.start()
        logger.info("所有辅助工具进程已成功启动。")

        # --- 2.【关键改造】启动视频预处理工作池 (Preprocess Worker Pool) ---
        logger.info(f"正在启动 {num_preprocess_workers} 个视频预处理工作进程...")
        for i in range(num_preprocess_workers):
            assigned_gpu_id = i % num_gpus
            logger.info(f"  - 分配 Preprocess-Worker-{i} 到 cuda:{assigned_gpu_id}")

            p = Process(
                target=preprocess_worker_loop,
                args=(self.preprocess_request_queue,
                      self.task_status_dict,
                      self.preprocess_results_dict,
                      self.init_wh_queue,
                      self.init_wh_queue_output,
                      assigned_gpu_id),
                daemon=False  # 非守护进程，确保在关闭时能处理完任务
            )
            p.start()
            self.preprocess_workers.append(p)
        logger.info(f"{len(self.preprocess_workers)} 个视频预处理工作进程已成功启动。")

        # --- 3.【关键改造】启动合成准备工作池 (Synthesis Prep Worker Pool) ---
        logger.info(f"正在启动 {num_synthesis_prep_workers} 个合成准备工作进程...")
        for _ in range(num_synthesis_prep_workers):
            p = Process(
                target=synthesis_prep_worker_loop,
                args=(self.synthesis_request_queue, self.ready_queue, self.task_status_dict),
                daemon=False  # 非守护进程
            )
            p.start()
            self.synthesis_prep_workers.append(p)
        logger.info(f"{len(self.synthesis_prep_workers)} 个合成准备工作进程已成功启动。")

        # --- 4. 启动GPU工作池 (GPU Worker Pool) ---
        # 这部分逻辑保持不变
        if num_gpus == 0:
            logger.warning("警告：启动了 0 个GPU工作进程。将无法执行任何视频合成。")

        logger.info(f"正在启动 {num_gpus} 个GPU工作进程...")
        for gpu_id in range(num_gpus):
            p = Process(
                target=gpu_worker_loop,
                args=(gpu_id,
                      self.chunk_queue,
                      self.writer_command_queue,
                      self.task_status_dict,
                      self.batch_size),
                daemon=True
            )
            p.start()
            self.gpu_workers.append(p)
        logger.info(f"{len(self.gpu_workers)} 个GPU工作进程已成功启动。")

        logger.info("所有工作进程均已启动。服务现在处于运行状态，等待任务提交。")
        logger.info("=============================================")

    #结束整个多进程服务框架
    def stop_service(self):
        """
        [最终架构] 优雅地关闭所有工作进程。

        通过向入口队列发送“毒丸”信号来逐级关闭整个服务，确保任务能被处理完毕。
        """
        logger.info("正在停止服务... 开始向所有工作队列发送“毒丸”信号。")

        # 1. 向两个CPU工作池的入口队列发送“毒丸”
        # 每个worker在完成当前任务后，会get到这个None，然后退出自己的while循环
        logger.info(f"正在通知 {len(self.preprocess_workers)} 个预处理工作进程关闭...")
        for _ in self.preprocess_workers:
            self.preprocess_request_queue.put(None)

        logger.info(f"正在通知 {len(self.synthesis_prep_workers)} 个合成准备工作进程关闭...")
        for _ in self.synthesis_prep_workers:
            self.synthesis_request_queue.put(None)

        # 2. 等待所有非守护进程（CPU workers）自然结束
        all_cpu_workers = self.preprocess_workers + self.synthesis_prep_workers
        logger.info("正在等待所有CPU工作进程完成当前任务并退出...")
        for p in all_cpu_workers:
            p.join(timeout=60)  # 等待最多60秒，让它完成当前任务
            if p.is_alive():
                logger.warning(f"CPU工作进程 {p.pid} 未能优雅退出，将强制终止。")
                p.terminate()

        # 3. 逐级关闭下游的守护进程
        # 当synthesis_prep_workers关闭后，ready_queue将不再有新数据，可以安全地关闭dispatcher
        self.ready_queue.put(None)

        # 等待所有守护进程结束
        all_daemon_workers = self.gpu_workers + self.utility_workers
        logger.info("正在等待所有守护进程（GPU, Dispatcher等）关闭...")
        for p in all_daemon_workers:
            p.join(timeout=10)  # 守护进程应该很快退出
            if p.is_alive():
                logger.warning(f"守护进程 {p.pid} 未能优雅退出，将强制终止。")
                p.terminate()

        logger.info("所有工作进程均已停止。服务已成功关闭。")

    #向服务提交一个新任务
    def submit_task(self, task: Task):
        """
        [阶段二改造] 向服务提交一个新任务。

        这个方法非常快，它只是将任务对象放入请求队列，
        然后立即返回，不会等待任务完成。
        """
        try:
            logger.info(f"Submitting new task: {task.task_id}")
            # 将任务对象放入请求队列的入口
            self.request_queue.put(task)
            # 初始化任务状态
            self.task_status_dict[task.task_id] = asdict(task)
            return True, f"Task {task.task_id} submitted successfully."
        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {e}", exc_info=True)
            return False, str(e)

    #查询一个任务的当前状态
    def get_task_status(self, task_id: str):
        """
        [新增] 查询一个任务的当前状态。
        """
        return self.task_status_dict.get(task_id, None)



    def preprocess(self, audio_url, video_url, code):
        """
        预处理步骤：下载并验证音视频文件。
        """
        s_pre = time.time()
        # 处理音频
        try:
            if audio_url.startswith('http:') or audio_url.startswith('https:'):
                _tmp_audio_path = os.path.join(GlobalConfig.instance().temp_dir, f'{code}.wav')
                download_file(audio_url, _tmp_audio_path)
            else:
                _tmp_audio_path = audio_url
        except Exception as e:
            traceback.print_exc()
            raise CustomError(f'[{code}]音频下载失败，异常信息:[{e.__str__()}]')

        # 处理视频
        try:
            if video_url.startswith('http:') or video_url.startswith('https:'):
                _tmp_video_path = os.path.join(GlobalConfig.instance().temp_dir, f'{code}.mp4')
                download_file(video_url, _tmp_video_path)
            else:
                _tmp_video_path = video_url
        except Exception as e:
            traceback.print_exc()
            raise CustomError(f'[{code}]视频下载失败，异常信息:[{e.__str__()}]')

        print(f'--------------------> download cost:{time.time() - s_pre}')
        return _tmp_audio_path, _tmp_video_path


    def change_task_status(self, code, status, progress, result, msg=''):
        """
        【最终修正版】线程安全地更新任务状态。
        修正了旧逻辑中只有在key已存在时才更新的问题。
        """
        try:
            with self.run_lock:
                # 无论key是否存在，都直接进行赋值更新
                self.task_dic[code] = (status, progress, result, msg)
        except Exception as e:
            traceback.print_exc()
            logger.error(f'[{code}]修改任务状态异常，异常信息:[{e.__str__()}]')


if __name__ == '__main__':
    """
    [最终架构] 主程序入口，用于启动和测试服务框架。
    """
    print("Initializing TransDhTask Service Framework...")
    # 直接调用构造函数创建唯一的服务管理器实例
    service_manager = TransDhTask()

    # 启动服务
    service_manager.start_service(
        num_cpu_workers=GlobalConfig.instance().num_cpu_workers,
        num_gpus=GlobalConfig.instance().num_gpus,
    )

    # --- 模拟提交任务 ---
    print("\n--- Simulating task submission ---")

    test_audio_path = 'data/test/test.mp3'
    test_video_path = 'data/test/test.mp4'

    tasks_to_submit = [
        Task(task_id="task_001", audio_url=test_audio_path, video_url=test_video_path, pn=1),
        Task(task_id="task_002", audio_url=test_audio_path, video_url=test_video_path, pn=0)
    ]

    for task in tasks_to_submit:
        service_manager.submit_task(task)

    submitted_task_ids = [t.task_id for t in tasks_to_submit]
    completed_tasks = 0

    # --- 监控任务状态和最终结果 ---
    print("\n--- Monitoring task status ---")
    while completed_tasks < len(submitted_task_ids):
        # 1. 打印实时状态
        status_report = []
        for task_id in submitted_task_ids:
            status = service_manager.get_task_status(task_id)
            if status:
                status_report.append(
                    f"  Task {task_id}: {status.get('status', 'loading')} ({status.get('progress', 0)}%)")
            else:
                status_report.append(f"  Task {task_id}: Not found yet.")

        print("\r" + " | ".join(status_report), end="", flush=True)

        # 2. 检查是否有任务完成（从result_queue获取）
        try:
            # 使用非阻塞的 get_nowait，避免卡住
            result = service_manager.result_queue.get_nowait()
            if result:
                completed_tasks += 1
                print(f"\n[Main] Task {result.get('task_id')} completed with status: {result.get('status')}")
        except Empty:
            # 队列为空，是正常情况，继续等待
            pass

        time.sleep(1)

    print("\n\nAll tasks have completed processing.")

    # --- 打印最终结果 ---
    print("\n--- Final Task Results ---")
    for task_id in submitted_task_ids:
        final_status = service_manager.get_task_status(task_id)
        if final_status and final_status.get('status') == 'success':
            print(f"  Task {task_id} Succeeded! Result path: {final_status.get('result_path')}")
        else:
            error_msg = final_status.get('error_message', 'Unknown error') if final_status else "Status not found"
            print(f"  Task {task_id} Failed! Reason: {error_msg}")

    # 优雅地关闭服务
    service_manager.stop_service()

    print("\nService has been shut down. Test finished.")