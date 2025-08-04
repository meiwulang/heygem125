# FILE: landmark2face_wy/digitalhuman_interface.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
import math  # 虽然被导入，但在还原的函数中未被使用

# 根据字节码，这些模块是 DigitalHumanModel 的核心依赖
from landmark2face_wy.options.test_options import TestOptions
from landmark2face_wy.models.l2faceaudio_model import L2FaceAudioModel
from face_lib.face_restore import GFPGAN
# y_utils.config 和 torchvision.transforms 在顶层被导入
from y_utils.config import GlobalConfig
import torchvision.transforms as transforms
from scipy.interpolate import splprep, splev
import os

def save_debug_image(work_id, frame_idx, step_name, image_data, base_dir='debug_output'):
    """
    根据全局开关，保存调试过程中的图像。
    能处理各种格式的数据（Tensor, Numpy, BGR, RGB, Float, Uint8, 单通道蒙版等）。
    """
    # 可以在这里加入一个全局开关来控制是否保存
    cfg = GlobalConfig.instance()
    if not cfg.enable_debug_save:
        return

    try:
        # --- 数据格式标准化 ---
        # 如果是PyTorch Tensor，先转移到CPU并转为Numpy
        if hasattr(image_data, 'cpu'):
            image_data = image_data.detach().cpu().numpy()

        # 如果是 C,H,W 格式 (e.g., 3, 256, 256)，转为 H,W,C
        if image_data.ndim == 3 and image_data.shape[0] in [1, 3]:
            image_data = np.transpose(image_data, (1, 2, 0))

        # 归一化到 [0, 255] 范围
        if image_data.max() <= 1.0:
            image_data = (image_data * 255)

        image_data = image_data.astype(np.uint8)

        # 颜色空间处理
        if image_data.ndim == 3 and image_data.shape[2] == 3:
            # 假设从GPU来的都是RGB，需要转为BGR以便cv2保存
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

        # --- 文件保存 ---
        # 创建目录
        debug_dir = os.path.join(base_dir, str(work_id), f'frame_{frame_idx:05d}')
        os.makedirs(debug_dir, exist_ok=True)

        # 保存文件
        file_path = os.path.join(debug_dir, f'{step_name}.png')
        cv2.imwrite(file_path, image_data)

    except Exception as e:
        print(f"警告: 无法保存调试图像 {work_id}/{frame_idx}/{step_name}. 错误: {e}")


class DigitalHumanModel:
    """
    数字人模型主类，负责加载模型、处理数据并执行推理。
    """

    """
    初始化数字人模型。

    Args:
        blend_dynamic (str): 动态融合模式，例如 'xseg'（使用分割蒙版）或 'lmk'（使用关键点蒙版）。
        chaofen_before (int): 是否在推理前使用GFPGAN进行人脸增强（超分）。1表示启用。
        face_blur_detect (bool): 是否启用人脸模糊检测。
    """

    def __init__(self, blend_dynamic, chaofen_before, face_blur_detect=False, gpu_id=0):

        init_start_time = time.time()
        print(f"【性能日志】DigitalHumanModel: 开始在 [cuda:{gpu_id}] 上初始化...")

        # 定义目标设备
        self.device = f'cuda:{gpu_id}'

        self.blend = True
        self.opt = TestOptions().parse()
        self.opt.isTrain = False

        model_load_start = time.time()
        # 从模型检查点加载配置 加载到CPU，避免抢占默认GPU
        temp_model = torch.load(self.opt.model_path, map_location='cpu', weights_only=False)
        print(f"【性能日志】DigitalHumanModel: 加载模型检查点完毕，耗时: {time.time() - model_load_start:.4f} 秒")

        self.opt.netG = temp_model["model_name"]
        self.opt.dataloader_size = temp_model["model_input_size"][0]
        self.opt.ngf = temp_model["model_ngf"]
        self.img_size = temp_model["model_input_size"][0]


        self.mask_re_cuda = torch.tensor(temp_model["input_mask_re"]).unsqueeze(0).unsqueeze(0).to(self.device)
        self.mask_cuda = torch.tensor(temp_model["input_mask"]).unsqueeze(0).unsqueeze(0).to(self.device)
        # self.fuse_mask_cuda = torch.tensor(self.fuse_mask).unsqueeze(0).unsqueeze(0).cuda().repeat(1, 3, 1, 1).half()
        self.nblend = temp_model["nblend"]

        core_model_start = time.time()

        # 初始化核心模型
        self.model = L2FaceAudioModel(self.opt)
        self.drivered_wh = temp_model["wh"]
        self.model.netG.load_state_dict(temp_model["face_G"])
        self.model.netG.to(self.device) # <-- 将模型移动到指定设备
        # self.model.netG.half()
        self.model.eval()
        print(f"【性能日志】DigitalHumanModel: 初始化核心唇语模型并移至 {self.device} 完毕，耗时: {time.time() - core_model_start:.4f} 秒")

        # 根据参数，条件性地初始化子模型
        # if blend_dynamic == 'xseg':
        #     from xseg.dfl_xseg_api import XsegNet
        #     self.xseg = XsegNet(model_name='xseg_net_private', provider='gpu')

        if chaofen_before == 1:
            gfpgan_load_start = time.time()
            self.gfpgan = GFPGAN(model_type="GFPGANv1.4", provider=self.device)
            print(f"【性能日志】DigitalHumanModel: 加载 GFPGAN 超分模型至 {self.device} 完毕，耗时: {time.time() - gfpgan_load_start:.4f} 秒")

        self.face_blur_detect = face_blur_detect
        if self.face_blur_detect:
            face_attr_load_start = time.time()
            from face_attr_detect.face_attr import FaceAttr
            self.face_attr = FaceAttr(model_name="face_attr_mbnv3", provider=self.device)
            print(f"【性能日志】DigitalHumanModel: 加载 FaceAttr 模糊检测模型至 {self.device} 完毕，耗时: {time.time() - face_attr_load_start:.4f} 秒")
        print(f"【性能日志】DigitalHumanModel: 全部初始化完成，总耗时: {time.time() - init_start_time:.4f} 秒")

    def tensor_norm(self, img_tensor, mask=None):
        """将范围在 [0, 255] 的图像张量归一化到 [-1, 1] 范围。"""
        img_tensor = img_tensor / 127.5 - 1.0
        if mask is not None:
            return (img_tensor + 1.0) * mask - 1.0
        return img_tensor

    def tensor_norm_no_training(self, img_tensor, mask=None):
        """将范围在 [0, 255] 的图像张量归一化到 [0, 1] 范围。"""
        img_tensor = img_tensor / 255.0
        if mask is not None:
            return img_tensor * mask
        return img_tensor


    @staticmethod
    def create_spline_mask(landmarks,
                           img_shape,
                           profile='full_face',
                           feather_kernel_size=31,
                           dilation_size=10,
                           neck_extension_ratio=0.2,
                           horizontal_erosion_ratio=0.0):
        """
        【新增核心函数，作为静态方法 - 最终增强版】
        新增 'lower_face_no_nose' 配置，实现更精细的下半脸控制。
        """
        ### --- 健壮性检查 --- ###
        if landmarks is None or len(landmarks) < 3:
            return np.zeros(img_shape[:2], dtype=np.float32)

        mask = np.zeros(img_shape[:2], dtype=np.uint8)

        ### --- 根据配置选择不同的蒙版生成策略 --- ###
        if profile == 'mouth_and_chin':
            # ---【终极配置: 仅口部与下巴区】---
            # 策略：合并嘴部和下巴点集，计算总凸包，实现最紧凑的包裹。

            mouth_points_indices = np.arange(48, 68)
            chin_points_indices = np.arange(4, 13)  # 选择下巴和下颌角部分

            combined_indices = np.concatenate([mouth_points_indices, chin_points_indices])
            combined_points = landmarks[combined_indices].astype(np.int32)

            # 对合并后的点集计算凸包，形成最精简的蒙版
            hull = cv2.convexHull(combined_points)
            cv2.fillConvexPoly(mask, hull, 255)

        if profile == 'lower_face_no_nose':
            # ---【全新配置: 下半脸无鼻版】---
            # 策略：选择从鼻翼两侧开始，向下包含整个脸颊和下巴的点，然后挖掉鼻子。

            # 1. 定义下半脸轮廓点。
            #    我们选取左右脸颊轮廓点(2-14)，并连接上嘴唇上方的鼻翼两侧点(31, 35)和嘴角上方的点(48, 54)
            #    来构成一个向上的封闭区域。
            lower_face_points_indices = np.concatenate([
                np.arange(2, 15),  # 右脸颊到左脸颊的下半部分
                [54, 48, 31, 35]  # 闭合嘴部上方的区域
            ])
            lower_face_points = landmarks[lower_face_points_indices].astype(np.int32)

            # 使用凸包来创建一个饱满的下半脸区域
            lower_face_hull = cv2.convexHull(lower_face_points)
            cv2.fillConvexPoly(mask, lower_face_hull, 255)

            # 2. 创建一个需要被排除的“鼻子”蒙版 (仅鼻头部分)
            nose_mask = np.zeros(img_shape[:2], dtype=np.uint8)
            nose_tip_points_indices = np.arange(31, 36)  # 仅选择鼻头和鼻翼
            nose_tip_points = landmarks[nose_tip_points_indices].astype(np.int32)
            nose_tip_hull = cv2.convexHull(nose_tip_points)

            # 对鼻头区域稍微做一些扩张
            cv2.fillConvexPoly(nose_mask, nose_tip_hull, 255)
            dilate_kernel_nose = np.ones((10, 10), np.uint8)  # 用较小的核扩张
            nose_mask = cv2.dilate(nose_mask, dilate_kernel_nose, iterations=1)

            # 3. 从主蒙版中减去（挖掉）鼻子蒙版
            mask = cv2.subtract(mask, nose_mask)

        elif profile in ('full_face', 'mouth_area', 'cheeks_chin_no_nose', 'face_with_neck'):
            # --- 将其他所有逻辑合并到一个分支中 ---
            if profile == 'full_face':
                # ... (原有 full_face 逻辑不变) ...
                points_indices = np.concatenate([np.arange(0, 17), np.arange(26, 16, -1)])
                points = landmarks[points_indices]
                if len(points) < 4:
                    cv2.fillPoly(mask, [points.astype(np.int32)], 255)
                else:
                    tck, u = splprep([points[:, 0], points[:, 1]], s=2.0, per=True)
                    u_new = np.linspace(u.min(), u.max(), 1000);
                    x_new, y_new = splev(u_new, tck, der=0)
                    cv2.fillPoly(mask, [np.c_[x_new, y_new].astype(np.int32)], 255)
            elif profile == 'mouth_area':
                # ... (原有 mouth_area 逻辑不变) ...
                points_indices = np.arange(48, 68)
                points = landmarks[points_indices]
                points_for_hull = points.astype(np.int32)
                points = cv2.convexHull(points_for_hull).squeeze()
                if len(points) < 4:
                    cv2.fillPoly(mask, [points.astype(np.int32)], 255)
                else:
                    tck, u = splprep([points[:, 0], points[:, 1]], s=2.0, per=True)
                    u_new = np.linspace(u.min(), u.max(), 1000);
                    x_new, y_new = splev(u_new, tck, der=0)
                    cv2.fillPoly(mask, [np.c_[x_new, y_new].astype(np.int32)], 255)
            elif profile == 'cheeks_chin_no_nose':
                # ... (原有 cheeks_chin_no_nose 逻辑不变) ...
                outer_points_indices = np.concatenate([np.arange(0, 17), np.arange(48, 60)])
                outer_points = landmarks[outer_points_indices].astype(np.int32)
                cv2.fillConvexPoly(mask, cv2.convexHull(outer_points), 255)
                nose_mask = np.zeros(img_shape[:2], dtype=np.uint8)
                nose_points_indices = np.arange(27, 36)
                nose_points = landmarks[nose_points_indices].astype(np.int32)
                cv2.fillConvexPoly(nose_mask, cv2.convexHull(nose_points), 255)
                nose_mask = cv2.dilate(nose_mask, np.ones((15, 15), np.uint8), iterations=1)
                mask = cv2.subtract(mask, nose_mask)
            elif profile == 'face_with_neck':
                base_face_mask_float = DigitalHumanModel.create_spline_mask(landmarks, img_shape,
                                                                            profile='cheeks_chin_no_nose',
                                                                            feather_kernel_size=0, dilation_size=0)
                mask = (base_face_mask_float * 255).astype(np.uint8)
                chin_points_indices = np.arange(4, 13)
                chin_points = landmarks[chin_points_indices]
                face_height = np.linalg.norm(landmarks[8] - landmarks[27])
                neck_extension_pixels = face_height * neck_extension_ratio
                neck_bottom_points = chin_points.copy();
                neck_bottom_points[:, 1] += neck_extension_pixels
                neck_area_points = np.concatenate([landmarks[[4, 12]], chin_points, neck_bottom_points]).astype(
                    np.int32)
                neck_mask = np.zeros(img_shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(neck_mask, cv2.convexHull(neck_area_points), 255)
                mask = cv2.bitwise_or(mask, neck_mask)

        ### --- 统一的后处理：扩张与羽化 --- ###
        # 1. (可选) 腐蚀操作，向内收缩蒙版
        if horizontal_erosion_ratio > 0:
            face_width_ref = np.linalg.norm(landmarks[16] - landmarks[0])
            erosion_pixels = int(face_width_ref * horizontal_erosion_ratio)
            if erosion_pixels > 0:
                # 只在水平方向腐蚀
                erosion_kernel = np.ones((1, erosion_pixels), np.uint8)
                mask = cv2.erode(mask, erosion_kernel, iterations=1)

        # 2. (可选) 扩张操作，向外扩展蒙版
        if dilation_size > 0:
            kernel = np.ones((dilation_size, dilation_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        # 3. (可选, 最后一步) 羽化操作，模糊边缘
        if feather_kernel_size > 0:
            if feather_kernel_size % 2 == 0: feather_kernel_size += 1
            mask = cv2.GaussianBlur(mask, (feather_kernel_size, feather_kernel_size), 0)

        final_mask = mask.astype(np.float32) / 255.0
        return final_mask

    """
        dilation_size (扩张尺寸)
        作用: 控制蒙版向外扩张的范围。可以理解为“保护区”的大小。
        取值原则:
        值越大: 融合区域越广。
        优点: 能更好地覆盖模型可能产生的微小抖动或面部动作带来的边缘变化，避免生成内容“露馅”。
        缺点: 会覆盖掉更多原始视频的高清像素（比如高质量的皮肤纹理），如果模型生成质量低于原始视频，可能会降低整体观感。
        值越小: 融合区域越精确、越紧凑。
        优点: 最大限度地保留原始视频的高清内容，只替换最核心的、必须改变的区域。
        缺点: 如果面部动作较大，可能会出现生成区域边缘与原始图像衔接不上的情况（比如嘴巴张得很大，但蒙版没跟上）。
        建议:
        blend_mask (脸内部融合): 可以设置得大一些 (如 20 到 30)。因为唇语模型主要影响嘴部，我们希望嘴部周围的皮肤也能平滑过渡，所以扩张大一点是有益的。
        full_spline_mask (脸与背景融合): 建议设置得小一些 (如 5 到 15)。因为脸和背景的边界（如发际线、下颌线）是我们希望严格保留的，过大的扩张会“吃掉”头发或衣领，造成不自然的模糊。
        feather_kernel_size (羽化核尺寸)
        作用: 控制蒙版边缘的模糊程度，即过渡带的宽度。
        取值原则:
        值越大: 边缘过渡带越宽，融合效果越“柔和”、“朦胧”。
        优点: 能极好地隐藏生成内容与原始图像之间的像素差异，让边界完全消失在平滑的过渡中。
        缺点: 如果过大，可能会让脸的边缘看起来过于模糊，失去锐利感。
        值越小: 边缘过渡带越窄，融合效果越“锐利”、“清晰”。
        优点: 能保持脸部轮廓的清晰度。
        缺点: 如果生成内容和原始图像在色调、亮度上有差异，窄边过渡可能会让边界显得有些突兀。
        建议:
        通常情况下，feather_kernel_size 应该与 dilation_size 成正比关系。一个大的扩张区域需要一个大的羽化核来平滑过渡。
        51 是一个比较大的值，能产生非常柔和的效果。对于大多数情况，这是一个安全且效果不错的选择。
        您可以尝试将它减小到 31 或 21，观察脸部边缘是否变得更加清晰，同时检查融合边界是否依然自然。
        """


    #26-27fps
    def inference_notraining_gpu_composite(self, audio_info, face_data_dict, original_frames_bgr, face_coords,
                                           spline_masks_cpu, params):
        """
        核心生成流程 (最终生产版: A-B-C三阶段重构, 含零开销调试)。
        """
        work_id, start_frame_idx = params[0], params[1]
        this_batch = len(audio_info)

        if this_batch == 0:
            return []

        # --- 在函数开始时获取一次全局调试开关，避免在循环中重复查询 ---
        is_debug_enabled = GlobalConfig.instance().enable_debug_save

        # ================================================================================= #
        #   A 阶段: CPU 的所有准备 (CPU-Side Preparation)
        # ================================================================================= #

        source_face_list_cpu = []
        lab_list_cpu = []
        for i in range(this_batch):
            std_crop_img_bgr = face_data_dict[i]["crop_img"]
            if is_debug_enabled:
                save_debug_image(work_id, start_frame_idx + i, '01_model_input_std_256', std_crop_img_bgr)

            source_face_rgb_chw = std_crop_img_bgr[:, :, ::-1].transpose(2, 0, 1).copy()
            source_face_list_cpu.append(source_face_rgb_chw)
            lab_list_cpu.append(audio_info[i].transpose(1, 0))

        source_face_batch_cpu_numpy = np.array(source_face_list_cpu)
        lab_batch_cpu_numpy = np.array(lab_list_cpu)

        # ================================================================================= #
        #   B 阶段: GPU 上的所有计算 (GPU-Side Computation)
        # ================================================================================= #

        # B1. 数据批量上传
        source_face_gpu = torch.from_numpy(source_face_batch_cpu_numpy).to(self.device)
        lab_gpu = torch.from_numpy(lab_batch_cpu_numpy).to(self.device)
        original_frames_gpu = torch.from_numpy(
            np.array(original_frames_bgr)[:, :, :, ::-1].transpose(0, 3, 1, 2).copy()
        ).to(self.device).float() / 255.0
        spline_masks_gpu = torch.from_numpy(np.array(spline_masks_cpu)).unsqueeze(1).to(self.device).float()

        # B2. GPU内部预处理: 派生模型输入
        mask_B_tensor = self.tensor_norm_no_training(source_face_gpu.clone(),
                                                     mask=self.mask_cuda.repeat(this_batch, 3, 1, 1))
        B_img__tensor = self.tensor_norm_no_training(source_face_gpu.clone(),
                                                     mask=self.mask_re_cuda.repeat(this_batch, 3, 1, 1))

        # B3. 核心模型推理 (在AMP上下文中)
        with torch.cuda.amp.autocast():
            fake_B_gpu = self.model.netG(mask_B_tensor, B_img__tensor, lab_gpu)

            # ================================================================= #
            #   新增：在第100个批次时，保存“黄金”输入和输出数据进行验证
            # ================================================================= #
            # 假设 work_id 是批次的唯一标识符。
            # 注意：请根据您的 work_id 是从0还是1开始，来决定使用 99 还是 100。
            # 这里我们以第100个批次（ID为100）为例。
            TARGET_BATCH_ID = 100
            output_dir = f"./verification_data_batch_{TARGET_BATCH_ID}"

            if start_frame_idx == TARGET_BATCH_ID and not os.path.exists(output_dir):
                print(f"\n[验证] 检测到目标批次 ID: {start_frame_idx}。正在保存黄金数据以供验证...")

                os.makedirs(output_dir, exist_ok=True)

                # 将输入张量转为 numpy 并保存
                np.save(os.path.join(output_dir, "input_mask_B.npy"), mask_B_tensor.cpu().detach().numpy())
                np.save(os.path.join(output_dir, "input_B_img.npy"), B_img__tensor.cpu().detach().numpy())
                np.save(os.path.join(output_dir, "input_lab.npy"), lab_gpu.cpu().detach().numpy())

                # 将 PyTorch 模型的输出转为 numpy 并保存
                np.save(os.path.join(output_dir, "output_pytorch.npy"), fake_B_gpu.cpu().detach().numpy())

                print(f"[验证] 黄金数据已成功保存至目录: {output_dir}\n")

            # ================================================================= #
            #   保存逻辑结束
            # ================================================================= #


            if self.nblend:
                B_img_tensor = self.tensor_norm_no_training(source_face_gpu)
                fake_B_gpu = torch.where(self.mask_re_cuda == 0, B_img_tensor, fake_B_gpu)

            # --- B4. GPU矢量化图像融合 ---
            if is_debug_enabled:
                for i in range(this_batch):
                    save_debug_image(work_id, start_frame_idx + i, '02_model_raw_output_fake_B_256', fake_B_gpu[i])

            foreground_canvas = torch.zeros_like(original_frames_gpu)
            paste_mask = torch.zeros_like(original_frames_gpu[:, 0:1, :, :])

            # --- 用于收集先决条件参数的列表 ---
            prerequisite_params_log = []

            for i in range(this_batch):
                y1, y2, x1, x2 = face_coords[i]
                h, w = original_frames_bgr[i].shape[:2]
                y1_s, y2_s, x1_s, x2_s = max(0, y1), min(h, y2), max(0, x1), min(w, x2)
                crop_h, crop_w = y2_s - y1_s, x2_s - x1_s

                # --- 收集当前帧的关键参数 ---
                log_entry = {
                    "frame_index_in_batch": i,
                    "target_coords": (y1_s, y2_s, x1_s, x2_s),
                    "target_size": (crop_h, crop_w)
                }
                prerequisite_params_log.append(log_entry)

                if crop_h <= 0 or crop_w <= 0: continue

                material_gpu = F.interpolate(fake_B_gpu[i].unsqueeze(0), size=(crop_h, crop_w), mode='bicubic',
                                             align_corners=False).squeeze(0)
                final_mask_gpu_resized = F.interpolate(spline_masks_gpu[i].unsqueeze(0), size=(crop_h, crop_w),
                                                       mode='bicubic', align_corners=False)

                final_mask_3ch = final_mask_gpu_resized.squeeze(0).repeat(3, 1, 1)
                patch = material_gpu * final_mask_3ch
                foreground_canvas[i, :, y1_s:y2_s, x1_s:x2_s] = patch
                paste_mask[i, :, y1_s:y2_s, x1_s:x2_s] = final_mask_gpu_resized

                if is_debug_enabled:
                    frame_abs_idx = start_frame_idx + i
                    save_debug_image(work_id, frame_abs_idx, '03_material_resized_AI_content', material_gpu.clamp(0, 1))
                    save_debug_image(work_id, frame_abs_idx, '04_mask_final_resized',
                                     final_mask_gpu_resized.squeeze().clamp(0, 1))
                    background_region_gpu_debug = original_frames_gpu[i, :, y1_s:y2_s, x1_s:x2_s]
                    fused_face_region_debug = patch + background_region_gpu_debug * (1 - final_mask_3ch)
                    save_debug_image(work_id, frame_abs_idx, '05_background_region_for_fusion',
                                     background_region_gpu_debug)
                    save_debug_image(work_id, frame_abs_idx, '06_fused_final_face_region', fused_face_region_debug.clamp(0, 1))

            # print("\n" + "=" * 80)
            # print(f"审查批次 {work_id} (帧 {start_frame_idx} onwards) 的循环矢量化先决条件:")
            # print("-" * 80)
            # print(f"{'帧索引 (批内)':<20} | {'目标坐标 (y1, y2, x1, x2)':<35} | {'目标尺寸 (h, w)':<20}")
            # print("-" * 80)
            for entry in prerequisite_params_log:
                coord_str = str(entry['target_coords'])
                size_str = str(entry['target_size'])
                # print(f"{entry['frame_index_in_batch']:<20} | {coord_str:<35} | {size_str:<20}")
            # print("=" * 80 + "\n")

            # B5. 最终并行合成
            final_frames_gpu = foreground_canvas + original_frames_gpu * (1 - paste_mask.repeat(1, 3, 1, 1))

            if is_debug_enabled:
                for i in range(this_batch):
                    y1, y2, x1, x2 = face_coords[i]
                    if (y2 - y1) > 0 and (x2 - x1) > 0:
                        save_debug_image(work_id, start_frame_idx + i, '07_final_frame_after_paste',
                                         final_frames_gpu[i].clamp(0, 1))

        # ================================================================================= #
        #   C 阶段: 从 GPU 下载 (Download from GPU)
        # ================================================================================= #

        # C1. 批量数据下载与格式转换
        final_frames_cpu_numpy_rgb = (
            final_frames_gpu.clamp(0, 1)
            .permute(0, 2, 3, 1)
            .mul(255)
            .byte()
            .cpu()
            .numpy()
        )

        # C2. 最终格式转换 (RGB -> BGR)
        final_frames_cpu_numpy_bgr = final_frames_cpu_numpy_rgb[:, :, :, ::-1]
        final_output_bgr = list(final_frames_cpu_numpy_bgr)

        # C3. 主动释放所有GPU张量
        del final_frames_gpu, source_face_gpu, mask_B_tensor, B_img__tensor, lab_gpu
        del original_frames_gpu, spline_masks_gpu, fake_B_gpu, foreground_canvas, paste_mask
        if 'B_img_tensor' in locals(): del B_img_tensor
        torch.cuda.empty_cache()

        return final_output_bgr
