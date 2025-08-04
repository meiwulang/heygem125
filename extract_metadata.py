import torch
import json
import os
import numpy as np

# ========================== 配置 ==========================
# 请将此路径修改为您项目中 dinet_v1_20240131.pth 文件的实际绝对或相对路径
PTH_FILE_PATH = "./landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth"


# ==========================================================

def extract_and_save_metadata(pth_path: str):
    """
    加载一个PyTorch的.pth检查点文件，提取其中预定义的元数据，
    并将其保存到一个同名的.json文件中。

    此脚本旨在将模型权重与模型配置/元数据分离，以便后续使用
    TensorRT引擎时，仍能加载这些关键的元数据。
    """
    print(f"--- 开始处理元数据提取任务 ---")
    print(f"输入文件: {pth_path}")

    # 1. 检查输入文件是否存在
    if not os.path.exists(pth_path):
        print(f"\n[错误] 输入文件不存在: {pth_path}")
        print("请检查 PTH_FILE_PATH 变量是否设置正确。")
        return

    # 2. 确定输出路径
    output_json_path = os.path.splitext(pth_path)[0] + ".json"
    print(f"输出文件: {output_json_path}")

    # 3. 定义需要提取的元数据键
    keys_to_extract = [
        "wh",  # 驱动视频人脸的平均宽高比 (float)
        "nblend",  # 是否启用nblend (bool or int)
        "input_mask",  # 输入蒙版 (Tensor or Numpy Array)
        "input_mask_re",  # 另一个输入蒙版 (Tensor or Numpy Array)
        "model_name",  # 模型名称 (str)
        "model_input_size",  # 模型输入尺寸 (list or tuple)
        "model_ngf",  # 模型的ngf参数 (int)
    ]
    print(f"计划提取的键: {keys_to_extract}")

    metadata_to_save = {}

    try:
        # 4. 加载.pth文件到CPU，避免占用GPU
        print("\n正在加载 .pth 文件 (可能需要一些时间)...")

        # 【关键修改】显式地将 weights_only 设置为 False，以允许加载非张量的Python对象（如Numpy数组）
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)

        print("文件加载成功。")

        # 5. 遍历并提取元数据
        print("正在提取并转换元数据...")
        for key in keys_to_extract:
            if key not in checkpoint:
                print(f"  [警告] 在.pth文件中未找到键: '{key}'，将跳过此键。")
                continue

            value = checkpoint[key]

            # 关键步骤：将Tensor或Numpy数组转换为JSON兼容的列表格式
            if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
                print(f"  - 正在转换 '{key}' (类型: {type(value).__name__}) 为列表...")
                metadata_to_save[key] = value.tolist()
            # 其他类型（如int, float, str, list）可以直接保存
            else:
                print(f"  - 正在提取 '{key}' (类型: {type(value).__name__})...")
                metadata_to_save[key] = value

        # 6. 保存到JSON文件
        print("\n正在将提取的元数据写入JSON文件...")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            # indent=4 使JSON文件格式化，易于阅读和检查
            json.dump(metadata_to_save, f, indent=4)

        print("\n--- 任务成功完成! ---")
        print(f"元数据已成功保存到: {output_json_path}")

    except Exception as e:
        print(f"\n[致命错误] 在处理过程中发生异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    extract_and_save_metadata(PTH_FILE_PATH)