# inspect_onnx_layers.py
# 一个简单的Python脚本，用于加载ONNX模型并打印所有层的名称和类型。

import onnx
import os

def inspect_onnx(onnx_path):
    """
    加载ONNX模型并打印其所有节点的详细信息。
    """
    print(f"--- 正在分析 ONNX 模型: {onnx_path} ---")

    try:
        model = onnx.load(onnx_path)
    except Exception as e:
        print(f"错误: 无法加载ONNX文件。 {e}")
        return

    print(f"\n模型中的所有节点 (层) 列表 (共 {len(model.graph.node)} 个):")
    print("-" * 70)

    # 遍历图中的所有节点（层）
    for i, node in enumerate(model.graph.node):
        print(f"  层 {i + 1}:")
        print(f"    - 名称 (Name) : {node.name}")
        print(f"    - 类型 (OpType) : {node.op_type}")
        print(f"    - 输入 (Inputs) : {node.input}")
        print(f"    - 输出 (Outputs): {node.output}")
        print("-" * 70)

    print(f"\n--- 分析完成 ---")
    print("请从此列表中找到您怀疑数值不稳定的层（通常在 'AdaAT' 模块内部），")
    print("并将其 '名称 (Name)' 复制到 build_engine_fp16.py 脚本的命令行参数中。")


if __name__ == '__main__':
    # 我们的目标是分析FP32的ONNX文件
    onnx_file = "dinet_v1_netG_fp32.onnx"

    if not os.path.exists(onnx_file):
        print(f"错误: 找不到ONNX文件 '{onnx_file}'。请确保它在当前目录。")
    else:
        inspect_onnx(onnx_file)