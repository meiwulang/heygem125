# final_showdown.py
# 最终对决脚本：全面对比 PyTorch, ONNX Runtime, 和 TensorRT (FP32) 的性能与精度

import torch
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import os
import argparse
import time
from skimage.metrics import structural_similarity as ssim
import cv2

# --- PyTorch 模型加载与推理 ---
from landmark2face_wy.models.l2faceaudio_model import L2FaceAudioModel
from landmark2face_wy.options.test_options import TestOptions
import numpy.core.multiarray
import torch.serialization


# --- 辅助函数 ---

def load_pytorch_model(checkpoint_path, device):
    print("正在加载 PyTorch 模型...")
    reconstruct = numpy.core.multiarray._reconstruct
    with torch.serialization.safe_globals([reconstruct]):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    opt = TestOptions().parse()
    opt.isTrain = False
    opt.gpu_ids = [device.index]
    opt.netG = checkpoint["model_name"]
    opt.ngf = checkpoint["model_ngf"]
    opt.dataloader_size = checkpoint["model_input_size"][0]

    model_wrapper = L2FaceAudioModel(opt)
    netG = model_wrapper.netG
    netG.load_state_dict(checkpoint["face_G"])
    netG.to(device)
    netG.eval()
    print("PyTorch 模型加载成功。")
    return netG


def benchmark_pytorch(model, inputs_torch, warmup_runs=10, benchmark_runs=50):
    print("-" * 70)
    print("正在测试 PyTorch (FP32) 模型...")

    # 预热
    print(f"  - 正在进行 {warmup_runs} 次预热运行...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(*inputs_torch)
    torch.cuda.synchronize()

    # 测试
    print(f"  - 正在进行 {benchmark_runs} 次基准测试运行...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        for _ in range(benchmark_runs):
            output = model(*inputs_torch)
        end_event.record()

    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / benchmark_runs
    throughput_qps = (benchmark_runs * inputs_torch[0].shape[0]) / (total_time_ms / 1000.0)

    print("  - 测试完成。")
    return output.cpu().numpy(), avg_latency_ms, throughput_qps


def benchmark_onnx(onnx_path, inputs_np, warmup_runs=10, benchmark_runs=50):
    print("-" * 70)
    print("正在测试 ONNX Runtime (FP32) 模型...")

    ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    input_names = [inp.name for inp in ort_session.get_inputs()]
    ort_inputs = {name: data for name, data in zip(input_names, inputs_np)}

    # 预热
    print(f"  - 正在进行 {warmup_runs} 次预热运行...")
    for _ in range(warmup_runs):
        _ = ort_session.run(None, ort_inputs)

    # 测试
    print(f"  - 正在进行 {benchmark_runs} 次基准测试运行...")
    start_time = time.perf_counter()
    for _ in range(benchmark_runs):
        outputs = ort_session.run(None, ort_inputs)
    end_time = time.perf_counter()

    total_time_s = end_time - start_time
    avg_latency_ms = (total_time_s * 1000) / benchmark_runs
    throughput_qps = (benchmark_runs * inputs_np[0].shape[0]) / total_time_s

    print("  - 测试完成。")
    return outputs[0], avg_latency_ms, throughput_qps


def benchmark_trt(engine_path, inputs_np, warmup_runs=10, benchmark_runs=50):
    print("-" * 70)
    print("正在测试 TensorRT (FP32) 引擎...")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    buffers = {}
    inputs_torch = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        if is_input:
            input_data = inputs_np[i]
            context.set_input_shape(name, input_data.shape)
            buffers[name] = torch.from_numpy(input_data).cuda()
        else:
            output_shape = context.get_tensor_shape(name)
            buffers[name] = torch.empty(size=tuple(output_shape), dtype=torch.float32).cuda()
            output_name = name
    for name, buf in buffers.items():
        context.set_tensor_address(name, buf.data_ptr())

    # 预热
    print(f"  - 正在进行 {warmup_runs} 次预热运行...")
    for _ in range(warmup_runs):
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # 测试
    print(f"  - 正在进行 {benchmark_runs} 次基准测试运行...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(benchmark_runs):
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    end_event.record()

    torch.cuda.current_stream().synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / benchmark_runs
    throughput_qps = (benchmark_runs * inputs_np[0].shape[0]) / (total_time_ms / 1000.0)

    print("  - 测试完成。")
    return buffers[output_name].cpu().numpy(), avg_latency_ms, throughput_qps


def compare_outputs_final(output_base, output_compare, base_name, compare_name):
    img_base = output_base[0]
    img_compare = output_compare[0]

    mae = np.mean(np.abs(img_base - img_compare))

    img_base_uint8 = (np.clip(np.transpose(img_base, (1, 2, 0)), 0, 1) * 255).astype(np.uint8)
    img_compare_uint8 = (np.clip(np.transpose(img_compare, (1, 2, 0)), 0, 1) * 255).astype(np.uint8)

    ssim_score = ssim(img_base_uint8, img_compare_uint8, channel_axis=2, data_range=255)
    psnr_score = cv2.PSNR(img_base_uint8, img_compare_uint8)

    return {
        "title": f"--- 🔬 {compare_name} vs {base_name} 视觉质量对比 ---",
        "mae": mae,
        "ssim": ssim_score,
        "psnr": psnr_score
    }


# --- 主函数 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="终极对决：对比 PyTorch, ONNX Runtime, 和 TensorRT (FP32) 的性能与精度。")
    parser.add_argument("--ckpt", type=str, default="./landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth",
                        help="PyTorch模型检查点文件路径。")
    parser.add_argument("--onnx", type=str, default="dinet_v1_netG_fp32.onnx", help="FP32 ONNX文件路径。")
    parser.add_argument("--engine", type=str, default="dinet_v1_netG_fp32_fp32.engine",
                        help="FP32 TensorRT引擎文件路径。")
    parser.add_argument("--batch", type=int, default=8,
                        help="用于基准测试的批处理大小（建议使用一个较小的、所有框架都能运行的批次）。")
    parser.add_argument("--runs", type=int, default=50, help="基准测试运行次数。")

    args = parser.parse_args()

    # 检查依赖
    try:
        import skimage
    except ImportError:
        print("\n错误：需要 scikit-image 库。请运行: pip install scikit-image\n")
        exit(1)

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建通用的输入数据
    print("创建固定的FP32输入数据用于测试...")
    inputs_np = [
        np.ones((args.batch, 3, 256, 256), dtype=np.float32) * 0.5,
        np.ones((args.batch, 3, 256, 256), dtype=np.float32) * 0.5,
        np.sin(np.arange(args.batch * 256 * 16).reshape(args.batch, 256, 16)).astype(np.float32)
    ]
    inputs_torch = [torch.from_numpy(data).to(device) for data in inputs_np]

    # 运行所有基准测试
    out_pt, lat_pt, thr_pt = benchmark_pytorch(
        load_pytorch_model(args.ckpt, device), inputs_torch, benchmark_runs=args.runs
    )

    out_ort, lat_ort, thr_ort = benchmark_onnx(
        args.onnx, inputs_np, benchmark_runs=args.runs
    )

    out_trt, lat_trt, thr_trt = benchmark_trt(
        args.engine, inputs_np, benchmark_runs=args.runs
    )

    # 对比输出精度
    quality_ort_vs_pt = compare_outputs_final(out_pt, out_ort, "PyTorch", "ONNX RT")
    quality_trt_vs_pt = compare_outputs_final(out_pt, out_trt, "PyTorch", "TensorRT")

    # 打印最终报告
    print("\n" + "=" * 70)
    print("                三方对决最终报告")
    print("=" * 70)
    print(f"测试批次大小 (Batch Size): {args.batch}\n")

    print("--- 🚀 性能对比 (Performance) ---")
    print(f"  - PyTorch   | Latency: {lat_pt:8.3f} ms | Throughput: {thr_pt:8.2f} frames/sec")
    print(f"  - ONNX RT   | Latency: {lat_ort:8.3f} ms | Throughput: {thr_ort:8.2f} frames/sec")
    print(f"  - TensorRT  | Latency: {lat_trt:8.3f} ms | Throughput: {thr_trt:8.2f} frames/sec")
    print("-" * 50)
    speedup_trt_vs_pt = lat_pt / lat_trt
    speedup_trt_vs_ort = lat_ort / lat_trt
    print(f"  ✅ TensorRT vs PyTorch 加速比: {speedup_trt_vs_pt:.2f}x")
    print(f"  ✅ TensorRT vs ONNX RT 加速比: {speedup_trt_vs_ort:.2f}x\n")

    print(quality_ort_vs_pt["title"])
    print(f"  - 平均绝对误差 (MAE):   {quality_ort_vs_pt['mae']:.8f}")
    print(f"  - 结构相似性 (SSIM):  {quality_ort_vs_pt['ssim']:.6f}")
    print(f"  - 峰值信噪比 (PSNR):  {quality_ort_vs_pt['psnr']:.2f} dB\n")

    print(quality_trt_vs_pt["title"])
    print(f"  - 平均绝对误差 (MAE):   {quality_trt_vs_pt['mae']:.8f}")
    print(f"  - 结构相似性 (SSIM):  {quality_trt_vs_pt['ssim']:.6f}")
    print(f"  - 峰值信噪比 (PSNR):  {quality_trt_vs_pt['psnr']:.2f} dB\n")

    print("--- 💡 最终结论 ---")
    if speedup_trt_vs_ort > 1.2 and quality_trt_vs_pt['ssim'] > 0.99:
        print("  🎉 结论：TensorRT 带来了显著的性能提升，且几乎没有精度损失。")
        print("     整个优化流程非常成功，强烈建议在生产环境中使用 TensorRT 引擎。")
    else:
        print("  🤔 结论：TensorRT 带来的性能提升有限或存在一定的精度差异。")
        print("     请根据报告中的具体数据，判断是否值得在生产中使用。")
    print("=" * 70)