# 文件名: api.py
import multiprocessing
import uuid
import time
from contextlib import asynccontextmanager
import uvicorn
import wave
import numpy as np
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# --- 核心导入 ---
# 假设您的后端服务代码都放在一个名为 'service' 的目录/模块下
# 您需要根据您的项目结构调整这个导入路径
from service.trans_dh_service import TransDhTask, Task
from y_utils.config import GlobalConfig
from y_utils.logger import setup_logger, logger
from typing import AsyncGenerator

# --- Pydantic 模型定义 ---
# 定义API请求和响应的数据结构

class TrainRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="可选的任务ID，未填则自动生成")
    video_url: str = Field(..., description="需要进行预处理的视频URL")
    callbackUrl: Optional[str] = Field(None, description="任务完成后的回调URL（当前暂未使用，但保留）")

class StandardResponse(BaseModel):
    code: int = Field(..., description="状态码，200表示成功")
    msg: str = Field(..., description="描述信息")
    time: str = Field(..., description="服务器响应时间戳")
    task_id: str = Field(..., description="本次请求创建的任务ID")
    # 注意：在/train的响应中，我们不直接返回model_id，因为它是异步生成的
    # model_id: Optional[str] = Field(None, description="模型的唯一标识")

# 为 /get_train 的响应创建一个专门的模型，以包含 model_id
class TaskStatusResponse(BaseModel):
    code: int = Field(..., description="状态码。200:成功, 201:处理中, 400:任务不存在, 500:失败")
    title: str = Field(..., description="用户友好的状态标题，如 '训练完成', '训练中'")
    msg: str = Field(..., description="详细信息或错误消息")
    time: str = Field(..., description="服务器响应时间戳")
    task_id: str = Field(..., description="被查询的任务ID")
    model_id: Optional[str] = Field(None, description="【关键】当且仅当训练成功时，此字段会包含生成的模型ID")
    # 保留一些旧的字段以兼容，尽管它们在训练阶段可能为None
    oss_url: Optional[str] = Field(None)
    img_url: Optional[str] = Field(None)
    jg: Optional[str] = Field(None)
    seconds: Optional[int] = Field(None)
    md5: Optional[str] = Field(None)

# 为 /create_task 的请求体创建一个专门的模型
class GenerateRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="可选的任务ID，未填则自动生成")
    audio_url: str = Field(..., description="用于驱动口型的音频URL")
    model_id: str = Field(..., description="【关键】由 /get_train 接口返回的，已预处理好的模型ID")
    callbackUrl: Optional[str] = Field(None, description="任务完成后的回调URL（当前暂未使用，但保留）")
    # 可以将其他可选参数也加进来
    pn: int = Field(1, description="乒乓模式开关。0:关闭, 1:开启")
    chaofen: int = Field(0, description="超分开关。0:关闭, 1:开启")


# --- 全局服务管理器【声明】 ---
service_manager: Optional[TransDhTask] = None


# DUMMY_AUDIO_PATH 变量可以移除，因为我们现在从配置中获取

# --- FastAPI 应用生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    管理服务框架的启动和关闭。
    这是一个标准的 FastAPI 生命周期事件处理器。
    """
    # =============================================================
    #                     启动 (Startup) 逻辑
    # =============================================================
    global service_manager

    setup_logger()
    logger.info("FastAPI application starting up...")

    cfg = GlobalConfig.instance()

    # 处理占位静音音频文件
    logger.info(f"正在检查或创建占位静音音频文件: {cfg.dummy_audio_path}")
    try:
        os.makedirs(os.path.dirname(cfg.dummy_audio_path), exist_ok=True)
        if not os.path.exists(cfg.dummy_audio_path):
            sample_rate = 16000
            silent_samples = np.zeros(int(sample_rate * 0.01), dtype=np.int16)
            with wave.open(cfg.dummy_audio_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(silent_samples.tobytes())
            logger.info("占位静音音频文件创建成功。")
        else:
            logger.info("占位静音音频文件已存在。")
    except Exception as e:
        logger.error(f"创建占位静音音频文件失败: {e}", exc_info=True)
        raise e

    # # 诊断CUDA设备
    # num_cuda_devices = 0
    # if torch.cuda.is_available():
    #     num_cuda_devices = torch.cuda.device_count()
    #     logger.info(f"PyTorch can see {num_cuda_devices} CUDA device(s).")
    #     for i in range(num_cuda_devices):
    #         logger.info(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
    # else:
    #     logger.warning("PyTorch reports that CUDA is not available. GPU workers will not be started.")
    #
    # desired_num_gpus = cfg.num_gpus
    # num_gpus_to_start = 0
    # if desired_num_gpus > num_cuda_devices:
    #     logger.warning(
    #         f"Configured num_gpus ({desired_num_gpus}) is greater than "
    #         f"the number of visible devices ({num_cuda_devices}). "
    #         f"Adjusting to start only {num_cuda_devices} GPU worker(s)."
    #     )
    #     num_gpus_to_start = num_cuda_devices
    # else:
    #     num_gpus_to_start = desired_num_gpus

    # 实例化并启动服务框架
    service_manager = TransDhTask()
    service_manager.start_service(
        num_preprocess_workers=cfg.num_preprocess_workers,
        num_synthesis_prep_workers=cfg.num_synthesis_prep_workers,
        num_gpus=cfg.num_gpus
    )

    logger.info("--- Application startup complete. Service is now running. ---")

    # 使用 yield 将控制权交还给 FastAPI，应用开始接收请求
    yield
    # =============================================================
    #                     关闭 (Shutdown) 逻辑
    # =============================================================
    # 当应用关闭时，代码会从这里继续执行

    logger.info("--- Application shutting down. Starting cleanup... ---")

    # 清理占位文件
    logger.info(f"正在清理占位静音音频文件: {cfg.dummy_audio_path}")
    try:
        if os.path.exists(cfg.dummy_audio_path):
            os.remove(cfg.dummy_audio_path)
            logger.info("占位静音音频文件已清理。")
    except Exception as e:
        logger.warning(f"清理占位静音音频文件时出错: {e}")

    # 优雅地停止后端服务
    if service_manager:
        logger.info("Stopping backend service.")
        service_manager.stop_service()

    logger.info("--- Cleanup complete. Application has shut down. ---")

# --- FastAPI 应用实例 ---
# 使用 lifespan 来确保服务在应用启动时启动，在关闭时停止
app = FastAPI(lifespan=lifespan)


# --- API 端点定义 ---

@app.post("/train", response_model=StandardResponse, tags=["模型训练 (Training)"])
async def train(request: TrainRequest):
    """
    **发起一个视频预处理（训练）任务。**

    此接口接收一个视频URL，并将其提交到后端的“视频预处理”生产线。
    这是一个异步操作，接口会立即返回一个 `train_task_id`。
    您需要使用这个ID去轮询 `/get_train` 接口来获取最终的 `model_id`。
    """
    # 1. 为本次预处理任务生成一个唯一的ID
    train_task_id = request.task_id if request.task_id else str(uuid.uuid4())
    logger.info(f"收到 /train 请求，任务ID: {train_task_id}，视频URL: {request.video_url}")

    # 2. 创建一个明确的“预处理”类型任务对象
    # 注意：audio_url 在这里是一个无意义的占位符
    task_to_submit = Task(
        task_id=train_task_id,
        video_url=request.video_url,
        audio_url="dummy.wav",  # 占位符，预处理任务不使用
        task_type='preprocess' # 【关键】指定任务类型为 'preprocess'
    )

    # 3. 通过服务管理器，调用专用的方法提交任务
    success, message = service_manager.submit_preprocess_task(task_to_submit)

    # 4. 如果提交失败，向上层抛出HTTP异常
    if not success:
        logger.error(f"任务 {train_task_id} 提交失败: {message}")
        raise HTTPException(status_code=500, detail=f"任务提交至后端服务失败: {message}")

    # 5. 如果提交成功，立即返回标准响应
    # 客户端的责任是保存 train_task_id，并用它来轮询任务状态
    response_data = {
        "code": 200,
        "msg": "视频预处理任务已成功提交。",
        "time": str(int(time.time())),
        "task_id": train_task_id
    }
    logger.info(f"为任务 {train_task_id} 返回成功响应: {response_data}")
    return response_data

@app.get("/get_train", response_model=TaskStatusResponse, tags=["模型训练 (Training)"])
async def get_train_status(task_id: str):
    """
    **查询一个视频预处理（训练）任务的状态。**

    通过 `/train` 接口返回的 `task_id` 来轮询此接口。
    - 如果任务正在处理中，会返回处理中的状态。
    - 如果任务失败，会返回错误信息。
    - 如果任务成功完成，`code`会是200，并且响应中会包含一个可用于后续合成的 `model_id`。
    """
    logger.info(f"收到 /get_train 请求，查询任务ID: {task_id}")

    # 1. 直接从服务管理器的共享字典中获取任务的最新状态
    task_data = service_manager.get_task_status(task_id)

    # 2. 如果任务ID不存在，返回一个标准的“未找到”错误
    if not task_data:
        logger.warning(f"在 /get_train 中未找到任务ID: {task_id}")
        # 注意：这里直接返回字典，FastAPI会自动根据 response_model 进行转换
        return {
            "code": 400,
            "title": "任务不存在",
            "msg": "提供的 task_id 无效或任务从未提交。",
            "time": str(int(time.time())),
            "task_id": task_id
        }

    # 3. 解析从后端获取的状态，并将其映射到API响应
    internal_status = task_data.get('status', 'pending')
    response_data = {
        "time": str(int(time.time())),
        "task_id": task_id,
        "model_id": None # 默认 model_id 为 None
    }

    # 根据内部状态决定API的 code 和 title
    if internal_status == 'success':
        response_data.update({
            "code": 200,
            "title": "训练完成",
            "msg": "视频预处理成功，模型已生成。",
            # 【关键】从 task_data 的 result_path 字段中提取 model_id
            "model_id": task_data.get('result_path')
        })
    elif internal_status == 'error':
        response_data.update({
            "code": 500,
            "title": "训练失败",
            "msg": task_data.get('error_message', '未知错误导致预处理失败。')
        })
    else: # 所有其他的状态 (pending, preprocessing_*, etc.) 都视为“处理中”
        response_data.update({
            "code": 201,
            "title": "训练中",
            "msg": f"任务正在处理中，当前阶段: {internal_status} ({task_data.get('progress', 0)}%)"
        })

    logger.info(f"为任务 {task_id} 返回状态响应: {response_data}")
    return response_data

@app.post("/create_task", response_model=StandardResponse, tags=["视频合成 (Synthesis)"])
async def create_task(request: GenerateRequest):
    """
    **使用一个预处理好的模型（model_id）和一段新音频，发起视频合成任务。**

    这是核心的生成接口。它接收一个 `model_id` 和一个 `audio_url`，
    并将其提交到后端的“合成准备”生产线。
    这是一个异步操作，接口会立即返回一个 `synthesis_task_id`。
    您需要使用这个ID去轮询 `/get_result` 接口来获取最终生成的视频。
    """
    # 1. 为本次合成任务生成一个唯一的ID
    synthesis_task_id = request.task_id if request.task_id else str(uuid.uuid4())
    logger.info(f"收到 /create_task 请求，任务ID: {synthesis_task_id}，模型ID: {request.model_id}")

    # 2. 【关键交互】从服务管理器的共享字典中，查找模型信息
    # 这是连接“训练”和“合成”两个阶段的桥梁。
    model_info = service_manager.preprocess_results_dict.get(request.model_id)

    # 3. 校验 model_id 是否有效
    if not model_info or model_info.get('status') != 'completed':
        error_msg = f"模型ID '{request.model_id}' 无效或尚未准备就绪。"
        logger.warning(f"任务 {synthesis_task_id} 提交失败: {error_msg}")
        raise HTTPException(status_code=404, detail=error_msg)

    # 从模型信息中获取预处理时保存的本地视频文件路径
    local_video_path = model_info['local_video_path']
    logger.info(f"为模型ID '{request.model_id}' 找到对应的本地视频路径: {local_video_path}")

    # 4. 创建一个明确的“合成”类型任务对象
    task_to_submit = Task(
        task_id=synthesis_task_id,
        audio_url=request.audio_url,
        video_url=local_video_path,  # 【关键】使用查到的本地视频路径
        task_type='synthesis',  # 【关键】指定任务类型为 'synthesis'
        model_id=request.model_id,  # 传递 model_id，以便下游流程（如dispatcher）使用
        # 传递其他可选参数
        pn=request.pn,
        chaofen=request.chaofen
    )

    # 5. 通过服务管理器，调用专用的方法提交任务
    success, message = service_manager.submit_synthesis_task(task_to_submit)

    # 6. 如果提交失败，向上层抛出HTTP异常
    if not success:
        logger.error(f"任务 {synthesis_task_id} 提交失败: {message}")
        raise HTTPException(status_code=500, detail=f"合成任务提交至后端服务失败: {message}")

    # 7. 如果提交成功，立即返回标准响应
    response_data = {
        "code": 200,
        "msg": "视频合成任务已成功提交。",
        "time": str(int(time.time())),
        "task_id": synthesis_task_id
    }
    logger.info(f"为任务 {synthesis_task_id} 返回成功响应: {response_data}")
    return response_data

@app.get("/get_result", response_model=TaskStatusResponse, tags=["视频合成 (Synthesis)"])
async def get_generation_status(task_id: str):
    """
    **查询一个视频合成任务的状态。**

    通过 `/create_task` 接口返回的 `task_id` 来轮询此接口。
    - 如果任务正在处理中，会返回处理中的状态。
    - 如果任务失败，会返回错误信息。
    - 如果任务成功完成，`code`会是200，并且响应中会包含最终生成的视频 `oss_url`。
    """
    logger.info(f"收到 /get_result 请求，查询任务ID: {task_id}")

    # 1. 逻辑与 /get_train 完全一致：直接从服务管理器的共享字典中获取状态
    task_data = service_manager.get_task_status(task_id)

    # 2. 如果任务ID不存在，返回“未找到”错误
    if not task_data:
        logger.warning(f"在 /get_result 中未找到任务ID: {task_id}")
        return {
            "code": 400,
            "title": "任务不存在",
            "msg": "提供的 task_id 无效或任务从未提交。",
            "time": str(int(time.time())),
            "task_id": task_id
        }

    # 3. 解析从后端获取的状态，并将其映射到API响应
    internal_status = task_data.get('status', 'pending')
    response_data = {
        "time": str(int(time.time())),
        "task_id": task_id,
        "model_id": task_data.get('model_id')  # 合成任务的状态中应包含它所使用的model_id
    }

    # 根据内部状态决定API的 code 和 title
    if internal_status == 'success':
        response_data.update({
            "code": 200,
            "title": "合成完成",
            "msg": "视频合成成功。",
            # 【关键】从 task_data 的 result_path 字段中提取最终的视频URL
            "oss_url": task_data.get('result_path')
        })
    elif internal_status == 'error':
        response_data.update({
            "code": 500,
            "title": "合成失败",
            "msg": task_data.get('error_message', '未知错误导致合成失败。')
        })
    else:  # 所有其他的状态 (pending, synthesis_*, waiting_dispatch, etc.) 都视为“处理中”
        response_data.update({
            "code": 201,
            "title": "合成中",
            "msg": f"任务正在处理中，当前阶段: {internal_status} ({task_data.get('progress', 0)}%)"
        })

    logger.info(f"为任务 {task_id} 返回状态响应: {response_data}")
    return response_data


# --- 应用启动入口 (用于直接运行测试) ---
if __name__ == "__main__":
    """
    这个代码块只会在 `python api.py` 直接运行时执行。
    当子进程导入此文件时，这里的代码不会被执行，从而避免了 RuntimeError。
    """
    logger.info("正在启动Uvicorn服务...")

    # 将 Uvicorn 的运行放在这里
    uvicorn.run(
        "api:app",  # 使用字符串路径 "文件名:FastAPI实例名"
        host="0.0.0.0",
        port=8000,
        reload=False  # 在生产或调试多进程时建议关闭热重载
    )
