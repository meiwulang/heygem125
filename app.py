import os
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
import time
import uuid
import gradio as gr

import service.trans_dh_service
from y_utils.config import GlobalConfig
from y_utils.logger import logger

class VideoProcessor:
    def __init__(self):
        self.task = service.trans_dh_service.TransDhTask()
        self.basedir = GlobalConfig.instance().result_dir
        self.is_initialized = False
        self._initialize_service()
        print("VideoProcessor init done")

    def _initialize_service(self):
        logger.info("开始初始化 trans_dh_service...")
        try:
            time.sleep(5)
            logger.info("trans_dh_service 初始化完成。")
            self.is_initialized = True
        except Exception as e:
            logger.error(f"初始化 trans_dh_service 失败: {e}")

    def process_video(
        self, mp4_path,audio_file
    ):
        while not self.is_initialized:
            logger.info("服务尚未完成初始化，等待 1 秒...")
            time.sleep(1)
        work_id = str(uuid.uuid1())
        code = work_id

        try:
            audio_path = audio_file
            self.task.task_dic[code] = ""
            xx = self.task.work_ausn(mp4_path,audio_path)
            return xx
        except Exception as e:
            raise gr.Error(str(e))


if __name__ == "__main__":
    processor = VideoProcessor()
    inputs = [
        gr.File(label="上传视频文件，"),
        gr.File(label="上传音频文件，")
    ]
    outputs = gr.Video(label="生成的视频")
    title = "数字人视频生成"
    description = "上传音频和视频文件，即可生成数字人视频"
    demo = gr.Interface(
        fn=processor.process_video,
        inputs=inputs,
        outputs=outputs,
        title=title,
        description=description,
    )
    demo.queue().launch(
        inbrowser=True,
        server_name="0.0.0.0",  # 监听所有网络接口
        server_port=7860,  # 指定端口
        share=False  # 不需要公共链接
    )