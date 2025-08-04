"""
File: config.py
Author: YuFangHui
Date: 2020-11-25
Description:
"""
import configparser, os
from y_utils import config
from y_utils import tools
base_dir = "./"


def get_config():
    """
    读取主配置文件。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config", "config.ini")

    # --- 【新增调试代码 1】 ---
    # print(f"--- CONFIG DEBUG [get_config] ---", flush=True)
    # print(f"Attempting to read config file from: {config_path}", flush=True)
    # --- 调试代码结束 ---

    config = configparser.ConfigParser()

    if not os.path.exists(config_path):
        # print(f"FATAL: File not found at the path above.", flush=True)
        raise FileNotFoundError(f"配置文件未找到！请确保 'config.ini' 文件存在于指定路径: {config_path}")

    config.read(config_path, encoding="utf-8")

    # --- 【新增调试代码 2】 ---
    # print(f"Successfully read config file. Sections found: {config.sections()}", flush=True)
    # print(f"--- END CONFIG DEBUG [get_config] ---", flush=True)
    # --- 调试代码结束 ---

    return config


class GeneralConfig:

    def __init__(self, config_path, section):
        self.conf = configparser.ConfigParser()
        self.conf.read(config_path)
        self.section = section
        print(self.conf.__dict__)

    def __getattr__(self, item):
        return self._GeneralConfig__get_option(item, self.section)

    def __get_option(self, option, section):
        """内部方法：获取具体配置值"""
        try:
            cfg_val = self.conf.get(section, option)
            return cfg_val
        except Exception as e:
            print(e)
            return None

class GlobalConfig:
    # 将单例实例的创建移到类级别，确保在任何实例方法调用前都存在
    _instance = None

    @classmethod
    def instance(cls, *args, **kwargs):
        """
        获取 GlobalConfig 的单例实例。
        这是访问全局配置的唯一入口。
        """
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    def get_config(self, section, key, default_value=None):
        """
        通用方法，从已加载的配置中获取字符串值。
        """
        main_config = get_config()
        # 使用插值来自动处理像 %(data_root)s 这样的变量
        try:
            value = main_config.get(section, key, fallback=default_value)
            return value
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default_value

    def get_boolean_config(self, section, key, default_value=False):
        """
        专门用于获取布尔值（True/False）的配置。
        """
        value_str = self.get_config(section, key)
        if value_str is None:
            return default_value
        # 明确地将字符串 'true', '1', 'yes', 'on' 视为 True
        return str(value_str).lower() in ['true', '1', 'yes', 'on']

    def __init__(self):
        """
        初始化全局配置类。
        加载所有配置，并将所有路径配置转换为干净、规范化的绝对路径。
        """
        if GlobalConfig._instance is not None:
            raise Exception("This is a singleton class. Use GlobalConfig.instance() to get the object.")

        # --- 第一步：确定项目根目录的绝对路径 ---
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # --- 第二步：读取、拼接并规范化所有路径 ---

        # [paths]
        self.data_root = os.path.normpath(os.path.join(project_root, self.get_config("paths", "data_root", "./data")))
        self.models_root = os.path.normpath(
            os.path.join(project_root, self.get_config("paths", "models_root", "./models")))
        self.assets_root = os.path.normpath(
            os.path.join(project_root, self.get_config("paths", "assets_root", "./assets")))

        # 使用根目录变量来构建子路径，然后规范化
        self.log_dir = os.path.normpath(os.path.join(self.data_root, "logs/"))
        self.result_dir = os.path.normpath(os.path.join(self.data_root, "results/"))
        self.temp_dir = os.path.normpath(os.path.join(self.data_root, "temp/"))
        self.cache_dir = os.path.normpath(os.path.join(self.data_root, "cache/"))
        self.debug_output_dir = os.path.normpath(os.path.join(self.data_root, "debug/"))

        self.face_detect_resources = os.path.normpath(os.path.join(self.models_root, "face_detection/"))
        self.head_pose_model = os.path.normpath(os.path.join(self.models_root, "face_detection/model_float32.onnx"))
        self.wenet_config_path = os.path.normpath(os.path.join(project_root,
                                                               self.get_config("paths", "wenet_config_path",
                                                                               "wenet/examples/aishell/aidata/conf/train_conformer_multi_cn.yaml")))
        self.wenet_model_path = os.path.normpath(os.path.join(project_root, self.get_config("paths", "wenet_model_path",
                                                                                            "wenet/examples/aishell/aidata/exp/conformer/wenetmodel.pt")))

        self.watermark_path = os.path.normpath(os.path.join(self.assets_root, "watermark.png"))
        self.digital_auth_path = os.path.normpath(os.path.join(self.assets_root, "auth_logo.png"))
        self.dummy_audio_path = os.path.normpath(os.path.join(self.assets_root, "dummy_silent.wav"))

        # [log]
        self.log_file = self.get_config("log", "log_file", "dh.log")
        # [performance]
        self.batch_size = int(self.get_config("performance", "synthesis_batch_size", 128))
        self.preprocess_batch_size = int(self.get_config("performance", "preprocess_batch_size", 128))

        # [workers]
        self.num_gpus = int(self.get_config("workers", "num_gpus", 2))
        self.num_preprocess_workers = int(self.get_config("workers", "num_preprocess_workers", 2))
        self.num_synthesis_prep_workers = int(self.get_config("workers", "num_synthesis_prep_workers", 6))

        # [queues]
        self.preprocess_request_queue_size = int(self.get_config("queues", "preprocess_request_queue_size", 100))
        self.synthesis_request_queue_size = int(self.get_config("queues", "synthesis_request_queue_size", 100))
        self.ready_queue_factor = int(self.get_config("queues", "ready_queue_factor", 2))
        self.gpu_pipeline_queue_factor = int(self.get_config("queues", "gpu_pipeline_queue_factor", 8))
        self.result_queue_size = int(self.get_config("queues", "result_queue_size", 100))

        # [processing]
        self.blend_dynamic = self.get_config("processing", "blend_dynamic", "xseg")
        self.blur_threshold = float(self.get_config("processing", "blur_threshold", 0.6))
        self.ffmpeg_crf = int(self.get_config("processing", "ffmpeg_crf", 18))
        self.std_face_crop_size = int(self.get_config("processing", "std_face_crop_size", 256))
        self.chaofen_before = self.get_boolean_config("processing", "chaofen_before", False)
        self.chaofen_after = self.get_boolean_config("processing", "chaofen_after", False)

        # [flags]
        self.enable_debug_save = self.get_boolean_config("flags", "enable_debug_save", False)
        self.temp_clean_switch = self.get_boolean_config("flags", "temp_clean_switch", True)

        # [register]
        self.register_url = self.get_config("register", "url", "")
        self.register_report_interval = int(self.get_config("register", "report_interval", 3600))
        self.register_enable = self.get_boolean_config("register", "enable", False)

        self.use_tensorrt = self.get_boolean_config("trt", "use_tensorrt", False)
        self.tensorrt_max_batch_size = int(self.get_config("trt", "tensorrt_max_batch_size", 60))

        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self._rebuild_paths()

    def force_set_project_root(self, new_root_path):
        """
        强制设置项目根目录，并重新构建所有绝对路径。
        这在子进程中特别有用，可以确保路径的正确性。
        """
        self.project_root = new_root_path
        self._rebuild_paths()

    def _rebuild_paths(self):
        """
        一个内部辅助方法，使用当前的 self.project_root 属性
        来重新构建和规范化所有与路径相关的配置属性。
        """
        # [paths] - 读取相对路径字符串
        data_root_rel = self.get_config("paths", "data_root", "./data")
        models_root_rel = self.get_config("paths", "models_root", "./models")
        assets_root_rel = self.get_config("paths", "assets_root", "./assets")

        # 使用 self.project_root 构建绝对路径并规范化
        self.data_root = os.path.normpath(os.path.join(self.project_root, data_root_rel))
        self.models_root = os.path.normpath(os.path.join(self.project_root, models_root_rel))
        self.assets_root = os.path.normpath(os.path.join(self.project_root, assets_root_rel))

        # 使用新生成的绝对根目录构建并规范化所有子路径
        self.log_dir = os.path.normpath(os.path.join(self.data_root, "logs/"))
        self.result_dir = os.path.normpath(os.path.join(self.data_root, "results/"))
        self.temp_dir = os.path.normpath(os.path.join(self.data_root, "temp/"))
        self.cache_dir = os.path.normpath(os.path.join(self.data_root, "cache/"))
        self.debug_output_dir = os.path.normpath(os.path.join(self.data_root, "debug/"))

        self.face_detect_resources = os.path.normpath(os.path.join(self.project_root,
                                                                   self.get_config("paths", "face_detect_resources",
                                                                                   "face_detect_utils/resources/")))
        self.head_pose_model = os.path.normpath(os.path.join(self.project_root,
                                                             self.get_config("paths", "head_pose_model",
                                                                             "face_detect_utils/resources/model_float32.onnx")))

        self.wenet_config_path = os.path.normpath(os.path.join(self.project_root,
                                                               self.get_config("paths", "wenet_config_path",
                                                                               "wenet/examples/aishell/aidata/conf/train_conformer_multi_cn.yaml")))
        self.wenet_model_path = os.path.normpath(os.path.join(self.project_root,
                                                              self.get_config("paths", "wenet_model_path",
                                                                              "wenet/examples/aishell/aidata/exp/conformer/wenetmodel.pt")))

        self.watermark_path = os.path.normpath(os.path.join(self.assets_root, "watermark.png"))
        self.digital_auth_path = os.path.normpath(os.path.join(self.assets_root, "auth_logo.png"))
        self.dummy_audio_path = os.path.normpath(os.path.join(self.assets_root, "dummy_silent.wav"))

        # [trt] - 构建 TensorRT 引擎文件的绝对路径
        tensorrt_engine_path_rel = self.get_config("trt", "tensorrt_engine_path", "models/default.engine")

        # 【核心修正】清理从配置文件读取的字符串，去除首尾的空格和双引号
        cleaned_path_rel = tensorrt_engine_path_rel.strip().strip('"')

        self.tensorrt_engine_path = os.path.normpath(os.path.join(self.project_root, cleaned_path_rel))

        # [paths] - 为 TRT 模式，构建原始 .pth 模型文件的绝对路径，用于加载元数据
        dinet_model_path_rel = self.get_config("trt", "dinet_original_model_path", "models/default.pth")
        cleaned_dinet_path_rel = dinet_model_path_rel.strip().strip('"')
        self.dinet_original_model_path = os.path.normpath(os.path.join(self.project_root, cleaned_dinet_path_rel))
