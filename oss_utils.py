import os
import time
import datetime
import logging
import uuid
from functools import lru_cache

import oss2

# 只获取logger实例，不进行配置
logger = logging.getLogger("oss_utils")


# 使用 lru_cache 来实现一个简单的、线程安全的单例模式
# maxsize=1 意味着这个函数的返回值会被缓存，下次以相同参数调用时直接返回缓存结果
@lru_cache(maxsize=1)
def get_oss_manager():
    """
    获取 OSSManager 的单例实例。
    在第一次被调用时，它会初始化并测试连接。
    """
    logger.info("First call to get_oss_manager, attempting to initialize...")
    # 延迟导入，避免循环依赖
    from y_utils.config import GlobalConfig
    cfg = GlobalConfig.instance()

    # 从配置中读取所有OSS相关参数
    access_key_id = cfg.get_config("obs", "access_key_id")
    access_key_secret = cfg.get_config("obs", "secret_access_key")
    bucket_name = cfg.get_config("obs", "bucket")
    endpoint = cfg.get_config("obs", "obs_server")  # obs_server 对应 endpoint

    # 检查核心配置是否存在
    if not all([access_key_id, access_key_secret, bucket_name, endpoint]):
        logger.warning("OSS configuration is incomplete in config.ini. OSS functionalities will be disabled.")
        return None  # 如果配置不完整，返回None

    return OSSManager(access_key_id, access_key_secret, bucket_name, endpoint)


class OSSManager:
    """
    Manager for Aliyun OSS operations.
    This class should be instantiated via the get_oss_manager() function.
    """

    def __init__(self, access_key_id, access_key_secret, bucket_name, endpoint):
        """
        Initialize the OSS manager. This is now a private-like method.
        """
        self.bucket_name = bucket_name
        self.endpoint = endpoint
        self.bucket_domain = f"{bucket_name}.{endpoint}"
        self.bucket = None  # 先声明为None

        try:
            self.auth = oss2.Auth(access_key_id, access_key_secret)
            # 注意endpoint前面需要协议头
            self.bucket = oss2.Bucket(self.auth, f"https://{self.endpoint}", self.bucket_name)
            # 测试连接
            self.bucket.get_bucket_info()
            logger.info("Successfully connected to Aliyun OSS.")
        except Exception as e:
            logger.error(f"Failed to connect to Aliyun OSS: {e}")
            # 如果连接失败，将bucket重置为None，以便后续方法可以检查
            self.bucket = None

    def _is_ready(self):
        """检查OSS客户端是否已成功初始化"""
        if self.bucket is None:
            logger.error("OSSManager is not ready. Check connection or configuration.")
            return False
        return True

    def upload_file(self, local_file_path: str, model_id: str):
        if not self._is_ready():
            return False, None


        try:
            now = datetime.datetime.now()
            year_month = now.strftime("%Y-%m")
            file_extension = os.path.splitext(local_file_path)[1]
            # 为了避免文件名冲突，依然使用uuid
            oss_key = f"{year_month}/{model_id}-{uuid.uuid4().hex[:8]}{file_extension}"

            result = self.bucket.put_object_from_file(oss_key, local_file_path)

            if result.status == 200:
                url = f"https://{self.bucket_domain}/{oss_key}"
                logger.info(f"Successfully uploaded file to {url}")
                return True, url
            else:
                logger.error(f"Failed to upload file, status code: {result.status}")
                return False, None
        except Exception as e:
            logger.error(f"Error uploading file to OSS: {e}")
            return False, None

    # upload_bytes 方法逻辑类似，也需要先检查 self._is_ready()

    def upload_video(self, local_file_path: str, task_id: str):
        if not self._is_ready():
            return False, None


        try:
            now = datetime.datetime.now()
            year_month_day = now.strftime("%Y-%m-%d")
            file_extension = os.path.splitext(local_file_path)[1]
            oss_key = f"{year_month_day}/{task_id}-{uuid.uuid4().hex[:8]}{file_extension}"
            # headers = {'Content-Disposition': f'attachment; filename="{os.path.basename(local_file_path)}"'}

            result = self.bucket.put_object_from_file(oss_key, local_file_path)

            if result.status == 200:
                url = f"https://{self.bucket_domain}/{oss_key}"
                logger.info(f"Successfully uploaded video to {url}")
                return True, url
            else:
                logger.error(f"Failed to upload video, status code: {result.status}")
                return False, None
        except Exception as e:
            logger.error(f"Error uploading video to OSS: {e}")
            return False, None

# --- 【核心修改】移除顶层的单例创建 ---
# 原始代码: oss_manager = OSSManager()
# 现在不再需要这一行。所有对OSS的操作都应通过 get_oss_manager() 获取实例。