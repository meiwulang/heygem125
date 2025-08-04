# 文件: y_utils/logger.py

import logging
import os
from logging.handlers import TimedRotatingFileHandler

# 1. 在模块顶层，只创建一个临时的、未配置的 logger 对象。
#    它现在只是一个占位符，不会读取任何配置。
logger = logging.getLogger("DigitalHumanService")
logger.setLevel(logging.INFO)  # 设置一个默认级别
# 添加一个临时的控制台处理器，以便在日志系统完全配置好之前就能看到启动信息
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


# 2. 将所有配置和初始化逻辑都封装到一个独立的函数中。
def setup_logger():
    """
    根据全局配置来设置真正的文件和控制台日志记录器。
    这个函数应该在主程序启动时，在所有其他业务逻辑之前被调用一次。
    """
    # 延迟导入，确保在调用时 GlobalConfig 已经准备就绪
    from y_utils.config import GlobalConfig

    # 这一步现在是安全的，因为它是在应用启动时被调用的
    cfg = GlobalConfig.instance()

    # 从配置中获取日志目录和文件名
    log_dir = cfg.log_dir
    log_file = cfg.log_file

    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 清除之前默认添加的临时处理器
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件处理器 (按天轮转，保留7天)
    log_path = os.path.join(log_dir, log_file)
    file_handler = TimedRotatingFileHandler(
        log_path, when="midnight", interval=1, backupCount=7, encoding='utf-8'
    )
    # 定义日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 将新的、配置好的处理器添加到全局 logger 对象
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 现在 logger 才算真正配置完成
    logger.info("Logger has been fully configured and is active.")

# 3. 移除旧的 create_logger 函数，或将其逻辑合并到 setup_logger 中。
#    为保持简洁，我们已经将其逻辑合并。