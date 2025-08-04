# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/h_utils/sweep_bot.py
# Compiled at: 2024-03-27 17:14:52
# Size of source mod 2**32: 1468 bytes
"""
@project : ai_detection_server
@author  : huyi
@file   : sweep_bot.py
@ide    : PyCharm
@time   : 2021-12-08 12:02:29
"""
import os, shutil
from y_utils.logger import logger as logger

def sweep(rubbish: list, flag=True):
    if flag:
        try:
            for x in rubbish:
                if os.path.exists(x) is False:
                    logger.info("扫地机器人无法找到目标:[{}]".format(x))
                if os.path.isfile(x):
                    logger.info("扫地机器人清理文件:[{}]".format(x))
                    os.remove(x)
                if os.path.isdir(x):
                    logger.info("扫地机器人清理目录:[{}]".format(x))
                for filename in os.listdir(x):
                    file_path = os.path.join(x, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    else:
                        if os.path.isdir(file_path):
                            shutil.rmtree(file_path)

        except Exception as e:
            try:
                logger.error("扫地机器人工作异常，异常信息:[{}]".format(e.__str__()))
            finally:
                e = None
                del e

    else:
        logger.info("扫地机器人关闭，无法工作")

