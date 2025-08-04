# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/service/server.py
# Compiled at: 2024-04-10 09:16:17
# Size of source mod 2**32: 1621 bytes
"""
@author: miracle
@version: 1.0.0
@license: Apache Licence
@file: server.py
@time: 2024/4/9 17:49
"""
from h_utils.request_utils import request_post
from y_utils.config import GlobalConfig
from y_utils.logger import logger as logger
import os, subprocess, time

def get_uuid():
    output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    for line in output.splitlines():
        if "UUID" in line:
            gpu_uuid = line.split("GPU-")[-1]
            gpu_uuid = gpu_uuid.replace(")", "")
            return gpu_uuid
        return ""
    return None


def read_params():
    with open("/code/license.txt", "r", encoding="utf-8") as ff:
        lc = ff.read()
    uuid = get_uuid()
    return {'uuid':uuid, 
     'license':lc}


def register_host():
    if not os.path.exists(GlobalConfig.instance().register_file):
        logger.warn("local register file not exists, register it.")
        result = request_post((GlobalConfig.instance().register_url + "/active"), (read_params()), timeout=60)
        if result:
            os.mknod(GlobalConfig.instance().register_file)
        return result
    logger.warn("local register file exists, skip.")
    return True


def repost_host():
    import requests
    headers = {"content-type": "application/json"}
    while True:
        try:
            requests.post((GlobalConfig.instance().register_url + "/report"), json=(read_params()), headers=headers, timeout=60)
        except BaseException as ex:
            try:
                pass
            finally:
                ex = None
                del ex

        else:
            time.sleep(GlobalConfig.instance().register_report_interval)

