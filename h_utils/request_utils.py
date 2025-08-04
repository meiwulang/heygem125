"""
@project : dhp-tools
@author  : huyi
@file   : request_utils.py
@ide    : PyCharm
@time   : 2021-09-03 16:00:32
"""

import json
import os
import sys
import time
import requests
from y_utils.logger import logger


def request_post(url, param, timeout):
    result = 0
    fails = 0
    while fails < 3:
        headers = {
            'content-type': 'application/json'
        }
        ret = requests.post(url, json=param, headers=headers, timeout=timeout)

        if ret.status_code == 200:
            text = ret.text
            if json.loads(text)['code'] != 0:
                fails += 1
                logger.info('第[{}]次失败回调结果为:{}'.format(fails, text))
                time.sleep(3)
            else:
                logger.info('成功回调结果为:{}'.format(text))
                result = 1
                break
        else:
            fails += 1
            logger.error('第[{}]次回调异常报错:{}'.format(fails, ret.status_code))
            logger.info('网络连接出现问题, 正在尝试再次请求: {}'.format(fails))
            time.sleep(3)
    return result


def download(url, file_path):
    count = 0
    r1 = requests.get(url, stream=True, verify=False, timeout=20)
    total_size = int(r1.headers['Content-Length'])

    temp_size = 0
    if os.path.exists(file_path):
        temp_size = os.path.getsize(file_path)

    while count < 10:
        if temp_size >= total_size:
            break
        count += 1
        logger.info('第[{}]次下载文件,已经下载数据大小:[{}],未下载数据大小:[{}]'.format(count, temp_size, total_size))

        headers = {
            'Range': 'bytes={}-{}'.format(temp_size, total_size - 1)
        }
        r = requests.get(url, stream=True, verify=False, headers=headers, timeout=20)

        with open(file_path, 'ab') as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    temp_size += len(chunk)
                    f.write(chunk)
                    f.flush()
                done = int(50 * temp_size / total_size)
                sys.stdout.write('\r[%s%s] %d%%' % ('█' * done, ' ' * (50 - done), 100 * temp_size // total_size))
                sys.stdout.flush()
    print('\n')
    return file_path


def download_file(file_url, local_path):
    chunk_size = 4096
    lst_size = 0
    total_size = 0

    r = requests.get(file_url, stream=True, verify=False, timeout=120)
    with open(local_path, 'wb') as ff:
        for chunk in r.iter_content(chunk_size=chunk_size):
            total_size += len(chunk)
            ff.write(chunk)
    return local_path
