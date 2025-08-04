#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import shutil
import os
from multiprocessing import Pool




def packaged_search(path, directory_file=None):
    '''
    遍历当前目录下文件及文件夹
    :param path:
    :param directory_file:
    :return:
    '''
    if directory_file:
        for i in os.listdir(path):
            if i == directory_file:
                path = os.path.join(path, directory_file)
                pack_so(path)
            elif os.path.isdir(os.path.join(path, i)):
                packaged_search(os.path.join(path, i), directory_file)
    else:
        pack_so(path)


def pack_so(path):
    '''
    递归遍历所有文件夹，并创建进程池，将任务放入进程
    :param path:
    :return:
    '''
    all_file_path = []
    for i in os.listdir(path):
        all_file_path.append(os.path.join(path, i))
    # 创建进程池
    p = Pool(8)
    for j in all_file_path:
        p.apply_async(pack_to_so_and_del_src, args=(j, ))
    p.close()
    p.join()
    for g in all_file_path:
        # 是文件夹递归
        if os.path.isdir(os.path.join(g)):
            pack_so(g)


def pack_to_so_and_del_src(path):
    '''
    将需要打包的.py脚本进行打包
    :param path:
    :return:
    '''
    if '.py' in path and '.pyc' not in path and '__init__.py' not in path:
        setup(
            ext_modules=cythonize(Extension(path.rsplit('/', 1)[1].rsplit('.', 1)[0], [path])),
            compiler_directives={'language_level': 3}
        )
        # path_os = os.getcwd().rsplit('/', 1)[0] + '/pack/build/lib.linux-x86_64-3.6'  # TODO
        path_os = os.getcwd().rsplit('/', 1)[0] + '/pack/build/lib.linux-x86_64-3.8'
        for j in os.listdir(path_os):
            # 将打好的包放入原文件夹下
            shutil.move(os.path.join(path_os, j), os.path.join(path.rsplit('/', 1)[0], j))
            # 删除.py文件
            # if path.rsplit('/', 1)[1] not in ['packaging_script.py', 'manage.py', 'client.py']:
            if path.rsplit('/', 1)[1] not in ['packaging_script.py', 'app.py', 'app_local.py', 'tts_config.py']:
                os.remove(path)
        # shutil.rmtree('./build')
    # 删除.c文件
    elif len(path.rsplit('.', 1)) == 2:
        if path.rsplit('.', 1)[1] == 'c':
            os.remove(path)


def view_log():
    '''
    删除log日志文件
    :return:
    '''
    pass


if __name__ == '__main__':
    path = os.getcwd().rsplit('/', 1)[0]
    packaged_search(path)
    # 查看版本号并创建外文件写入
    # edition = os.popen('git show')
    # with open('./edition.txt', 'w') as e:
    #     e.write(edition.readline())

"""

usage:
    python3 packaging_script.py build_ext
打包说明：

"""