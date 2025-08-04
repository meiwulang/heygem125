# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/y_utils/tools.py
# Compiled at: 2024-03-27 17:15:16
# Size of source mod 2**32: 605 bytes
"""
File: service.py
Author: YuFangHui
Date: 2020-11-25
Description:
"""
import os
from os.path import exists

def check_dir(dir_path, flag=True):
    if exists(dir_path):
        return True
    if flag:
        os.makedirs(dir_path)
    return exists(dir_path)


def print_content(content):
    print("type:", type(content), "\n", content, "\n")


def read_file(path):
    f = open(path, "r", encoding="utf-8")
    content = f.read()
    f.close()
    return content


def write_file(path, content):
    f = open(path, "w", encoding="utf-8")
    f.write(content)
    f.close()

