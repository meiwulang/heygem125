# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/h_utils/custom.py
# Compiled at: 2024-03-27 17:14:52
# Size of source mod 2**32: 317 bytes
"""
@project : dhp-tools
@author  : huyi
@file   : custom.py
@ide    : PyCharm
@time   : 2021-08-18 20:18:23
"""

class CustomError(Exception):

    def __init__(self, errorinfor):
        self.error = errorinfor

    def __str__(self):
        return self.error

