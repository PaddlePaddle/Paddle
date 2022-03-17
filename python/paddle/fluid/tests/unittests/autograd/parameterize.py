# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module provide parameterized test functions.
"""
import collections
import contextlib
import functools
import inspect
import re
import sys

import numpy as np

TEST_CASE_NAME = 'suffix'


def place(devices, key='place'):
    """A Decorator for a class which will make the class running on different 
    devices .

    Args:
        devices (Sequence[Paddle.CUDAPlace|Paddle.CPUPlace]): Device list.
        key (str, optional): Defaults to 'place'.
    """

    def decorate(cls):
        module = sys.modules[cls.__module__].__dict__
        raw_classes = {
            k: v
            for k, v in module.items() if k.startswith(cls.__name__)
        }

        for raw_name, raw_cls in raw_classes.items():
            for d in devices:
                test_cls = dict(raw_cls.__dict__)
                test_cls.update({key: d})
                new_name = raw_name + '.' + d.__class__.__name__
                module[new_name] = type(new_name, (raw_cls, ), test_cls)
            del module[raw_name]
        return cls

    return decorate


def parameterize(fields, values=None):
    """Decorator for a unittest class which make the class running on different 
    test cases.

    Args:
        fields (Sequence): The feild name sequence of test cases.
        values (Sequence, optional): The test cases sequence. Defaults to None.

    """
    fields = [fields] if isinstance(fields, str) else fields
    params = [dict(zip(fields, vals)) for vals in values]

    def decorate(cls):
        test_cls_module = sys.modules[cls.__module__].__dict__
        for i, values in enumerate(params):
            test_cls = dict(cls.__dict__)
            values = {
                k: staticmethod(v) if callable(v) else v
                for k, v in values.items()
            }
            test_cls.update(values)
            name = cls.__name__ + str(i)
            name = name + '.' + \
                values.get('suffix') if values.get('suffix') else name

            test_cls_module[name] = type(name, (cls, ), test_cls)

        for m in list(cls.__dict__):
            if m.startswith("test"):
                delattr(cls, m)
        return cls

    return decorate
