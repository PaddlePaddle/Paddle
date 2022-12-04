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

import sys

import config
import numpy as np
from parameterized import parameterized, parameterized_class

TEST_CASE_NAME = 'suffix'


def xrand(shape=(10, 10, 10), dtype=config.DEFAULT_DTYPE, min=1.0, max=10.0):
    return (np.random.rand(*shape).astype(dtype)) * (max - min) + min


def place(devices, key='place'):
    def decorate(cls):
        module = sys.modules[cls.__module__].__dict__
        raw_classes = {
            k: v for k, v in module.items() if k.startswith(cls.__name__)
        }

        for raw_name, raw_cls in raw_classes.items():
            for d in devices:
                test_cls = dict(raw_cls.__dict__)
                test_cls.update({key: d})
                new_name = raw_name + '.' + d.__class__.__name__
                module[new_name] = type(new_name, (raw_cls,), test_cls)
            del module[raw_name]
        return cls

    return decorate


# alias
parameterize = parameterized
param_cls = parameterized_class
param_func = parameterized
