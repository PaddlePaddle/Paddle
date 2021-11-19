#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import contextlib
from collections import abc
from enum import Enum
from math import inf

import paddle
import paddle.distributed as dist
from paddle.fluid import core

# Set global device id
global dev_id
if core.is_compiled_with_cuda():
    dev_id = int(os.environ.get('FLAGS_selected_gpus', 0))
elif core.is_compiled_with_npu():
    dev_id = int(os.environ.get('FLAGS_selected_npus', 0))
else:
    raise ValueError("This device doesn't support.")


class Taskflow:
    """
    Task flows, one way linked list for task acquisition.
    """

    def __init__(self, task, callback):
        self.task = task
        self.callback = callback


class Type(Enum):
    """
    Type of trainable parameters
    """
    fp16 = paddle.float16
    fp32 = paddle.float32


def GpuInfo(fn):
    """
    Displays GPU usage information before and after the function。
    """

    def used(*args, **kw):
        # Before using
        b_info = os.popen("nvidia-smi -i {} | grep MiB".format(str(
            dev_id))).read()
        before_info = (int(b_info.split()[8][:-3]),
                       int(b_info.split()[10][:-3]))
        print(
            "====== Current device {} ====== Total has {} MiB, Has used {} MiB ======".
            format(str(dev_id), str(before_info[1]), str(before_info[0])))
        result = fn(*args, **kw)
        # After using
        a_info = os.popen("nvidia-smi -i {} | grep MiB".format(str(
            dev_id))).read()
        after_info = (int(a_info.split()[8][:-3]), int(a_info.split()[10][:-3]))
        print(
            "====== Current device {} ====== Total has {} MiB, Has used {} MiB, Self use {} MiB ======".
            format(
                str(dev_id),
                str(after_info[1]),
                str(after_info[0]), str(after_info[0] - before_info[0])))
        return result

    return used


@contextlib.contextmanager
def device_guard(dev_id, device="cpu"):
    origin_device = paddle.device.get_device()
    if device == "cpu":
        paddle.set_device(device)
    elif device == "gpu":
        paddle.set_device("gpu:{}".format(dev_id))
    try:
        yield
    finally:
        paddle.set_device(origin_device)
