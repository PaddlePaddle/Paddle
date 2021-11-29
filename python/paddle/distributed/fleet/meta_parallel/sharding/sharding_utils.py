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
