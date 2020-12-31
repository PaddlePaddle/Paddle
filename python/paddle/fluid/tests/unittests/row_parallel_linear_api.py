# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import argparse
import os
import sys
import signal
import time
import socket
from contextlib import closing
from six import string_types
import math
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle.fluid.unique_name as nameGen
from paddle.fluid import core
import paddle.distributed.fleet as fleet
from paddle.fluid.incubate.fleet.base import role_maker
import unittest
from multiprocessing import Process
import paddle.fluid.layers as layers
from functools import reduce
from test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main

paddle.enable_static()


class TestRowParallelLinearAPI(TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank):
        with fluid.program_guard(main_prog, startup_program):
            fleet.init(is_collective=True)
            np.random.seed(2020)
            np_array = np.random.rand(1000, 16)

            data = paddle.static.data(
                name='tindata', shape=[10, 1000], dtype="float32")
            paddle.distributed.broadcast(data, src=0)
            data = paddle.split(data, 2, axis=1)[rank]
            if rank == 0:
                param_attr = paddle.fluid.ParamAttr(
                    initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                        np_array[0:500, :]), )
            else:
                param_attr = paddle.fluid.ParamAttr(
                    initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                        np_array[500:1000, :]), )

            linear_out = paddle.distributed.split(
                data,
                size=(1000, 8),
                operation='linear',
                axis=0,
                num_partitions=2,
                weight_attr=param_attr,
                bias_attr=False, )

            return [linear_out]


if __name__ == "__main__":
    runtime_main(TestRowParallelLinearAPI, "row_parallel_linear")
