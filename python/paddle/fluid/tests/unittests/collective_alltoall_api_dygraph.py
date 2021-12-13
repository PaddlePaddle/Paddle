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
import unittest
from multiprocessing import Process
import paddle.fluid.layers as layers
from functools import reduce
from test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main


class TestCollectiveAllToAllAPI(TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        with fluid.program_guard(main_prog, startup_program):
            tindata = paddle.to_tensor(indata)
            tindata = paddle.split(tindata, 2, axis=0)
            tout_data = []
            paddle.distributed.alltoall(tindata, tout_data)
            output_data = []
            for data in tout_data:
                output_data.append(data.numpy())
            return output_data


if __name__ == "__main__":
    runtime_main(TestCollectiveAllToAllAPI, "alltoall")
