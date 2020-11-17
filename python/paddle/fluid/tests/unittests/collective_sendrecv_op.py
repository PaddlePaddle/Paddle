# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from test_collective_base import TestCollectiveRunnerBase, runtime_main

paddle.enable_static()


class TestCollectiveSendRecv(TestCollectiveRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        ring_id = self.global_ring_id
        with fluid.program_guard(main_prog, startup_program):
            tindata = layers.data(
                name="tindata", shape=[10, 1000], dtype='float64')
            if self.rank == 0:
                main_prog.global_block().append_op(
                    type="send_v2",
                    inputs={'X': tindata},
                    attrs={
                        'ring_id': ring_id,
                        'peer': 1,
                        'use_calc_stream': True
                    })
            else:
                main_prog.global_block().append_op(
                    type="recv_v2",
                    outputs={'Out': tindata},
                    attrs={
                        'peer': 0,
                        'ring_id': ring_id,
                        'dtype': tindata.dtype,
                        'out_shape': tindata.shape,
                        'use_calc_stream': True,
                    })
            return tindata


if __name__ == "__main__":
    runtime_main(TestCollectiveSendRecv, "sendrecv", 0)
