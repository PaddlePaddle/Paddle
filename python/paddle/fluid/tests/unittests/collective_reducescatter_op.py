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


class TestCollectiveReduceScatter(TestCollectiveRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        ring_id = 0
        nranks = 2
        with fluid.program_guard(main_prog, startup_program):
            tindata = layers.data(
                name="tindata", shape=[10, 1000], dtype='float32')
            toutdata = main_prog.current_block().create_var(
                name="outofrs",
                dtype='float32',
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False)
            main_prog.global_block().append_op(
                type="c_reducescatter",
                inputs={'X': tindata},
                attrs={'ring_id': ring_id,
                       'nranks': nranks},
                outputs={'Out': toutdata})
            main_prog.global_block().append_op(
                type="c_sync_comm_stream",
                inputs={'X': toutdata},
                outputs={'Out': toutdata},
                attrs={'ring_id': ring_id})
            return toutdata


if __name__ == "__main__":
    runtime_main(TestCollectiveReduceScatter, "reduce_scatter", 0)
