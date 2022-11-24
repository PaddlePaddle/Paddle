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

import numpy as np
import argparse
import os
import sys
import signal
import time
from contextlib import closing
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
from test_collective_base_npu import TestCollectiveRunnerBase, runtime_main

paddle.enable_static()


class TestCollectiveIdentity(TestCollectiveRunnerBase):

    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        ring_id = 0
        nranks = 2
        with fluid.program_guard(main_prog, startup_program):
<<<<<<< HEAD
            tindata = layers.data(name="tindata",
                                  shape=[10, 1000],
                                  dtype='float32')
=======
            tindata = layers.data(
                name="tindata", shape=[10, 1000], dtype='float32'
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            toutdata = main_prog.current_block().create_var(
                name="outofgather",
                dtype='float32',
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
<<<<<<< HEAD
                stop_gradient=False)
            main_prog.global_block().append_op(type="c_identity",
                                               inputs={'X': tindata},
                                               outputs={'Out': toutdata},
                                               attrs={
                                                   'ring_id': ring_id,
                                                   'nranks': nranks
                                               })
=======
                stop_gradient=False,
            )
            main_prog.global_block().append_op(
                type="c_identity",
                inputs={'X': tindata},
                outputs={'Out': toutdata},
                attrs={'ring_id': ring_id, 'nranks': nranks},
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            return toutdata


if __name__ == "__main__":
    runtime_main(TestCollectiveIdentity, "identity", 0)
