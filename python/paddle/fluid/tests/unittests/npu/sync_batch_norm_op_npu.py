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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import numpy as np
import argparse
import os
import sys

sys.path.append("..")
import signal
import time
from contextlib import closing
<<<<<<< HEAD
=======
from six import string_types
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
from test_sync_batch_norm_base_npu import (
    TestSyncBatchNormRunnerBase,
    runtime_main,
)
from paddle.fluid.tests.unittests.op_test import (
    OpTest,
    _set_use_system_allocator,
)

from paddle.fluid.tests.unittests.test_sync_batch_norm_op import (
    create_or_get_tensor,
)
=======
from test_sync_batch_norm_base_npu import TestSyncBatchNormRunnerBase, runtime_main
from paddle.fluid.tests.unittests.op_test import OpTest, _set_use_system_allocator

from paddle.fluid.tests.unittests.test_sync_batch_norm_op import create_or_get_tensor
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

_set_use_system_allocator(False)
paddle.enable_static()


class TestSyncBatchNormOpTraining(TestSyncBatchNormRunnerBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.global_ring_id = 0

        self.dtype = np.float32
        self.N = 8
        self.C = 16
        self.H = 32
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]
        self.atol = 1e-3

<<<<<<< HEAD
    def get_model(
        self,
        main,
        startup,
        place,
        layout,
        seed,
        sync_bn=False,
        only_forward=False,
    ):
=======
    def get_model(self,
                  main,
                  startup,
                  place,
                  layout,
                  seed,
                  sync_bn=False,
                  only_forward=False):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        """Build program."""
        use_cudnn = False
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
<<<<<<< HEAD
                data = paddle.static.data(
                    name='input',
                    shape=[-1] + self.dshape,
                    dtype=self.dtype,
                )
                conv = paddle.static.nn.conv2d(
=======
                data = fluid.layers.data(name='input',
                                         shape=self.dshape,
                                         dtype=self.dtype,
                                         append_batch_size=False)
                conv = fluid.layers.conv2d(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    input=data,
                    num_filters=32,
                    filter_size=1,
                    param_attr=fluid.ParamAttr(name='conv2d_weight'),
                    bias_attr=False,
<<<<<<< HEAD
                    use_cudnn=use_cudnn,
                )
                bn = paddle.static.nn.batch_norm(
=======
                    use_cudnn=use_cudnn)
                bn = fluid.layers.batch_norm(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    conv,
                    param_attr=fluid.ParamAttr(name='bn_scale'),
                    bias_attr=fluid.ParamAttr(name='bn_bias'),
                    moving_mean_name='bn_moving_mean',
                    moving_variance_name='bn_moving_variance',
                    data_layout=layout,
<<<<<<< HEAD
                    is_test=only_forward,
                )
                # if self.dtype == np.float16:
                #     bn = fluid.layers.cast(bn, 'float32')
                sigmoid = paddle.nn.functional.sigmoid(bn)
                out = paddle.sum(sigmoid)
=======
                    is_test=only_forward)
                # if self.dtype == np.float16:
                #     bn = fluid.layers.cast(bn, 'float32')
                sigmoid = fluid.layers.sigmoid(bn)
                out = fluid.layers.reduce_sum(sigmoid)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                # if not sync_bn:
                #     out = out / core.get_npu_device_count()
                if not only_forward:
                    sgd_opt = fluid.optimizer.SGD(learning_rate=0.0)
                    sgd_opt.backward(out)
        return [out, conv, bn]


if __name__ == "__main__":
    # print('sync_batch_norm_op_npu.py __main__')

    runtime_main(TestSyncBatchNormOpTraining, "identity", 0)
