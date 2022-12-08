# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

sys.path.append("..")
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
from test_sync_batch_norm_base_mlu import (
    TestSyncBatchNormRunnerBase,
    runtime_main,
)
from op_test import OpTest, _set_use_system_allocator

from test_sync_batch_norm_op import create_or_get_tensor

_set_use_system_allocator(False)
paddle.enable_static()


class TestSyncBatchNormOpTraining(TestSyncBatchNormRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

        self.dtype = np.float32
        self.bn_dtype = np.float32
        self.N = 8
        self.C = 16
        self.H = 32
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]
        self.atol = 1e-3

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
        """Build program."""
        use_cudnn = False
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                data = fluid.layers.data(
                    name='input',
                    shape=self.dshape,
                    dtype=self.dtype,
                    append_batch_size=False,
                )
                conv = fluid.layers.conv2d(
                    input=data,
                    num_filters=32,
                    filter_size=1,
                    param_attr=fluid.ParamAttr(name='conv2d_weight'),
                    bias_attr=False,
                    use_cudnn=use_cudnn,
                )
                if self.bn_dtype == np.float16:
                    conv = fluid.layers.cast(conv, 'float16')
                bn = paddle.static.nn.batch_norm(
                    conv,
                    param_attr=fluid.ParamAttr(name='bn_scale'),
                    bias_attr=fluid.ParamAttr(name='bn_bias'),
                    moving_mean_name='bn_moving_mean',
                    moving_variance_name='bn_moving_variance',
                    data_layout=layout,
                    is_test=only_forward,
                )
                if self.bn_dtype == np.float16:
                    bn = fluid.layers.cast(bn, 'float32')
                sigmoid = paddle.nn.functional.sigmoid(bn)
                out = paddle.sum(sigmoid)
                # if not sync_bn:
                #     out = out / core.get_mlu_device_count()
                if not only_forward:
                    sgd_opt = fluid.optimizer.SGD(learning_rate=0.0)
                    sgd_opt.backward(out)
        return [out, conv, bn]


if __name__ == "__main__":
    # print('sync_batch_norm_op_mlu.py __main__')

    runtime_main(TestSyncBatchNormOpTraining, "identity", 0)
