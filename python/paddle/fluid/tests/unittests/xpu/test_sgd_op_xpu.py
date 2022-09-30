#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import sys
import os

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.op import Operator

from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper


class XPUTestSgdOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'sgd'
        self.use_dynamic_create_class = False

    class TestSGDOp(XPUOpTest):

        def setUp(self):
            self.op_type = "sgd"
            self.dtype = self.in_type
            self.conf()
            w = np.random.random((self.h, self.w)).astype(self.dtype)
            g = np.random.random((self.h, self.w)).astype(self.dtype)
            lr = np.array([0.1]).astype(self.dtype)

            self.inputs = {'Param': w, 'Grad': g, 'LearningRate': lr}
            self.outputs = {'ParamOut': w - lr * g}

        def conf(self):
            self.h = 102
            self.w = 105

        def test_check_output_with_place(self):
            self.check_output_with_place(paddle.XPUPlace(0))

    class TestSGDOpCase8X(TestSGDOp):

        def conf(self):
            self.h = 10
            self.w = 64


support_types = get_xpu_op_support_types('sgd')
for stype in support_types:
    create_test_class(globals(), XPUTestSgdOp, stype)


class TestSGDOpWithLargeInput(unittest.TestCase):

    def runTest(self):
        data = fluid.layers.fill_constant(shape=[1], value=128, dtype='int64')
        label = fluid.layers.fill_constant(shape=[1, 150],
                                           value=0.5,
                                           dtype='float32')
        emb = fluid.embedding(input=data, size=(10000, 150), dtype='float32')
        out = fluid.layers.l2_normalize(x=emb, axis=-1)

        cost = fluid.layers.square_error_cost(input=out, label=label)
        avg_cost = paddle.mean(cost)
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        sgd_optimizer.minimize(avg_cost)

        place = paddle.XPUPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        result = exe.run(fluid.default_main_program(), fetch_list=[avg_cost])


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
