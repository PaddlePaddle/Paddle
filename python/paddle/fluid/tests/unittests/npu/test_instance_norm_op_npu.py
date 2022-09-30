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
import unittest
import sys

sys.path.append("..")
from op_test import OpTest

import paddle
from paddle import fluid
from paddle.static import Program, program_guard
from paddle.fluid import core
from paddle.fluid.op import Operator
from paddle.fluid.dygraph import to_variable
from paddle.fluid.framework import _test_eager_guard

paddle.enable_static()


class TestInstanceNorm(unittest.TestCase):

    def test_dygraph(self):
        places = [fluid.NPUPlace(0)]
        for p in places:
            shape = [4, 10, 4, 4]

            def compute_v1(x):
                with fluid.dygraph.guard(p):
                    bn = fluid.dygraph.InstanceNorm(shape[1])
                    y = bn(fluid.dygraph.to_variable(x))
                return y.numpy()

            def compute_v2(x):
                with fluid.dygraph.guard(p):
                    bn = paddle.nn.InstanceNorm2D(shape[1])
                    y = bn(fluid.dygraph.to_variable(x))
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-03)

    def test_static(self):
        places = [fluid.NPUPlace(0)]
        for p in places:
            exe = fluid.Executor(p)
            shape = [4, 10, 16, 16]

            def compute_v1(x_np):
                with program_guard(Program(), Program()):
                    ins = fluid.dygraph.InstanceNorm(shape[1])
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = ins(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            def compute_v2(x_np):
                with program_guard(Program(), Program()):
                    ins = paddle.nn.InstanceNorm2D(shape[1])
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = ins(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-03)


if __name__ == '__main__':
    unittest.main()
