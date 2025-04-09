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

import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    check_run_big_shape_test,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestStackOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'stack'
        self.use_dynamic_create_class = False

    class TestStackOp(XPUOpTest):
        def initDefaultParameters(self):
            self.num_inputs = 4
            self.input_dim = (5, 6, 7)
            self.axis = 0

        def setUp(self):
            self.initDefaultParameters()
            self.initParameters()
            self.__class__.use_xpu = True
            self.__class__.op_type = 'stack'
            self.dtype = self.in_type
            self.x = []
            for i in range(self.num_inputs):
                if self.dtype == np.uint16:
                    data = np.random.random(size=self.input_dim).astype(
                        np.float32
                    )
                    self.x.append(convert_float_to_uint16(data))
                else:
                    self.x.append(
                        np.random.random(size=self.input_dim).astype(self.dtype)
                    )

            tmp = []
            x_names = self.get_x_names()
            for i in range(self.num_inputs):
                tmp.append((x_names[i], self.x[i]))

            self.inputs = {'X': tmp}
            self.outputs = {'Y': np.stack(self.x, axis=self.axis)}
            self.attrs = {'axis': self.axis}

        def initParameters(self):
            pass

        def get_x_names(self):
            x_names = []
            for i in range(self.num_inputs):
                x_names.append(f'x{i}')
            return x_names

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0))

        def test_check_grad(self):
            self.check_grad_with_place(
                paddle.XPUPlace(0), self.get_x_names(), 'Y'
            )

    class TestStackOp1(TestStackOp):
        def initParameters(self):
            self.num_inputs = 16

    class TestStackOp2(TestStackOp):
        def initParameters(self):
            self.num_inputs = 30

    class TestStackOp3(TestStackOp):
        def initParameters(self):
            self.axis = -1

    class TestStackOp4(TestStackOp):
        def initParameters(self):
            self.axis = -4

    class TestStackOp5(TestStackOp):
        def initParameters(self):
            self.axis = 1

    class TestStackOp6(TestStackOp):
        def initParameters(self):
            self.axis = 3

    class TestStackOp7(TestStackOp):
        def initParameters(self):
            self.num_inputs = 4
            self.input_dim = (5, 6, 7)
            self.axis = 0
            self.dtype = np.int64

    class TestStackOp8(TestStackOp):
        def initParameters(self):
            self.num_inputs = 4
            self.input_dim = (5, 6, 7)
            self.axis = 0
            self.dtype = np.int32

    @check_run_big_shape_test()
    class TestStackOpLargeShape1(TestStackOp):
        def initParameters(self):
            self.num_inputs = 5
            self.input_dim = (1, 8192, 64)
            self.axis = 2


class TestStackSkipScenarioDynamic(unittest.TestCase):
    def test_skip_scenario(self):
        paddle.disable_static()
        paddle.set_device("xpu")

        def print_hook(name):
            def hook(grad):
                temp = grad  # Nonsense, just do something with the input

            return hook

        # Build tensors: first 5 each row need grad, rest 15 are no-grad
        d = []
        for j in range(4):
            a = []
            for i in range(20):
                b = paddle.to_tensor([float(j * 20 + i)], dtype='float32')
                if i < 5:
                    b.stop_gradient = False
                    b.register_hook(print_hook(f'i_{i}_j_{j}'))
                else:
                    b.stop_gradient = True
                a.append(b)

            c = paddle.stack(a)  # shape=[20]
            d.append(c)

        e = paddle.concat(d, axis=-1)  # shape=[20,4]
        e.backward()
        paddle.enable_static()


class TestStackSkipScenarioDynamic2(unittest.TestCase):
    def test_skip_scenario_mixed_segments(self):
        """
        Scenario:
          - For each of 4 rows, we create 20 single-element tensors:
            * Indices [0..4]   : stop_gradient = True
            * Indices [5..9]   : stop_gradient = False
            * Indices [10..14] : stop_gradient = True
            * Indices [15..19] : stop_gradient = False
        """

        paddle.disable_static()
        paddle.set_device("xpu")

        def print_hook(name):
            def hook(grad):
                temp = grad  # Nonsense, just do something with the input

            return hook

        d = []
        for j in range(4):
            a = []
            for i in range(20):
                val = float(j * 20 + i)
                b = paddle.to_tensor([val], dtype='float32')

                # First 5 => no grad
                # Second 5 => grad
                # Third 5 => no grad
                # Fourth 5 => grad
                if (0 <= i < 5) or (10 <= i < 15):
                    b.stop_gradient = True
                else:
                    b.stop_gradient = False
                    b.register_hook(print_hook(f'i_{i}_j_{j}'))

                a.append(b)

            c = paddle.stack(a)  # shape=[20]
            d.append(c)

        e = paddle.concat(d, axis=-1)  # shape=[20,4]
        e.backward()
        paddle.enable_static()


support_types = get_xpu_op_support_types('stack')
for stype in support_types:
    create_test_class(globals(), XPUTestStackOp, stype)

if __name__ == "__main__":
    unittest.main()
