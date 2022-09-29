#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from op_test import OpTest
from paddle.fluid.framework import _test_eager_guard


class TestIdentityLossOp(OpTest):

    def setUp(self):
        self.max_relative_error = 0.006
        self.python_api = paddle.incubate.identity_loss

        self.inputs = {}
        self.initTestCase()
        self.dtype = np.float64

        self.op_type = "identity_loss"
        self.attrs = {}
        self.attrs['reduction'] = self.reduction

        input = np.random.random(self.shape).astype(self.dtype)

        self.inputs['X'] = input
        if self.reduction == 0:
            output = input.sum()
        elif self.reduction == 1:
            output = input.mean()
        else:
            output = input
        self.outputs = {'Out': output}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_eager=True)
        paddle.disable_static()

    def test_check_grad_normal(self):
        paddle.enable_static()
        self.check_grad(['X'], 'Out', check_eager=True)
        paddle.disable_static()

    def initTestCase(self):
        self.shape = (4, 10, 10)
        self.reduction = 0


class TestCase1(TestIdentityLossOp):

    def initTestCase(self):
        self.shape = (8, 16, 8)
        self.reduction = 0


class TestCase2(TestIdentityLossOp):

    def initTestCase(self):
        self.shape = (8, 16)
        self.reduction = 1


class TestCase3(TestIdentityLossOp):

    def initTestCase(self):
        self.shape = (4, 8, 16)
        self.reduction = 2


class TestIdentityLossFloat32(TestIdentityLossOp):

    def set_attrs(self):
        self.dtype = 'float32'


class TestIdentityLossOpError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            input_data = np.random.random((2, 4)).astype("float32")

            def test_int():
                paddle.incubate.identity_loss(x=input_data, reduction=3)

            self.assertRaises(Exception, test_int)

            def test_string():
                paddle.incubate.identity_loss(x=input_data,
                                              reduction="wrongkey")

            self.assertRaises(Exception, test_string)

            def test_dtype():
                x2 = fluid.layers.data(name='x2', shape=[1], dtype='int32')
                paddle.incubate.identity_loss(x=x2, reduction=1)

            self.assertRaises(TypeError, test_dtype)
        paddle.disable_static()


class TestIdentityLossAPI(unittest.TestCase):

    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.place = fluid.CPUPlace()

    def identity_loss_ref(self, input, reduction):
        if reduction == 0 or reduction == "sum":
            return input.sum()
        elif reduction == 1 or reduction == "mean":
            return input.mean()
        else:
            return input

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_shape)
            out1 = paddle.incubate.identity_loss(x)
            out2 = paddle.incubate.identity_loss(x, reduction=0)
            out3 = paddle.incubate.identity_loss(x, reduction=1)
            out4 = paddle.incubate.identity_loss(x, reduction=2)

            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x},
                          fetch_list=[out1, out2, out3, out4])
        ref = [
            self.identity_loss_ref(self.x, 2),
            self.identity_loss_ref(self.x, 0),
            self.identity_loss_ref(self.x, 1),
            self.identity_loss_ref(self.x, 2)
        ]
        for out, out_ref in zip(res, ref):
            np.testing.assert_allclose(out, out_ref, rtol=0.0001)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)

        def test_case(x, reduction):
            x_tensor = paddle.to_tensor(x)
            out = paddle.incubate.identity_loss(x_tensor, reduction)
            out_ref = self.identity_loss_ref(x, reduction)
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.0001)

        test_case(self.x, 0)
        test_case(self.x, 1)
        test_case(self.x, 2)
        test_case(self.x, "sum")
        test_case(self.x, "mean")
        test_case(self.x, "none")
        paddle.enable_static()

    def test_errors(self):
        paddle.disable_static()
        x = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        x = paddle.to_tensor(x)
        self.assertRaises(Exception, paddle.incubate.identity_loss, x, -1)
        self.assertRaises(Exception, paddle.incubate.identity_loss, x, 3)
        self.assertRaises(Exception, paddle.incubate.identity_loss, x,
                          "wrongkey")
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12], 'int32')
            self.assertRaises(TypeError, paddle.incubate.identity_loss, x)


if __name__ == '__main__':
    unittest.main()
