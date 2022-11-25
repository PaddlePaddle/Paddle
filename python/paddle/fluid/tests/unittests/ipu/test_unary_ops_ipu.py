#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
<<<<<<< HEAD

=======
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
import paddle
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest


class TestBase(IPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
    def setUp(self):
        self.set_atol()
        self.set_test_op()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()

    def set_test_op(self):
<<<<<<< HEAD
        self.op = paddle.abs
=======
        self.op = paddle.fluid.layers.abs
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}

    def set_data_feed(self):
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
<<<<<<< HEAD
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
=======
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        out = self.op(x, **self.op_attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


class TestAcos(TestBase):
<<<<<<< HEAD
=======

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
    @property
    def fp16_enabled(self):
        return False

    def set_atol(self):
        super().set_atol()
        self.atol = 1e-6

    def set_test_op(self):
<<<<<<< HEAD
        self.op = paddle.acos
=======
        self.op = paddle.fluid.layers.acos
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestAsin(TestAcos):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.asin
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.asin
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestSinh(TestAcos):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.sinh
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.sinh
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestAtan(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.atan
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.atan
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestCeil(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.ceil
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.ceil
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestCos(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.cos
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.cos
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestCosh(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.cosh
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.cosh
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestErf(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.erf
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.erf
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestExp(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.exp
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.exp
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestFloor(TestBase):
<<<<<<< HEAD
=======

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
    @property
    def fp16_enabled(self):
        return False

    def set_test_op(self):
<<<<<<< HEAD
        self.op = paddle.floor
=======
        self.op = paddle.fluid.layers.floor
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestLog(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.log
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.log
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestReciprocal(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.reciprocal
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.reciprocal
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestRelu(TestBase):
<<<<<<< HEAD
=======

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
    def set_test_op(self):
        self.op = paddle.fluid.layers.relu
        self.op_attrs = {}


class TestRound(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.round
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.round
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestSigmoid(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.nn.functional.sigmoid
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.sigmoid
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestSign(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.sign
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.sign
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestSin(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.sin
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.sin
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestSoftplus(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.nn.functional.softplus
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.softplus
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestSoftsign(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.nn.functional.softsign
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.softsign
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestSqrt(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.sqrt
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.sqrt
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestTan(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.tan
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.tan
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


class TestTanh(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.tanh
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.tanh
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        self.op_attrs = {}


if __name__ == "__main__":
    unittest.main()
