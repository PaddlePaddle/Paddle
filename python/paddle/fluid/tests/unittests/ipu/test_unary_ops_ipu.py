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
import paddle
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest


class TestBase(IPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
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
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
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
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
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

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
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
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestAsin(TestAcos):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.asin
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.asin
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestSinh(TestAcos):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.sinh
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.sinh
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestAtan(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.atan
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.atan
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestCeil(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.ceil
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.ceil
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestCos(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.cos
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.cos
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestCosh(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.cosh
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.cosh
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestErf(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.erf
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.erf
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestExp(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.exp
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.exp
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestFloor(TestBase):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    @property
    def fp16_enabled(self):
        return False

    def set_test_op(self):
<<<<<<< HEAD
        self.op = paddle.floor
=======
        self.op = paddle.fluid.layers.floor
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestLog(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.log
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.log
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestReciprocal(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.reciprocal
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.reciprocal
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestRelu(TestBase):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
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
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestSigmoid(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.nn.functional.sigmoid
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.sigmoid
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestSign(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.sign
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.sign
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestSin(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.sin
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.sin
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestSoftplus(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.nn.functional.softplus
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.softplus
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestSoftsign(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.nn.functional.softsign
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.softsign
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestSqrt(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.sqrt
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.sqrt
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestTan(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.tan
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.tan
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


class TestTanh(TestBase):
<<<<<<< HEAD
    def set_test_op(self):
        self.op = paddle.tanh
=======

    def set_test_op(self):
        self.op = paddle.fluid.layers.tanh
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_attrs = {}


if __name__ == "__main__":
    unittest.main()
