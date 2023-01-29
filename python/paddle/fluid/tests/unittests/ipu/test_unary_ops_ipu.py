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
import paddle.nn.functional as F
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest


class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_test_op()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()

    def set_test_op(self):
        self.op = paddle.abs
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
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
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
    @property
    def fp16_enabled(self):
        return False

    def set_atol(self):
        super().set_atol()
        self.atol = 1e-6

    def set_test_op(self):
        self.op = paddle.acos
        self.op_attrs = {}


class TestAsin(TestAcos):
    def set_test_op(self):
        self.op = paddle.asin
        self.op_attrs = {}


class TestSinh(TestAcos):
    def set_test_op(self):
        self.op = paddle.sinh
        self.op_attrs = {}


class TestAtan(TestBase):
    def set_test_op(self):
        self.op = paddle.atan
        self.op_attrs = {}


class TestCeil(TestBase):
    def set_test_op(self):
        self.op = paddle.ceil
        self.op_attrs = {}


class TestCos(TestBase):
    def set_test_op(self):
        self.op = paddle.cos
        self.op_attrs = {}


class TestCosh(TestBase):
    def set_test_op(self):
        self.op = paddle.cosh
        self.op_attrs = {}


class TestErf(TestBase):
    def set_test_op(self):
        self.op = paddle.erf
        self.op_attrs = {}


class TestExp(TestBase):
    def set_test_op(self):
        self.op = paddle.exp
        self.op_attrs = {}


class TestFloor(TestBase):
    @property
    def fp16_enabled(self):
        return False

    def set_test_op(self):
        self.op = paddle.floor
        self.op_attrs = {}


class TestLog(TestBase):
    def set_test_op(self):
        self.op = paddle.log
        self.op_attrs = {}


class TestReciprocal(TestBase):
    def set_test_op(self):
        self.op = paddle.reciprocal
        self.op_attrs = {}


class TestRelu(TestBase):
    def set_test_op(self):
        self.op = F.relu
        self.op_attrs = {}


class TestRound(TestBase):
    def set_test_op(self):
        self.op = paddle.round
        self.op_attrs = {}


class TestSigmoid(TestBase):
    def set_test_op(self):
        self.op = paddle.nn.functional.sigmoid
        self.op_attrs = {}


class TestSign(TestBase):
    def set_test_op(self):
        self.op = paddle.sign
        self.op_attrs = {}


class TestSin(TestBase):
    def set_test_op(self):
        self.op = paddle.sin
        self.op_attrs = {}


class TestSoftplus(TestBase):
    def set_test_op(self):
        self.op = paddle.nn.functional.softplus
        self.op_attrs = {}


class TestSoftsign(TestBase):
    def set_test_op(self):
        self.op = paddle.nn.functional.softsign
        self.op_attrs = {}


class TestSqrt(TestBase):
    def set_test_op(self):
        self.op = paddle.sqrt
        self.op_attrs = {}


class TestTan(TestBase):
    def set_test_op(self):
        self.op = paddle.tan
        self.op_attrs = {}


class TestTanh(TestBase):
    def set_test_op(self):
        self.op = paddle.tanh
        self.op_attrs = {}


if __name__ == "__main__":
    unittest.main()
