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
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def setUp(self):
        self.set_atol()
        self.set_test_op()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()

    def set_test_op(self):
<<<<<<< HEAD
        self.op = paddle.fluid.layers.abs
=======
        self.op = paddle.abs
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
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
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')
=======
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
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
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    @property
    def fp16_enabled(self):
        return False

    def set_atol(self):
        super().set_atol()
        self.atol = 1e-6

    def set_test_op(self):
<<<<<<< HEAD
        self.op = paddle.fluid.layers.acos
=======
        self.op = paddle.acos
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestAsin(TestAcos):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.asin
=======
    def set_test_op(self):
        self.op = paddle.asin
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestSinh(TestAcos):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.sinh
=======
    def set_test_op(self):
        self.op = paddle.sinh
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestAtan(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.atan
=======
    def set_test_op(self):
        self.op = paddle.atan
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestCeil(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.ceil
=======
    def set_test_op(self):
        self.op = paddle.ceil
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestCos(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.cos
=======
    def set_test_op(self):
        self.op = paddle.cos
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestCosh(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.cosh
=======
    def set_test_op(self):
        self.op = paddle.cosh
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestErf(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.erf
=======
    def set_test_op(self):
        self.op = paddle.erf
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestExp(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.exp
=======
    def set_test_op(self):
        self.op = paddle.exp
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestFloor(TestBase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    @property
    def fp16_enabled(self):
        return False

    def set_test_op(self):
<<<<<<< HEAD
        self.op = paddle.fluid.layers.floor
=======
        self.op = paddle.floor
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestLog(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.log
=======
    def set_test_op(self):
        self.op = paddle.log
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestReciprocal(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.reciprocal
=======
    def set_test_op(self):
        self.op = paddle.reciprocal
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestRelu(TestBase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_test_op(self):
        self.op = paddle.fluid.layers.relu
        self.op_attrs = {}


class TestRound(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.round
=======
    def set_test_op(self):
        self.op = paddle.round
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestSigmoid(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.sigmoid
=======
    def set_test_op(self):
        self.op = paddle.nn.functional.sigmoid
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestSign(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.sign
=======
    def set_test_op(self):
        self.op = paddle.sign
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestSin(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.sin
=======
    def set_test_op(self):
        self.op = paddle.sin
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestSoftplus(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.softplus
=======
    def set_test_op(self):
        self.op = paddle.nn.functional.softplus
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestSoftsign(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.softsign
=======
    def set_test_op(self):
        self.op = paddle.nn.functional.softsign
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestSqrt(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.sqrt
=======
    def set_test_op(self):
        self.op = paddle.sqrt
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestTan(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.tan
=======
    def set_test_op(self):
        self.op = paddle.tan
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


class TestTanh(TestBase):
<<<<<<< HEAD

    def set_test_op(self):
        self.op = paddle.fluid.layers.tanh
=======
    def set_test_op(self):
        self.op = paddle.tanh
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.op_attrs = {}


if __name__ == "__main__":
    unittest.main()
