#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class TestMul(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_test_op()

    @property
    def fp16_enabled(self):
        if IPUOpTest.use_ipumodel():
            return False
        else:
            return True

    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_mul

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
        y = paddle.static.data(
            name=self.feed_list[1], shape=self.feed_shape[1], dtype='float32'
        )
        out = self.op(x, y, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def run_test_base(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

    def test_case0(self):
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(2, 3, 4, 5))

        self.feed_fp32 = {
            "x": data_x.astype('float32'),
            "y": data_y.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data_x.astype('float16'),
            "y": data_y.astype('float16'),
        }
        self.attrs = {}
        self.set_feed_attr()
        self.run_test_base()

    def test_case1(self):
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(3, 4))
        self.feed_fp32 = {
            "x": data_x.astype('float32'),
            "y": data_y.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data_x.astype('float16'),
            "y": data_y.astype('float16'),
        }
        self.set_feed_attr()
        self.attrs = {"axis": 1}
        self.run_test_base()

    def test_case2(self):
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(5))
        self.feed_fp32 = {
            "x": data_x.astype('float32'),
            "y": data_y.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data_x.astype('float16'),
            "y": data_y.astype('float16'),
        }
        self.set_feed_attr()
        self.attrs = {"axis": -1}
        self.run_test_base()

    def test_case3(self):
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(2))
        self.feed_fp32 = {
            "x": data_x.astype('float32'),
            "y": data_y.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data_x.astype('float16'),
            "y": data_y.astype('float16'),
        }
        self.set_feed_attr()
        self.attrs = {"axis": 0}
        self.run_test_base()


class TestAdd(TestMul):
    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_add


class TestSub(TestMul):
    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_sub


class TestDiv(TestMul):
    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_div


class TestMin(TestMul):
    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_min


class TestMax(TestMul):
    def set_test_op(self):
        self.op = paddle.maximum


class TestPow(TestMul):
    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_pow


class TestMod(TestMul):
    def set_atol(self):
        self.atol = 1e-7
        self.rtol = 1e-5
        self.atol_fp16 = 1e-2
        self.rtol_fp16 = 1e-3

    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_mod


if __name__ == "__main__":
    unittest.main()
