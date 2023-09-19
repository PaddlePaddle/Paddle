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
from op_test_ipu import IPUOpTest

import paddle
import paddle.static


class TestGreaterThan(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_test_op()

    def set_test_op(self):
        self.op = paddle.base.layers.greater_than

    def set_op_attrs(self):
        self.attrs = {}

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

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_data_feed0(self):
        x = np.random.randn(3, 4, 5)
        y = np.random.randn(3, 4, 5)
        self.feed_fp32 = {
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
        }
        self.feed_fp16 = {
            "x": x.astype(np.float16),
            "y": y.astype(np.float16),
        }
        self.set_feed_attr()

    def set_data_feed1(self):
        x = np.ones([1, 10])
        y = np.ones([10])
        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}
        self.set_feed_attr()

    def set_data_feed2(self):
        x = np.ones([1, 10])
        y = np.zeros([1, 10])
        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}
        self.set_feed_attr()

    def set_data_feed3(self):
        x = np.zeros([1, 10])
        y = np.ones([1, 10])
        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}
        self.set_feed_attr()

    def test_case0(self):
        self.set_data_feed0()
        self.set_op_attrs()
        self.run_test_base()

    def test_case1(self):
        self.set_data_feed1()
        self.set_op_attrs()
        self.run_test_base()

    def test_case2(self):
        self.set_data_feed2()
        self.set_op_attrs()
        self.run_test_base()

    def test_case3(self):
        self.set_data_feed3()
        self.set_op_attrs()
        self.run_test_base()


class TestLessThan(TestGreaterThan):
    def set_test_op(self):
        self.op = paddle.base.layers.less_than


class TestEqual(TestGreaterThan):
    def set_test_op(self):
        self.op = paddle.base.layers.equal


class TestGreaterEqual(TestGreaterThan):
    def set_test_op(self):
        self.op = paddle.base.layers.greater_equal


class TestLessEqual(TestGreaterThan):
    def set_test_op(self):
        self.op = paddle.base.layers.less_equal


if __name__ == "__main__":
    unittest.main()
