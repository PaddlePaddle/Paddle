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


class TestLogicalAnd(IPUOpTest):

    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_test_op()

    @property
    def fp16_enabled(self):
        return False

    def set_test_op(self):
        self.op = paddle.fluid.layers.logical_and

    def set_op_attrs(self):
        self.attrs = {}

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype=self.feed_dtype[0])
        y = paddle.static.data(name=self.feed_list[1],
                               shape=self.feed_shape[1],
                               dtype=self.feed_dtype[1])
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
        self.feed_dtype = ['bool', 'bool']

    def set_data_feed0(self):
        x = np.random.choice([True, False], size=(1, 3, 5, 5))
        y = np.random.choice([True, False], size=(1, 3, 5, 5))
        self.feed_fp32 = {
            "x": x.astype('bool'),
            "y": y.astype('bool'),
        }
        self.set_feed_attr()

    def test_case0(self):
        self.set_data_feed0()
        self.set_op_attrs()
        self.run_test_base()


class TestLogicalOr(TestLogicalAnd):

    def set_test_op(self):
        self.op = paddle.fluid.layers.logical_or


if __name__ == "__main__":
    unittest.main()
