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


class TestMean(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_test_op()

    def set_test_op(self):
        self.op = paddle.mean

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
        out = self.op(x, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def run_test_base(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

    def set_data_feed0(self):
        data = np.random.uniform(size=[2, 4])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}
        self.feed_fp16 = {"in_0": data.astype(np.float16)}
        self.set_feed_attr()

    def set_data_feed1(self):
        data = np.random.uniform(size=[2, 2, 2])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}
        self.feed_fp16 = {"in_0": data.astype(np.float16)}
        self.set_feed_attr()

    def set_op_attr0(self):
        self.attrs = {}
        self.attrs['dim'] = None
        self.attrs['keep_dim'] = False

    def test_case0(self):
        self.set_data_feed0()
        self.set_op_attr0()
        self.run_test_base()

    def test_case1(self):
        self.set_data_feed0()
        self.set_op_attr0()
        self.attrs['dim'] = 0
        self.run_test_base()

    def test_case2(self):
        self.set_data_feed0()
        self.set_op_attr0()
        self.attrs['dim'] = -1
        self.run_test_base()

    def test_case3(self):
        self.set_data_feed0()
        self.set_op_attr0()
        self.attrs['dim'] = 1
        self.run_test_base()

    def test_case4(self):
        self.set_data_feed0()
        self.attrs = {}
        self.attrs['dim'] = 1
        self.attrs['keep_dim'] = True
        self.run_test_base()

    def test_case5(self):
        self.set_data_feed1()
        self.attrs = {}
        self.attrs['dim'] = [1, 2]
        self.attrs['keep_dim'] = False
        self.run_test_base()

    def test_case6(self):
        self.set_data_feed1()
        self.attrs = {}
        self.attrs['dim'] = [0, 1]
        self.attrs['keep_dim'] = False
        self.run_test_base()

    def test_case7(self):
        self.set_data_feed1()
        self.attrs = {}
        self.attrs['dim'] = [0, 1]
        self.attrs['keep_dim'] = True
        self.run_test_base()


class TestMax(TestMean):
    def set_test_op(self):
        self.op = paddle.max


class TestMin(TestMean):
    def set_test_op(self):
        self.op = paddle.min


class TestSum(TestMean):
    def set_test_op(self):
        self.op = paddle.paddle.sum


class TestLogsumexp(TestMean):
    def set_test_op(self):
        self.op = paddle.logsumexp

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
        if 'dim' in self.attrs:
            self.attrs['axis'] = self.attrs['dim']
            del self.attrs['dim']
        if 'keep_dim' in self.attrs:
            self.attrs['keepdim'] = self.attrs['keep_dim']
            del self.attrs['keep_dim']
        out = self.op(x, **self.attrs)
        self.fetch_list = [out.name]


class TestAll(TestMean):
    @property
    def fp16_enabled(self):
        return False

    def set_data_feed0(self):
        data = np.random.choice(a=[False, True], size=(2, 4))
        self.feed_fp32 = {"in_0": data.astype(bool)}
        self.set_feed_attr()

    def set_data_feed1(self):
        data = np.random.choice(a=[False, True], size=(2, 2, 2))
        self.feed_fp32 = {"in_0": data.astype(bool)}
        self.set_feed_attr()

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='bool'
        )
        out = self.op(x, **self.attrs)
        self.fetch_list = [out.name]

    def set_test_op(self):
        self.op = paddle.all


class TestAny(TestAll):
    def set_test_op(self):
        self.op = paddle.any


if __name__ == "__main__":
    unittest.main()
