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

    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_feed()
        self.set_op_attrs()

    def set_atol(self):
        self.atol = 1e-6
        self.rtol = 1e-6
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_feed(self):
        data = np.random.uniform(size=[5, 5])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['min'] = 0.1
        self.attrs['max'] = 3.4

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')
        x = paddle.clip(x, **self.attrs)
        self.fetch_list = [x.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


class TestNoMin(TestBase):

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['max'] = 3.4


class TestNoMax(TestBase):

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['min'] = 0.1


class TestNoMinNoMax(TestBase):

    def set_op_attrs(self):
        self.attrs = {}


class TestMinMaxTensor(TestBase):

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')

        min = paddle.fluid.layers.fill_constant(name="min",
                                                shape=[1],
                                                dtype='float32',
                                                value=0.1)
        max = paddle.fluid.layers.fill_constant(name="max",
                                                shape=[1],
                                                dtype='float32',
                                                value=3.4)
        x = paddle.clip(x, min=min, max=max)
        self.fetch_list = [x.name]


class TestMinTensor(TestBase):

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')

        min = paddle.fluid.layers.fill_constant(name="min",
                                                shape=[1],
                                                dtype='float32',
                                                value=0.1)
        x = paddle.clip(x, min=min)
        self.fetch_list = [x.name]


class TestMaxTensor(TestBase):

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')

        max = paddle.fluid.layers.fill_constant(name="max",
                                                shape=[1],
                                                dtype='float32',
                                                value=3.4)
        x = paddle.clip(x, max=max)
        self.fetch_list = [x.name]


class TestCombine1(TestBase):

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')

        min = paddle.fluid.layers.fill_constant(name="min",
                                                shape=[1],
                                                dtype='float32',
                                                value=0.1)
        x = paddle.clip(x, min=min, max=3.4)
        self.fetch_list = [x.name]


class TestCombine2(TestBase):

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')

        max = paddle.fluid.layers.fill_constant(name="max",
                                                shape=[1],
                                                dtype='float32',
                                                value=3.4)
        x = paddle.clip(x, min=0.1, max=max)
        self.fetch_list = [x.name]


class TestIntInput(TestBase):

    def set_feed(self):
        data = np.random.uniform(size=[5, 5])
        self.feed_fp32 = {'x': data.astype(np.int32)}
        self.feed_fp16 = {'x': data.astype(np.int32)}
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='int32')

        x = paddle.clip(x, min=0.1, max=3.4)
        self.fetch_list = [x.name]


class TestIntMinMax(TestBase):

    def set_feed(self):
        data = np.random.uniform(size=[5, 5])
        self.feed_fp32 = {'x': data.astype(np.int32)}
        self.feed_fp16 = {'x': data.astype(np.int32)}
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='int32')
        min = paddle.fluid.layers.fill_constant(name="min",
                                                shape=[1],
                                                dtype='int32',
                                                value=1)
        max = paddle.fluid.layers.fill_constant(name="max",
                                                shape=[1],
                                                dtype='int32',
                                                value=3)
        x = paddle.clip(x, min=min, max=max)
        self.fetch_list = [x.name]


if __name__ == "__main__":
    unittest.main()
