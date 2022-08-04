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
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_data_feed(self):
        data = np.random.uniform(size=[3, 4, 5, 6])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}
        self.feed_fp16 = {"in_0": data.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        self.attrs = {
            "axes": [1, 2, 3],
            "starts": [-3, 0, 2],
            "ends": [3, 2, 4],
            "strides": [1, 1, 1]
        }

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')
        out = paddle.fluid.layers.strided_slice(x, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


class TestCase1(TestBase):

    def set_data_feed(self):
        data = np.random.uniform(size=[2, 4])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}
        self.feed_fp16 = {"in_0": data.astype(np.float16)}

    def set_op_attrs(self):
        self.attrs = {
            "axes": [0, 1],
            "starts": [1, 3],
            "ends": [2, 0],
            "strides": [1, -1]
        }


@unittest.skip('Only strides of 1 or -1 are supported.')
class TestCase2(TestBase):

    def set_data_feed(self):
        data = np.random.uniform(size=[2, 4])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}
        self.feed_fp16 = {"in_0": data.astype(np.float16)}

    def set_op_attrs(self):
        self.attrs = {
            "axes": [0, 1],
            "starts": [1, 3],
            "ends": [-1, 1000],
            "strides": [1, 3]
        }


@unittest.skip('dynamic graph is not support on IPU')
class TestCase3(TestBase):

    def set_data_feed(self):
        x = np.random.uniform(size=[4, 5, 6])
        s = np.array([0, 0, 2])
        e = np.array([3, 2, 4])
        self.feed_fp32 = {
            "x": x.astype(np.float32),
            "starts": s.astype(np.int32),
            "ends": e.astype(np.int32)
        }
        self.feed_fp16 = {
            "x": x.astype(np.float16),
            "starts": s.astype(np.int32),
            "ends": e.astype(np.int32)
        }

    def set_op_attrs(self):
        self.attrs = {"strides": [1, 1, 1], "axes": [0, 1, 2]}

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')
        starts = paddle.static.data(name=self.feed_list[1],
                                    shape=self.feed_shape[1],
                                    dtype='int32')
        ends = paddle.static.data(name=self.feed_list[2],
                                  shape=self.feed_shape[2],
                                  dtype='int32')
        out = paddle.fluid.layers.strided_slice(x,
                                                starts=starts,
                                                ends=ends,
                                                **self.attrs)
        self.fetch_list = [out.name]


if __name__ == "__main__":
    unittest.main()
