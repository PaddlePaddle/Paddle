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

    def set_feed(self):
        data = np.random.uniform(size=[5, 4, 2, 3])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        self.attrs = {"pad": [1, 2, 3, 4]}

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
        pad = paddle.nn.functional.pad(x, **self.attrs)
        self.fetch_list = [pad.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


@unittest.skip("Do not support `pad` as a tensor")
class TestCase1(TestBase):
    def set_op_attrs(self):
        self.attrs = {}

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
        const_attrs = {
            'name': 'y',
            'shape': [4],
            'dtype': 'int32',
            'value': 2,
        }
        y = paddle.fluid.layers.fill_constant(**const_attrs)
        pad = paddle.nn.functional.pad(x, pad=y)
        self.fetch_list = [pad.name]


class TestCase2(TestBase):
    def set_op_attrs(self):
        self.attrs = {"pad": [2, 5], "data_format": "NCL"}

    def set_feed(self):
        data = np.random.uniform(size=[4, 2, 3])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())


class TestCase3(TestBase):
    def set_op_attrs(self):
        self.attrs = {"pad": [2, 5, 2, 3, 6, 3], "data_format": "NCDHW"}

    def set_feed(self):
        data = np.random.uniform(size=[2, 3, 4, 2, 3])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())


class TestCase4(TestBase):
    def set_op_attrs(self):
        self.attrs = {"pad": [2, 2, 1, 1], "mode": "reflect"}


@unittest.skip("replicate mode is not supported")
class TestCase5(TestBase):
    def set_op_attrs(self):
        self.attrs = {"pad": [1, 2, 3, 4], "mode": "replicate"}


@unittest.skip("circular mode is not supported")
class TestCase6(TestBase):
    def set_op_attrs(self):
        self.attrs = {"pad": [1, 2, 3, 4], "mode": "circular"}


@unittest.skip("Only support NCL, NCHW, NCDHW")
class TestCase7(TestBase):
    def set_op_attrs(self):
        self.attrs = {"pad": [1, 2], "data_format": "NLC"}


@unittest.skip("Only support NCL, NCHW, NCDHW")
class TestCase8(TestBase):
    def set_op_attrs(self):
        self.attrs = {"pad": [1, 2, 3, 4], "data_format": "NHWC"}


@unittest.skip("Only support NCL, NCHW, NCDHW")
class TestCase9(TestBase):
    def set_op_attrs(self):
        self.attrs = {"pad": [1, 2, 3, 4, 1, 3], "data_format": "NDHWC"}


if __name__ == "__main__":
    unittest.main()
