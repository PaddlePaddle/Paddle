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
        self.set_feed_attr()
        self.set_op_attrs()

    def set_atol(self):
        self.atol = 1e-6
        self.rtol = 1e-6
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_feed(self):
        data1 = np.random.uniform(size=[10])
        data2 = np.random.uniform(size=[20])
        self.feed_fp32 = {
            'x': data1.astype(np.float32),
            'y': data2.astype(np.float32)
        }
        self.feed_fp16 = {
            'x': data1.astype(np.float16),
            'y': data2.astype(np.float16)
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['axis'] = [0, 1]

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype=self.feed_dtype[0])
        y = paddle.static.data(name=self.feed_list[1],
                               shape=self.feed_shape[1],
                               dtype=self.feed_dtype[1])
        r1, r2 = paddle.meshgrid(x, y)
        self.fetch_list = [r1.name, r2.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        for k, v in self.output_dict.items():
            self.output_dict[k] = np.concatenate([vv.flatten() for vv in v])
        self.check()


class TestCase1(TestBase):

    def set_feed(self):
        data1 = np.random.uniform(size=[10])
        data2 = np.random.uniform(size=[20])
        data3 = np.random.uniform(size=[30])
        self.feed_fp32 = {
            'x': data1.astype(np.float32),
            'y': data2.astype(np.float32),
            'z': data3.astype(np.float32)
        }
        self.feed_fp16 = {
            'x': data1.astype(np.float16),
            'y': data2.astype(np.float16),
            'z': data3.astype(np.float16)
        }

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype=self.feed_dtype[0])
        y = paddle.static.data(name=self.feed_list[1],
                               shape=self.feed_shape[1],
                               dtype=self.feed_dtype[1])
        z = paddle.static.data(name=self.feed_list[2],
                               shape=self.feed_shape[2],
                               dtype=self.feed_dtype[2])
        r1, r2, r3 = paddle.meshgrid(x, y, z)
        self.fetch_list = [r1.name, r2.name, r3.name]


class TestCase2(TestBase):

    def set_feed(self):
        data1 = np.random.uniform(size=[100])
        data2 = np.random.uniform(size=[200])
        self.feed_fp32 = {
            'x': data1.astype(np.int32),
            'y': data2.astype(np.int32)
        }
        self.feed_fp16 = {
            'x': data1.astype(np.int32),
            'y': data2.astype(np.int32)
        }


if __name__ == "__main__":
    unittest.main()
