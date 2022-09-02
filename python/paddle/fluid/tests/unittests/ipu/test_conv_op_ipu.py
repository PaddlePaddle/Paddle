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
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['num_filters'] = 3
        self.attrs['filter_size'] = 3
        self.attrs['stride'] = 1
        self.attrs['padding'] = 0
        self.attrs['dilation'] = 1
        self.attrs['groups'] = 1
        self.attrs['data_format'] = 'NCHW'

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')
        x = paddle.fluid.layers.conv2d(x, **self.attrs)
        self.fetch_list = [x.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


class TestCase1(TestBase):

    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['num_filters'] = 1


class TestCase2(TestBase):

    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['filter_size'] = [3, 3]


class TestCase2_1(TestBase):

    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['filter_size'] = [3, 2]


class TestCase3(TestBase):

    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['stride'] = [2, 3]


class TestCase4(TestBase):

    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['dilation'] = [2, 2]


class TestCase5(TestBase):
    # Depthwise conv2d
    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['groups'] = 3


class TestCase6(TestBase):

    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['padding'] = 2


class TestCase7(TestBase):

    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['padding'] = [2, 3]


class TestCase8(TestBase):

    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['padding'] = [1, 2, 2, 3]


# depthwise_conv2d Op
class TestCase9(TestBase):

    def set_feed(self):
        data = np.random.uniform(size=[1, 3, 10, 10])
        weight = np.random.uniform(size=[3, 1, 3, 3])
        self.feed_fp32 = {
            'in_0': data.astype(np.float32),
            'in_1': weight.astype(np.float32)
        }
        self.feed_fp16 = {
            'in_0': data.astype(np.float16),
            'in_1': weight.astype(np.float16)
        }
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['groups'] = 3

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')
        weight = paddle.static.data(name=self.feed_list[1],
                                    shape=self.feed_shape[1],
                                    dtype='float32')
        x = paddle.nn.functional.conv2d(x, weight, **self.attrs)
        self.fetch_list = [x.name]


if __name__ == "__main__":
    unittest.main()
