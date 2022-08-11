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
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    @property
    def fp16_enabled(self):
        return False

    def set_data_feed(self):
        data = np.random.uniform(size=[1, 3, 3, 3])
        self.feed_fp32 = {'x': data.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float32'

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype=self.feed_dtype[0])
        out = paddle.cast(x, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


class TestEnableFp16(TestBase):

    @property
    def fp16_enabled(self):
        return True

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def set_data_feed(self):
        data = np.random.uniform(size=[1, 3, 3, 3])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float32'


class TestCase2(TestBase):

    def set_atol(self):
        super().set_atol()
        self.atol = 1e-3
        self.rtol = 1e-3

    def set_data_feed(self):
        self.feed_fp32 = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('float32'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float16'


class TestCase3(TestBase):

    def set_data_feed(self):
        self.feed_fp32 = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('float32'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'int32'


class TestCase4(TestBase):

    def set_data_feed(self):
        self.feed_fp32 = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('int32'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float32'


class TestCase5(TestBase):

    def set_data_feed(self):
        self.feed_fp32 = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('float16'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'int32'


class TestCase6(TestBase):

    def set_data_feed(self):
        self.feed_fp32 = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('int32'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float16'


@unittest.skip('float64 is not supported')
class TestCase7(TestBase):

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float64'


@unittest.skip('skip float16 to float32')
class TestCase8(TestBase):

    def set_data_feed(self):
        self.feed_fp32 = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('float16'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float32'


@unittest.skip('int32 to int8 is not supported')
class TestCase9(TestBase):

    def set_atol(self):
        super().set_atol()
        self.atol = 1

    def set_data_feed(self):
        self.feed_fp32 = {
            "x":
            np.random.randint(low=1, high=100, size=[1, 3, 3,
                                                     3]).astype('int32'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'int8'


if __name__ == "__main__":
    unittest.main()
