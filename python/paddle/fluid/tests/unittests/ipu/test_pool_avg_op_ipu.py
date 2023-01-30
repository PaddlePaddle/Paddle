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
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest


class TestBase(IPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_data_feed(self):
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        self.attrs = {
<<<<<<< HEAD
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
=======
            "pool_size": 3,
            "pool_type": 'avg',
            "pool_stride": 1,
            "pool_padding": 0,
            "global_pooling": False,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            "ceil_mode": False,
            "exclusive": True,
            "data_format": 'NCHW',
        }

    @IPUOpTest.static_graph
    def build_model(self):
<<<<<<< HEAD
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
        out = paddle.nn.functional.avg_pool2d(x, **self.attrs)
=======
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')
        out = paddle.fluid.layers.pool2d(x, **self.attrs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
    def set_attrs(self):
        super().set_attrs()
        self.attrs['kernel_size'] = 3


class TestCase1_2(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['kernel_size'] = [3, 1]


class TestCase2(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['stride'] = 2


class TestCase2_2(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['stride'] = [2, 1]


class TestCase3(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['padding'] = [1, 1]


class TestCase3_2(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['padding'] = [1, 1, 2, 2]
=======

    def set_attrs(self):
        super().set_attrs()
        self.attrs['pool_size'] = 3


class TestCase1_2(TestBase):

    def set_attrs(self):
        super().set_attrs()
        self.attrs['pool_size'] = [3, 1]


class TestCase2(TestBase):

    def set_attrs(self):
        super().set_attrs()
        self.attrs['pool_stride'] = 2


class TestCase2_2(TestBase):

    def set_attrs(self):
        super().set_attrs()
        self.attrs['pool_stride'] = [2, 1]


class TestCase3(TestBase):

    def set_attrs(self):
        super().set_attrs()
        self.attrs['pool_padding'] = [1, 1]


class TestCase3_2(TestBase):

    def set_attrs(self):
        super().set_attrs()
        self.attrs['pool_padding'] = [1, 1, 2, 2]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


@unittest.skip('the results has a positional offset')
class TestCase3_3(TestBase):
<<<<<<< HEAD
    def set_attrs(self):
        super().set_attrs()
        self.attrs['padding'] = [1, 2, 1, 1]
=======

    def set_attrs(self):
        super().set_attrs()
        self.attrs['pool_padding'] = [1, 2, 1, 1]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


@unittest.skip('paddle output has nan')
class TestCase3_4(TestBase):
<<<<<<< HEAD
    def set_attrs(self):
        super().set_attrs()
        self.attrs['size'] = 1
        self.attrs['padding'] = 1


class TestCase5(TestBase):
=======

    def set_attrs(self):
        super().set_attrs()
        self.attrs['pool_size'] = 1
        self.attrs['pool_padding'] = 1


class TestCase4(TestBase):

    def set_attrs(self):
        super().set_attrs()
        self.attrs['global_pooling'] = True


class TestCase5(TestBase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_attrs(self):
        super().set_attrs()
        self.attrs['ceil_mode'] = True


class TestCase6(TestBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_attrs(self):
        super().set_attrs()
        self.attrs['exclusive'] = False


<<<<<<< HEAD
=======
class TestAdaptive(TestBase):

    def set_op_attrs(self):
        self.attrs = {
            "pool_size": 1,
            "pool_type": 'avg',
            "require_index": False
        }

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')
        out = paddle.fluid.layers.adaptive_pool2d(x, **self.attrs)
        self.fetch_list = [out.name]


>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
if __name__ == "__main__":
    unittest.main()
