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

    def set_data_feed(self):
        x = np.random.uniform(size=[20, 30])
        y = np.random.uniform(size=[30, 20])
        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        self.attrs = {
            "transpose_x": False,
            "transpose_y": False,
        }

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
        y = paddle.static.data(
            name=self.feed_list[1], shape=self.feed_shape[1], dtype='float32'
        )

        out = paddle.matmul(x, y, **self.attrs)
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
    def set_op_attrs(self):
        self.attrs = {
            "transpose_x": True,
            "transpose_y": True,
        }


class TestCase2(TestBase):
    def set_op_attrs(self):
        self.attrs = {
            "transpose_x": True,
            "transpose_y": True,
        }

    def set_atol(self):
        self.atol = 1e-10
        self.rtol = 1e-6
        self.atol_fp16 = 1e-2
        self.rtol_fp16 = 1e-3


class TestCase3(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[5, 4, 3, 2])
        y = np.random.uniform(size=[5, 4, 2, 3])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}


class TestCase4(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[4, 3, 2])
        y = np.random.uniform(size=[4, 2, 3])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}


class TestCase5(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[4, 2, 3])
        y = np.random.uniform(size=[3, 2])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}


class TestCase6(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[3])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": x.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": x.astype(np.float16)}


@unittest.skip("not supported")
class TestCase6_2(TestCase6):
    def set_data_feed(self):
        x = np.random.uniform(size=[3])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": x.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": x.astype(np.float16)}

    def set_op_attrs(self):
        self.attrs = {
            "transpose_x": True,
            "transpose_y": True,
        }


class TestCase7(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[1, 3, 4, 5])
        y = np.random.uniform(size=[1, 3, 4, 5])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}

    def set_op_attrs(self):
        self.attrs = {
            "transpose_x": False,
            "transpose_y": True,
        }


class TestCase8(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[3, 1])
        y = np.random.uniform(size=[1, 2])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}


@unittest.skip("not supported")
class TestCase8_2(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[3])
        y = np.random.uniform(size=[2])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}

    def set_op_attrs(self):
        self.attrs = {
            "transpose_x": True,
            "transpose_y": True,
        }


@unittest.skip("dim > 4 is not supported")
class TestCase9(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[6, 5, 4, 2, 3])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": x.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": x.astype(np.float16)}


class TestCase10(TestBase):
    def set_op_attrs(self):
        self.attrs = {
            "transpose_y": True,
        }

    def set_data_feed(self):
        x = np.random.uniform(size=[4, 2, 3])
        y = np.random.uniform(size=[2, 3])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}


class TestCase11(TestBase):
    def set_op_attrs(self):
        self.attrs = {
            "transpose_x": True,
        }

    def set_data_feed(self):
        x = np.random.uniform(size=[4, 3, 2])
        y = np.random.uniform(size=[3, 2])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}


if __name__ == "__main__":
    unittest.main()
