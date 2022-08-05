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
        x = np.random.uniform(size=[1, 2, 6, 10])
        self.feed_fp32 = {"x": x.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs["size"] = [12, 12]

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype="float32")
        out = paddle.nn.functional.interpolate(x, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


class TestCase0(TestBase):

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs["size"] = [3, 4]


class TestCase1(TestBase):

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs["scale_factor"] = [2, 1]


@unittest.skip("Only one of size or scale_factor should be defined")
class TestCase2(TestBase):

    def set_op_attrs(self):
        self.attrs = {"size": [12, 12], "scale_factor": [2, 1]}


class TestCase3(TestBase):

    def set_op_attrs(self):
        self.attrs = {"scale_factor": 2.5}


class TestBilinear(TestBase):

    @property
    def fp16_enabled(self):
        return False

    def set_atol(self):
        self.atol = 1e-6
        self.rtol = 1e-6
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_op_attrs(self):
        self.attrs = {"size": [12, 12], "mode": "bilinear"}


# Take long time
class TestBicubic(TestBase):

    @property
    def fp16_enabled(self):
        return False

    def set_atol(self):
        self.atol = 1e-6
        self.rtol = 1e-6
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_op_attrs(self):
        self.attrs = {"size": [12, 12], "mode": "bicubic"}


# Trilinear requires 5-D input
class TestTrilinear(TestBase):

    @property
    def fp16_enabled(self):
        return False

    def set_atol(self):
        self.atol = 1e-6
        self.rtol = 1e-6
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_data_feed(self):
        x = np.random.uniform(size=[2, 3, 3, 6, 10])
        self.feed_fp32 = {"x": x.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16)}

    def set_op_attrs(self):
        self.attrs = {
            "size": [12, 12, 12],
            "mode": "trilinear",
            "data_format": "NCDHW"
        }


# Linear requires 3-D input
class TestLinear(TestBase):

    @property
    def fp16_enabled(self):
        return False

    def set_atol(self):
        self.atol = 1e-6
        self.rtol = 1e-6
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_data_feed(self):
        x = np.random.uniform(size=[3, 6, 10])
        self.feed_fp32 = {"x": x.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16)}

    def set_op_attrs(self):
        self.attrs = {"size": [12], "mode": "linear", "data_format": "NCW"}


@unittest.skip(
    "Transfer to Pool Op with 2-D ksize, now we only support 1-D ksize.")
class TestArea(TestBase):

    def set_data_feed(self):
        x = np.random.uniform(size=[2, 3, 6, 6])
        self.feed_fp32 = {"x": x.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16)}

    def set_op_attrs(self):
        self.attrs = {"size": 12, "mode": "area"}


# align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear
class TestAlignCorners(TestBase):

    @property
    def fp16_enabled(self):
        return False

    def set_op_attrs(self):
        self.attrs = {
            "size": [12, 12],
            "align_corners": True,
            "mode": "bilinear"
        }


#
class TestAlignMode(TestBase):

    def set_op_attrs(self):
        self.attrs = {"size": [12, 12], "align_mode": 1}


if __name__ == "__main__":
    unittest.main()
