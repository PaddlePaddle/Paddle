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


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
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
        data = np.random.uniform(size=[1, 3, 3, 3]).astype('float32')
        self.feed_fp32 = {"x": data.astype(np.float32)}
        self.feed_fp16 = {"x": data.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        self.attrs = {}

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype=self.feed_dtype[0])
        out = paddle.fluid.layers.conv2d(x, num_filters=3, filter_size=3)
        out = paddle.fluid.layers.Print(out, **self.attrs)

        if self.is_training:
            loss = paddle.mean(out)
            adam = paddle.optimizer.Adam(learning_rate=1e-2)
            adam.minimize(loss)
            self.fetch_list = [loss.name]
        else:
            self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)


class TestCase1(TestBase):

    def set_op_attrs(self):
        self.attrs = {"message": "input_data"}


class TestTrainCase1(TestBase):

    def set_op_attrs(self):
        # "forward" : print forward
        # "backward" : print forward and backward
        # "both": print forward and backward
        self.attrs = {"message": "input_data2", "print_phase": "both"}

    def set_training(self):
        self.is_training = True
        self.epoch = 2


@unittest.skip("attrs are not supported")
class TestCase2(TestBase):

    def set_op_attrs(self):
        self.attrs = {
            "first_n": 10,
            "summarize": 10,
            "print_tensor_name": True,
            "print_tensor_type": True,
            "print_tensor_shape": True,
            "print_tensor_layout": True,
            "print_tensor_lod": True
        }


if __name__ == "__main__":
    unittest.main()
