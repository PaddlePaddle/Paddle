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
from op_test_ipu import IPUOpTest

import paddle
import paddle.nn.functional as F
import paddle.static


class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_atol(self):
        self.atol = 5e-6
        self.rtol = 1e-5
        self.atol_fp16 = 1e-2
        self.rtol_fp16 = 1e-3

    def set_data_feed(self):
        np_data = np.random.uniform(low=-1, high=1, size=[1, 3, 100, 100])
        self.feed_fp32 = {"x": np_data.astype('float32')}
        self.feed_fp16 = {"x": np_data.astype('float16')}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        self.attrs = {}

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
        conv1 = paddle.nn.Conv2D(self.feed_shape[0][1], 3, 3, bias_attr=False)(
            x
        )

        conv2 = paddle.nn.Conv2D(self.feed_shape[0][1], 3, 3, bias_attr=False)(
            x
        )

        add1 = conv1 + conv2
        conv3 = paddle.nn.Conv2D(add1.shape[1], 8, 8, bias_attr=False)(add1)
        out = F.relu(conv3, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


class TestIntInput(TestBase):
    def set_data_feed(self):
        embedding = np.random.uniform(size=[10, 20])
        indice = np.array([1, 3, 5]).astype(np.int32)
        self.feed_fp32 = {
            "embedding": embedding.astype(np.float32),
            "indice": indice,
        }
        self.feed_fp16 = {
            "embedding": embedding.astype(np.float16),
            "indice": indice,
        }

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
        y = paddle.static.data(
            name=self.feed_list[1], shape=self.feed_shape[1], dtype='int32'
        )
        out = paddle.gather(x, index=y)
        self.fetch_list = [out.name]


if __name__ == "__main__":
    unittest.main()
