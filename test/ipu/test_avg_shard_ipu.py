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
from op_test_ipu import IPUOpTest

import paddle
import paddle.static


class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()

    def set_atol(self):
        self.atol = 2e-6
        self.rtol = 1e-5
        self.atol_fp16 = 1e-2
        self.rtol_fp16 = 1e-3

    def set_data_feed(self):
        data = np.random.uniform(size=[1, 3, 128, 128])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
        x = paddle.nn.Conv2D(
            in_channels=x.shape[1],
            out_channels=3,
            kernel_size=3,
            bias_attr=False,
        )(x)
        x = paddle.nn.Conv2D(
            in_channels=x.shape[1],
            out_channels=3,
            kernel_size=3,
            bias_attr=False,
        )(x)
        x = paddle.nn.Conv2D(
            in_channels=x.shape[1],
            out_channels=3,
            kernel_size=3,
            bias_attr=False,
        )(x)
        x = paddle.nn.Conv2D(
            in_channels=x.shape[1],
            out_channels=3,
            kernel_size=3,
            bias_attr=False,
        )(x)

        self.fetch_list = [x]

    def run_model(self, exec_mode):
        ipu_strategy = paddle.static.IpuStrategy()
        ipu_strategy.set_graph_config(is_training=self.is_training)
        ipu_strategy.set_options({'need_avg_shard': True})
        self.run_op_test(exec_mode, ipu_strategy)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


if __name__ == "__main__":
    unittest.main()
