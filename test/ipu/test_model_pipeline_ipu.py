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
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()

    def set_data_feed(self):
        data = np.random.uniform(size=[2, 3, 10, 10])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}

    def set_feed_attr(self):
        self.feed_shape = [(1, 3, 10, 10)]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        image = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
        with paddle.static.ipu_shard_guard(index=0):
            conv1 = paddle.nn.Conv2D(
                in_channels=image.shape[1],
                out_channels=3,
                kernel_size=3,
                bias_attr=False,
            )(image)

        with paddle.static.ipu_shard_guard(index=1):
            conv2 = paddle.nn.Conv2D(
                in_channels=conv1.shape[1],
                out_channels=3,
                kernel_size=3,
                bias_attr=False,
            )(conv1)

            loss = paddle.mean(conv2)
        self.fetch_list = [loss]

    def run_model(self, exec_mode):
        ipu_strategy = paddle.static.IpuStrategy()
        ipu_strategy.set_graph_config(
            num_ipus=2, is_training=False, enable_manual_shard=True
        )
        ipu_strategy.set_pipelining_config(
            enable_pipelining=True, batches_per_step=2
        )
        self.run_op_test(exec_mode, ipu_strategy=ipu_strategy)

    def test(self):
        self.build_model()
        self.run_model(IPUOpTest.ExecutionMode.IPU_FP32)


if __name__ == "__main__":
    unittest.main()
