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
        self.set_op_attrs()

    def set_atol(self):
        self.atol = 1e-6
        self.rtol = 1e-5
        self.atol_fp16 = 1e-2
        self.rtol_fp16 = 1e-3

    def set_data_feed(self):
        x = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {"x": x.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        self.attrs = {"epsilon": 1e-05}

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )

        if self.is_training:
            ch = self.feed_shape[0][1]
            conv1 = paddle.static.nn.conv2d(
                x, num_filters=ch, filter_size=3, bias_attr=False
            )
            scale = paddle.ParamAttr(trainable=True)
            bias = paddle.ParamAttr(trainable=True)
            out = paddle.nn.InstanceNorm2D(
                num_features=ch, weight_attr=scale, bias_attr=bias, **self.attrs
            )(conv1)
            loss = paddle.mean(out)
            adam = paddle.optimizer.Adam(learning_rate=1e-2)
            adam.minimize(loss)
            self.fetch_list = [loss.name]
        else:
            out = paddle.nn.InstanceNorm2D(
                num_features=self.feed_shape[0][1],
                weight_attr=True,
                bias_attr=True,
                **self.attrs,
            )(x)
            self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


class TestTrainCase1(TestBase):
    def set_training(self):
        self.is_training = True
        self.epoch = 10


if __name__ == "__main__":
    unittest.main()
