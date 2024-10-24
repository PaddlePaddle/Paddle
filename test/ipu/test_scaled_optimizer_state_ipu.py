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
import paddle.static


class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_attrs()

    @property
    def fp16_enabled(self):
        return False

    def set_training(self):
        self.is_training = True
        self.epoch = 100

    def set_data_feed(self):
        data = np.random.uniform(size=[1, 3, 10, 10]).astype('float32')
        self.feed_fp32 = {"image": data.astype(np.float32)}
        self.feed_fp16 = {"image": data.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_attrs(self):
        self.attrs = {
            "optimizer": 'lamb',
            "weight_decay": 0.0,
            "scaled_optimizer_state": True,
        }

    @IPUOpTest.static_graph
    def build_model(self):
        image = paddle.static.data(
            name='image', shape=[1, 3, 10, 10], dtype='float32'
        )
        conv1 = paddle.nn.Conv2D(
            in_channels=3, out_channels=3, kernel_size=3, bias_attr=False
        )(image)
        loss = paddle.mean(conv1)

        weight_decay = self.attrs['weight_decay']
        opt = paddle.optimizer.Adam(
            learning_rate=1e-1, weight_decay=weight_decay
        )
        if self.attrs['optimizer'] == 'lamb':
            opt = paddle.optimizer.Lamb(
                learning_rate=1e-1, lamb_weight_decay=weight_decay
            )
        opt.minimize(loss)
        self.feed_list = [image.name]
        self.fetch_list = [loss]

    def run_model(self, exec_mode):
        ipu_strategy = paddle.static.IpuStrategy()
        ipu_strategy.set_graph_config(is_training=self.is_training)
        if self.is_ipu_mode(exec_mode):
            if "use_no_bias_optimizer" in self.attrs.keys():
                ipu_strategy.set_options(
                    {
                        "use_no_bias_optimizer": self.attrs[
                            "use_no_bias_optimizer"
                        ]
                    }
                )
            if "scaled_optimizer_state" in self.attrs.keys():
                ipu_strategy.set_options(
                    {
                        "scaled_optimizer_state": self.attrs[
                            "scaled_optimizer_state"
                        ]
                    }
                )
        self.run_op_test(exec_mode, ipu_strategy=ipu_strategy)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


class TestScaledAdam(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'adam',
            "weight_decay": 0.0,
            "scaled_optimizer_state": True,
        }

    def set_atol(self):
        super().set_atol()
        self.atol = 1e-5
        self.rtol = 1e-5


@unittest.skip('cpu do not support AdamNoBias')
class TestScaledAdamNoBias(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'adam',
            "weight_decay": 0.0,
            "use_no_bias_optimizer": True,
            "scaled_optimizer_state": True,
        }


@unittest.skip('cpu do not support LambNoBias')
class TestScaledLambNoBias(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'lamb',
            "weight_decay": 0.0,
            "use_no_bias_optimizer": True,
            "scaled_optimizer_state": True,
        }


if __name__ == "__main__":
    unittest.main()
