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
        self.set_data_feed()
        self.set_feed_attr()
        self.set_attrs()
        self.set_training()

    @property
    def fp16_enabled(self):
        return False

    def set_atol(self):
        super().set_atol()
        self.atol = 1e-6
        self.rtol = 1e-5

    def set_data_feed(self):
        self.feed_fp32 = {
            "image": np.random.uniform(size=[1, 3, 10, 10]).astype('float32'),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_attrs(self):
        self.attrs = {
            "optimizer": 'sgd',
            "weight_decay": 0.0,
        }

    def set_training(self):
        self.is_training = True
        self.epoch = 100

    @IPUOpTest.static_graph
    def build_model(self):
        image = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )
        conv1 = paddle.nn.Conv2D(
            in_channels=image.shape[1],
            out_channels=3,
            kernel_size=3,
            bias_attr=False,
        )(image)
        loss = paddle.mean(conv1)
        self.fetch_list = [loss]

        weight_decay = self.attrs['weight_decay']
        # Only support ClipGradByGlobalNorm
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        if self.attrs['optimizer'] == 'sgd':
            opt = paddle.optimizer.SGD(
                learning_rate=1e-1, weight_decay=weight_decay, grad_clip=clip
            )
        elif self.attrs['optimizer'] == 'adam':
            opt = paddle.optimizer.Adam(
                learning_rate=1e-1, weight_decay=weight_decay, grad_clip=clip
            )
        elif self.attrs['optimizer'] == 'lamb':
            opt = paddle.optimizer.Lamb(
                learning_rate=1e-1,
                lamb_weight_decay=weight_decay,
                grad_clip=clip,
            )
        else:
            raise ValueError(
                f"Not supported optimizer {self.attrs['optimizer']} for test"
            )
        opt.minimize(loss)

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


class TestAdam(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'adam',
            "weight_decay": 0.0,
        }


class TestLamb(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'lamb',
            "weight_decay": 0.1,
        }


if __name__ == "__main__":
    unittest.main()
