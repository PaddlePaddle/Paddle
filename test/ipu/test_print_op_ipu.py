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
from op_test_ipu import IPUD2STest, IPUOpTest

import paddle
import paddle.static
from paddle.jit import to_static


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
        x = paddle.static.data(
            name=self.feed_list[0],
            shape=self.feed_shape[0],
            dtype=self.feed_dtype[0],
        )
        out = paddle.nn.Conv2D(
            in_channels=x.shape[1], out_channels=3, kernel_size=3
        )(x)

        out = paddle.static.Print(out, **self.attrs)

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
            "print_tensor_lod": True,
        }


class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=3, out_channels=1, kernel_size=2, stride=1
        )

    @to_static(full_graph=True)
    def forward(self, x, target=None):
        x = self.conv(x)
        print(x)
        x = paddle.flatten(x, 1, -1)
        if target is not None:
            x = paddle.nn.functional.softmax(x)
            loss = paddle.nn.functional.cross_entropy(
                x, target, reduction='none', use_softmax=False
            )
            loss = paddle.incubate.identity_loss(loss, 1)
            return x, loss
        return x


class TestD2S(IPUD2STest):
    def setUp(self):
        self.set_data_feed()

    def set_data_feed(self):
        self.data = paddle.uniform((8, 3, 10, 10), dtype='float32')
        self.label = paddle.randint(0, 10, shape=[8], dtype='int64')

    def _test(self, use_ipu=False):
        paddle.seed(self.SEED)
        np.random.seed(self.SEED)
        model = SimpleLayer()
        optim = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=model.parameters()
        )

        if use_ipu:
            paddle.set_device('ipu')
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(
                num_ipus=1,
                is_training=True,
                micro_batch_size=1,
                enable_manual_shard=False,
            )
            ipu_strategy.set_optimizer(optim)

        result = []
        for _ in range(2):
            # ipu only needs call model() to do forward/backward/grad_update
            pred, loss = model(self.data, self.label)
            if not use_ipu:
                loss.backward()
                optim.step()
                optim.clear_grad()
            result.append(loss)

        if use_ipu:
            ipu_strategy.release_patch()

        return np.array(result)

    def test_training(self):
        ipu_loss = self._test(True).flatten()
        cpu_loss = self._test(False).flatten()
        np.testing.assert_allclose(ipu_loss, cpu_loss, rtol=1e-05, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
