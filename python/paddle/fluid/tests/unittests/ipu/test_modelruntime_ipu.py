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

import os
import unittest

import numpy as np
import paddle
from op_test_ipu import IPUOpTest


class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=3, out_channels=1, kernel_size=2, stride=1
        )

    def forward(self, x, target=None):
        x = self.conv(x)
        x = paddle.fluid.layers.flatten(x, axis=1)
        if target is not None:
            x = paddle.fluid.layers.softmax(x)
            loss = paddle.fluid.layers.cross_entropy(x, target)
            return x, loss
        return x


class TestBase(IPUOpTest):
    def setUp(self):
        self.ipu_model = None
        self.set_attrs()
        if 'POPLAR_IPUMODEL' in os.environ:
            self.ipu_model = os.environ['POPLAR_IPUMODEL']
            del os.environ['POPLAR_IPUMODEL']

    def set_attrs(self):
        self.timeout = 0.0
        self.batch_size = 8

    def tearDown(self):
        if getattr(self, 'ipu_model', None):
            os.environ['POPLAR_IPUMODEL'] = self.ipu_model
        paddle.framework.core.IpuBackend.get_instance().reset()

    def generate_feed(self):
        return {
            "X": np.random.rand(8, 3, 10, 10).astype(np.float32),
            "Y": np.random.randint(0, 10, [8], dtype="int64"),
        }

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name='X', shape=[self.batch_size, 3, 10, 10], dtype='float32'
        )
        label = paddle.static.data(
            name='Y', shape=[self.batch_size], dtype='int64'
        )
        model = SimpleLayer()
        pred, loss = model(x, label)
        self.feed_list = [x.name, label.name]
        self.fetch_list = [pred.name, loss.name]

    def reset_seeds(self):

        np.random.seed(self.SEED)
        paddle.seed(self.SEED)
        self.main_prog.random_seed = self.SEED
        self.startup_prog.random_seed = self.SEED

    def _test(self, use_ipu=False):

        self.reset_seeds()
        place = paddle.IPUPlace() if use_ipu else paddle.CPUPlace()

        executor = paddle.static.Executor(place)
        executor.run(self.startup_prog)

        if use_ipu:
            paddle.set_device('ipu')
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(
                num_ipus=1,
                is_training=False,
                micro_batch_size=self.batch_size,
                enable_manual_shard=False,
            )
            ipu_strategy.set_options(
                {
                    'enable_model_runtime_executor': True,
                    'timeout_ms': self.timeout,
                }
            )
            program = paddle.static.IpuCompiledProgram(
                self.main_prog, ipu_strategy=ipu_strategy
            ).compile(self.feed_list, self.fetch_list)
        else:
            program = self.main_prog

        epochs = 10
        preds = []
        losses = []
        for epoch in range(epochs):
            feed = self.generate_feed()
            dy_batch = feed["X"].shape[0]
            if not use_ipu:
                # padding inputs
                pad_batch = self.batch_size - dy_batch
                for k, v in feed.items():
                    pad_size = tuple(
                        (
                            (0, 0 if i != 0 else pad_batch)
                            for i in range(len(v.shape))
                        )
                    )
                    feed[k] = np.pad(v, pad_size, 'constant', constant_values=0)

            pred, loss = executor.run(
                program, feed=feed, fetch_list=self.fetch_list
            )
            if not use_ipu:
                pred = pred[0:dy_batch]
                loss = loss[0:dy_batch]

            preds.append(pred)
            losses.append(loss)

        return np.concatenate(preds, axis=0), np.concatenate(losses, axis=0)

    def test_infer(self):
        self.build_model()
        ipu_pred, ipu_loss = self._test(True)
        cpu_pred, cpu_loss = self._test(False)
        np.testing.assert_allclose(
            ipu_pred.flatten(), cpu_pred.flatten(), rtol=1e-05, atol=1e-4
        )
        np.testing.assert_allclose(
            ipu_loss.flatten(), cpu_loss.flatten(), rtol=1e-05, atol=1e-4
        )


class TestAutoBatch(TestBase):
    def set_attrs(self):
        self.timeout = 0.01
        # fixed batch
        self.batch_size = 8

    def generate_feed(self):
        # generate dynamic batch
        batch = np.random.randint(1, self.batch_size)
        return {
            "X": np.random.rand(batch, 3, 10, 10).astype(np.float32),
            "Y": np.random.randint(0, 10, [batch], dtype="int64"),
        }


if __name__ == "__main__":
    unittest.main()
