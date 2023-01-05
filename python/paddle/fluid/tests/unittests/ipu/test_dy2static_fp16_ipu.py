# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import unittest

import numpy as np

import paddle
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUD2STest


class SimpleLayer(paddle.nn.Layer):
    def __init__(self, use_ipu=False):
        super().__init__()
        self.use_ipu = use_ipu
        self.conv = paddle.nn.Conv2D(
            in_channels=3, out_channels=1, kernel_size=2, stride=1
        )

    def forward(self, x, target=None):
        x = self.conv(x)
        x = paddle.flatten(x, 1, -1)
        if target is not None:
            x = paddle.nn.functional.softmax(x)
            loss = paddle.paddle.nn.functional.cross_entropy(
                x, target, reduction='none', use_softmax=False
            )
            if self.use_ipu:
                loss = paddle.incubate.identity_loss(loss, 1)
            else:
                loss = paddle.mean(loss)
            return x, loss
        return x


class TestBase(IPUD2STest):
    def setUp(self):
        super().setUp()
        self.save_path = tempfile.TemporaryDirectory()

    def tearDown(self):
        super().tearDown()
        self.save_path.cleanup()

    def _test(self, use_ipu=False):
        paddle.seed(self.SEED)
        np.random.seed(self.SEED)
        model = SimpleLayer(use_ipu)
        specs = [
            paddle.static.InputSpec(
                name="x", shape=[32, 3, 10, 10], dtype="float32"
            ),
            paddle.static.InputSpec(name="target", shape=[32], dtype="int64"),
        ]
        model = paddle.jit.to_static(model, input_spec=specs)
        optim = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=model.parameters()
        )
        data = paddle.uniform((32, 3, 10, 10), dtype='float32')
        label = paddle.randint(0, 10, shape=[32], dtype='int64')
        model_path = '{}/model_state_dict_{}.pdparams'.format(
            self.save_path, 'ipu' if use_ipu else 'cpu'
        )
        optim_path = '{}/optim_state_dict_{}.pdopt'.format(
            self.save_path, 'ipu' if use_ipu else 'cpu'
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
            ipu_strategy.set_precision_config(enable_fp16=True)
            ipu_strategy.set_optimizer(optim)
            data = data.astype(np.float16)

        epochs = 100
        result = []
        for _ in range(epochs):
            # ipu only needs call model() to do forward/backward/grad_update
            pred, loss = model(data, label)
            if not use_ipu:
                loss.backward()
                optim.step()
                optim.clear_grad()
            result.append(loss)

        if use_ipu:
            paddle.fluid.core.IpuBackend.get_instance().weights_to_host()

        paddle.save(model.state_dict(), model_path)
        paddle.save(optim.state_dict(), optim_path)
        model.set_state_dict(paddle.load(model_path))
        optim.set_state_dict(paddle.load(optim_path))

        for _ in range(epochs):
            # ipu only needs call model() to do forward/backward/grad_update
            pred, loss = model(data, label)
            if not use_ipu:
                loss.backward()
                optim.step()
                optim.clear_grad()

            result.append(loss)

        if use_ipu:
            ipu_strategy.release_patch()

        return np.array(result)

    def test_training(self):
        cpu_loss = self._test(False).flatten()
        ipu_loss = self._test(True).flatten()
        np.testing.assert_allclose(ipu_loss, cpu_loss, rtol=1e-05, atol=0.01)


if __name__ == "__main__":
    unittest.main()
