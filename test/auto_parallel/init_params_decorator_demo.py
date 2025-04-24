# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset


class TestInitParamsDecorator:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        paddle.set_device(self._backend)

    def test_Single_card_perspective_training(self):
        # Implicit initialization
        class LazyInitLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.weight = self.create_parameter(shape=[10, 5])

            def forward(self, x):
                return paddle.matmul(x, self.weight)

        with paddle.LazyGuard():
            layer = LazyInitLayer()

        assert not layer.weight._is_initialized()

        x = paddle.randn([4, 10])
        output = layer(x)

        assert layer.weight._is_initialized()

        # Explicit initialization
        class ManualInitLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.weight = self.create_parameter(shape=[10, 5])
                self.bias = self.create_parameter(shape=[5], is_bias=True)

            def forward(self, x):
                return paddle.matmul(x, self.weight) + self.bias

        with paddle.LazyGuard():
            layer = ManualInitLayer()

        assert not layer.weight._is_initialized()
        assert not layer.bias._is_initialized()

        for p in layer.parameters():
            if not p._is_initialized():
                p.initialize()

        assert layer.weight._is_initialized()
        assert layer.bias._is_initialized()

    def test_distributed_training(self):
        mesh = dist.ProcessMesh([0, 1, 2, 3], dim_names=['mp'])

        class RandomDataset(Dataset):
            def __init__(self, seq_len, hidden, num_samples=100):
                super().__init__()
                self.seq_len = seq_len
                self.hidden = hidden
                self.num_samples = num_samples

            def __getitem__(self, index):
                input = np.random.uniform(
                    size=[self.seq_len, self.hidden]
                ).astype("float32")
                label = np.random.uniform(
                    size=[self.seq_len, self.hidden]
                ).astype('float32')
                return (input, label)

            def __len__(self):
                return self.num_samples

        class MlpModel(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.w0 = self.create_parameter(shape=[1024, 4096])
                self.w1 = self.create_parameter(shape=[4096, 1024])

                self.w0 = dist.shard_tensor(self.w0, mesh, [dist.Shard(1)])
                self.w1 = dist.shard_tensor(self.w1, mesh, [dist.Shard(0)])

            def forward(self, x):
                y = paddle.matmul(x, self.w0)
                z = paddle.matmul(y, self.w1)
                return z

        with paddle.LazyGuard():
            model = MlpModel()

        assert not model.w0._is_initialized()
        assert not model.w1._is_initialized()

        for p in model.parameters():
            p.initialize()

        dataset = RandomDataset(128, 1024)
        sampler = BatchSampler(
            dataset,
            batch_size=4,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
        )
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=model.parameters()
        )
        loss_fn = paddle.nn.MSELoss()

        for step, (inputs, labels) in enumerate(dataloader):
            if step >= 3:
                break
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            opt.clear_grad()

            assert model.w0._is_initialized()
            assert model.w1._is_initialized()
            assert not paddle.isnan(loss)

    def test_params_decorator(self):
        self.test_Single_card_perspective_training()
        self.test_distributed_training()


if __name__ == '__main__':
    TestInitParamsDecorator().test_params_decorator()
