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

import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset


class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(2, 2, bias_attr=False)

    def forward(self, x):
        x = paddle.sum(x, axis=0)
        return x


class MLPLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = paddle.nn.Linear(4, 8, bias_attr=False)
        self.linear2 = paddle.nn.Linear(8, 4, bias_attr=False)
        self.linear3 = paddle.nn.Linear(4, 1, bias_attr=False)
        mesh1 = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.linear1.weight = dist.shard_tensor(
            self.linear1.weight, mesh1, [dist.Replicate()]
        )
        self.linear2.weight = dist.shard_tensor(
            self.linear2.weight, mesh1, [dist.Replicate()]
        )
        self.linear3.weight = dist.shard_tensor(
            self.linear3.weight, mesh1, [dist.Replicate()]
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class RandomDataset(Dataset):
    def __init__(self, inputs, labels):
        super().__init__()
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, index):
        input = self.inputs[index]
        label = self.labels[index]
        return input, label

    def __len__(self):
        return 50


class TestDisableDPSpmd:
    def __init__(self):
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def test_disable_dp_spmd_case1(self):
        a = paddle.to_tensor(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype="float32"
        )
        a = dist.shard_tensor(a, self._mesh, [dist.Shard(0)])
        b = paddle.sum(a, axis=0)
        a_local_value = a._local_value()[0].numpy()
        b_local_value = b._local_value().numpy()

        assert b.placements[0] == dist.Shard(0)
        assert np.array_equal(a_local_value, b_local_value)

    def test_disable_dp_spmd_case2(self):
        a = paddle.to_tensor(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype="float32"
        )
        linear = paddle.nn.Linear(2, 2, bias_attr=False)
        a = dist.shard_tensor(a, self._mesh, [dist.Shard(0)])
        linear.weight = dist.shard_tensor(
            linear.weight, self._mesh, [dist.Replicate()]
        )
        c = linear(a)
        c.backward()
        assert linear.weight.batch_dim == -1
        assert linear.weight.placements[0] == dist.Replicate()

    def test_disable_dp_spmd_case3(self):
        datas = np.random.rand(50, 2, 2).astype("float32")
        labels = np.random.rand(50, 1).astype("float32")
        dataset = RandomDataset(datas, labels)
        sampler = BatchSampler(
            dataset,
            batch_size=2,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
        )
        dist_dataloader = dist.shard_dataloader(
            dataloader, shard_dims=[0, 0], meshes=[self._mesh, self._mesh]
        )

        def loss_fn(x, label):
            return paddle.mean(x)

        model = MyLayer()
        model = dist.to_static(model, dist_dataloader, loss_fn)
        model.train()

        program = model._engine._pir_dist_main_progs["train"]
        for op in program.global_block().ops:
            if op.name() == "pd_op.sum":
                out = op.result(0)
                assert out.placements[0] == dist.Shard(0)

    def test_disable_dp_spmd_case4(self):
        loss_dy = self.run_dy()
        loss_dy2s = self.run_dy2s()

        np.testing.assert_equal(loss_dy, loss_dy2s)

    def set_seed(self):
        paddle.seed(2025)
        np.random.seed(2025)
        random.seed(2025)

    def run_dy(self):
        self.set_seed()
        if dist.get_rank() == 0:
            batch_size = 2
        else:
            batch_size = 4

        datas = np.random.rand(50, 4).astype("float32")
        labels = np.random.rand(50, 1).astype("float32")
        dataset = RandomDataset(datas, labels)

        sampler = BatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
        )
        dist_dataloader = dist.shard_dataloader(
            dataloader, shard_dims=[0, 0], meshes=[self._mesh, self._mesh]
        )

        def loss_fn(x, label):
            return paddle.mean(x)

        model = MLPLayer()
        opt = paddle.optimizer.AdamW(
            learning_rate=0.01, parameters=model.parameters()
        )

        model.train()
        dy_loss = 0
        for step, inputs in enumerate(dist_dataloader()):
            if step > 10:
                break

            data, label = inputs[0], inputs[1]
            logits = model(data)

            loss = loss_fn(logits, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
            dy_loss = loss._local_value().numpy()

        return dy_loss

    def run_dy2s(self):
        self.set_seed()
        if dist.get_rank() == 0:
            batch_size = 2
        else:
            batch_size = 4

        datas = np.random.rand(50, 4).astype("float32")
        labels = np.random.rand(50, 1).astype("float32")
        dataset = RandomDataset(datas, labels)

        sampler = BatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
        )
        dist_dataloader = dist.shard_dataloader(
            dataloader, shard_dims=[0, 0], meshes=[self._mesh, self._mesh]
        )

        def loss_fn(x, label):
            return paddle.mean(x)

        model = MLPLayer()
        opt = paddle.optimizer.AdamW(
            learning_rate=0.01, parameters=model.parameters()
        )

        model = dist.to_static(model, dist_dataloader, loss_fn, opt)
        model.train()

        dy2s_loss = 0
        for step, inputs in enumerate(dist_dataloader()):
            if step > 10:
                break

            loss = model(inputs)
            dy2s_loss = loss
        return dy2s_loss

    def run_test_case(self):
        self.test_disable_dp_spmd_case1()
        self.test_disable_dp_spmd_case2()
        self.test_disable_dp_spmd_case3()
        self.test_disable_dp_spmd_case4()


if __name__ == "__main__":
    test = TestDisableDPSpmd()
    test.run_test_case()
