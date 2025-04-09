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


class RandomDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        input = np.random.uniform(size=[2, 2]).astype("float32")
        label = np.random.randint(0, 2, size=[]).astype("int64")
        return input, label

    def __len__(self):
        return 100


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
        dataset = RandomDataset()
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

    def run_test_case(self):
        self.test_disable_dp_spmd_case1()
        self.test_disable_dp_spmd_case2()
        self.test_disable_dp_spmd_case3()


if __name__ == "__main__":
    test = TestDisableDPSpmd()
    test.run_test_case()
