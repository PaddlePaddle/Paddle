# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import nn
from paddle.distributed import Shard, fleet
from paddle.distributed.fleet import auto


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def dygraph_parallel_cross_entropy(data, label):
    model = fleet.meta_parallel.ParallelCrossEntropy()
    loss = model(data, label)
    return paddle.mean(loss)


def dygraph_cross_entropy(data, label):
    model = paddle.nn.CrossEntropyLoss()
    loss = model(data, label)
    return loss


class MyDataset(paddle.io.Dataset):
    def __init__(self, data, label):
        self._data = data
        self._label = label

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return self._data.shape[0]


class MyMLP(nn.Layer):
    def __init__(self, process_mesh, placements):
        super().__init__()
        self.process_mesh = process_mesh
        self.placements = placements

    def forward(self, x):
        dist.shard_tensor(
            x, self.process_mesh, self.placements, stop_gradient=False
        )
        return x


def auto_parallel_cross_entropy(data, label, process_mesh, placements):
    with paddle.LazyGuard():
        model = MyMLP(process_mesh, placements)
        loss_layer = paddle.nn.CrossEntropyLoss()
        auto.fetch("input0@GRAD", "logits_grad", logging=False)
        auto.fetch("input0", "logits", logging=False)
        auto.fetch(
            "softmax_with_cross_entropy_0.tmp_1",
            "loss_before_mean",
            logging=False,
        )
    optimizer = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=model.parameters()
    )
    engine = auto.Engine(model, loss_layer, optimizer)
    train_dataset = MyDataset(data, label)
    log = engine.fit(train_dataset, epochs=1, batch_size=data.shape[0])
    logit_grad = np.array(log.history["fetches"][0]["logits_grad"])
    loss = np.array(log.history["loss"])
    logits = np.array(log.history["fetches"][0]["logits"])
    paddle.disable_static()
    return loss, logit_grad


class TestHybridDistTraining:
    def __init__(self):
        self.nsample = 2
        self.seq_len = 2
        self.nclass = 4
        self.seed = 100

    def run_test_case(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 2,
            "mp_degree": 2,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

        nsample = self.nsample
        nclass = self.nclass
        seq_len = self.seq_len
        seed = self.seed

        set_random_seed(seed)
        rank_id = dist.get_rank()

        paddle.seed(rank_id * 10)
        random.seed(seed)
        np.random.seed(seed)
        process_mesh = dist.ProcessMesh(
            mesh=[[0, 1], [2, 3]], dim_names=["x", "y"]
        )

        integral_np_label = np.random.randint(0, nclass, (nsample, seq_len, 1))
        integral_label = paddle.to_tensor(integral_np_label, dtype="int64")

        integral_np_data = np.random.randn(nsample, seq_len, nclass).astype(
            "float32"
        )
        integral_data = paddle.to_tensor(integral_np_data)
        integral_data.stop_gradient = False

        loss_dygraph = dygraph_cross_entropy(integral_data, integral_label)
        dp_start_idx = rank_id // 2 * (nsample // 2)
        dp_end_idx = dp_start_idx + (nsample // 2)
        mp_start_idx = rank_id % 2 * (nclass // 2)
        mp_end_idx = mp_start_idx + (nclass // 2)
        # the dataloader cannot support shard on non-batch dim,
        # so we should slice the data and label tensor manually
        mp_sliced_np_data = integral_np_data[:, :, mp_start_idx:mp_end_idx]
        loss_auto, auto_grad = auto_parallel_cross_entropy(
            mp_sliced_np_data,
            integral_np_label,
            process_mesh,
            [Shard(0), Shard(2)],
        )
        pd_loss_auto = paddle.to_tensor(loss_auto)
        paddle.distributed.all_reduce(pd_loss_auto)
        pd_loss_auto = pd_loss_auto / 4
        np.testing.assert_allclose(
            loss_dygraph.numpy(), pd_loss_auto.numpy(), rtol=1e-6
        )

        loss_dygraph.backward()

        sliced_grad = integral_data.grad[
            dp_start_idx:dp_end_idx, :, mp_start_idx:mp_end_idx
        ]
        partial_grad = sliced_grad.clone().detach()

        np.testing.assert_allclose(
            partial_grad.numpy(),
            auto_grad,
            rtol=1e-6,
        )


if __name__ == '__main__':
    TestHybridDistTraining().run_test_case()
