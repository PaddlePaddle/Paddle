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
    paddle.disable_static()
    return loss, logit_grad


class TestMpDistTraining:
    def __init__(self):
        self.nsample = 40
        self.nclass = 20
        self.seed = 100

    def run_test_case(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": self.model_parallel_size,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

        nsample = self.nsample
        nclass = self.nclass
        seed = self.seed

        set_random_seed(seed)
        rank_id = dist.get_rank()

        paddle.seed(rank_id * 10)
        random.seed(seed)
        np.random.seed(seed)
        check_group = dist.new_group(list(range(self.model_parallel_size)))
        process_mesh = dist.ProcessMesh(mesh=[0, 1], dim_names=["x"])

        np_label = np.random.randint(0, nclass, (nsample, 1))
        label = paddle.to_tensor(np_label, dtype="int64")

        data = paddle.randn(
            shape=[nsample, nclass // self.model_parallel_size],
            dtype='float32',
        )
        data.stop_gradient = False

        integral_data = []
        partial_data = data.clone().detach()
        paddle.distributed.all_gather(
            integral_data, partial_data, group=check_group
        )
        integral_data = paddle.concat(integral_data, axis=-1)
        integral_data = integral_data.detach().clone()
        integral_data.stop_gradient = False

        loss_dygraph_parallel = dygraph_parallel_cross_entropy(data, label)
        loss_auto, auto_grad = auto_parallel_cross_entropy(
            data.numpy(), np_label, process_mesh, [Shard(1)]
        )

        np.testing.assert_allclose(
            loss_dygraph_parallel.numpy(), loss_auto, rtol=1e-6
        )

        loss_dygraph_parallel.backward()

        integral_grad = []
        partial_grad = data.grad.clone().detach()
        paddle.distributed.all_gather(
            integral_grad, partial_grad, group=check_group
        )
        integral_grad = paddle.concat(integral_grad, axis=-1)

        integral_auto_grad = []
        paddle.distributed.all_gather(
            integral_auto_grad,
            paddle.to_tensor(auto_grad),
            group=check_group,
        )
        integral_auto_grad = paddle.concat(integral_auto_grad, axis=-1)

        parallel_grad = integral_grad.numpy()
        auto_grad = integral_auto_grad.numpy()
        np.testing.assert_allclose(
            parallel_grad,
            auto_grad,
            rtol=1e-6,
        )


if __name__ == '__main__':
    TestMpDistTraining().run_test_case()
