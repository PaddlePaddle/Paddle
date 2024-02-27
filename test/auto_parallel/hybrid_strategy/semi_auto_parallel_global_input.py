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

import os

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset

SEQ_LEN = 4
HIDDLE_SIZE = 8
global_mesh = dist.ProcessMesh(
    [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=['pp', 'dp', 'mp']
)
mesh0 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['dp', 'mp'])
mesh1 = dist.ProcessMesh([[4, 5], [6, 7]], dim_names=['dp', 'mp'])


class MlpModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.w0 = dist.shard_tensor(
            self.create_parameter(shape=[HIDDLE_SIZE, HIDDLE_SIZE]),
            mesh0,
            [dist.Replicate(), dist.Shard(1)],
        )
        self.w1 = dist.shard_tensor(
            self.create_parameter(shape=[HIDDLE_SIZE, HIDDLE_SIZE]),
            mesh1,
            [dist.Replicate(), dist.Shard(0)],
        )

        self.global_input = paddle.uniform(
            shape=[SEQ_LEN, HIDDLE_SIZE],
            dtype=paddle.float32,
            min=-0.0001,
            max=0.0001,
        )
        self.global_input = dist.shard_tensor(
            self.global_input,
            global_mesh,
            [dist.Replicate(), dist.Replicate(), dist.Replicate()],
        )

    def process_global_input(self, input):
        return input + 0.0001

    def forward(self, x):
        # x: [bs, seq_len, hidden]
        # forward on mesh0
        global_input = self.process_global_input(self.global_input)

        global_input1 = dist.reshard(
            global_input, mesh0, [dist.Replicate(), dist.Replicate()]
        )
        x = x + global_input1
        y = paddle.matmul(x, self.w0)
        # forward on mesh1
        y = dist.reshard(y, mesh1, [dist.Shard(0), dist.Shard(2)])
        global_input2 = dist.reshard(
            global_input, mesh1, [dist.Replicate(), dist.Replicate()]
        )

        y = y + global_input2
        z = paddle.matmul(y, self.w1)
        return z


class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=8):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples
        self.inputs = [
            np.random.uniform(size=[self.seq_len, self.hidden]).astype(
                "float32"
            )
            for _ in range(num_samples)
        ]
        self.labels = [
            np.array(index, dtype="float32") for index in range(num_samples)
        ]

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return self.num_samples


def create_dataloader():
    dataset = RandomDataset(SEQ_LEN, HIDDLE_SIZE)
    sampler = BatchSampler(
        dataset,
        batch_size=2,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
    )
    dist_dataloader = dist.shard_dataloader(
        dataloader=dataloader, meshes=[mesh0, mesh1], shard_dims="dp"
    )
    return dist_dataloader


def loss_fn(logits, label):
    # logits: [bs, seq_len, hidden], label: [bs]
    loss = paddle.nn.MSELoss(reduction="sum")
    logits = paddle.sum(logits, axis=[1, 2])
    return loss(logits, label)


class TestSemiAutoParallelGlobalInput:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._run_static = eval(os.getenv("run_static"))
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        paddle.set_device(self._backend)
        self.dist_dataloader = create_dataloader()

    def test_basic(self):
        if self._run_static:
            model = MlpModel()
            opt = paddle.optimizer.AdamW(
                learning_rate=0.001, parameters=model.parameters()
            )
            dist_model = dist.to_static(
                model, self.dist_dataloader, loss_fn, opt
            )

            for step, (input, label) in enumerate(self.dist_dataloader()):
                loss = dist_model(input, label)
        else:
            model = MlpModel()
            opt = paddle.optimizer.AdamW(
                learning_rate=0.001, parameters=model.parameters()
            )
            dist_opt = dist.shard_optimizer(opt)

            for step, (input, label) in enumerate(self.dist_dataloader()):
                logits = model(input)
                loss = loss_fn(logits, label)
                loss.backward()
                dist_opt.step()
                dist_opt.clear_grad()
            loss = loss._local_value().numpy()
        cur_rank = paddle.distributed.get_rank()
        if cur_rank == 5 or cur_rank == 4:
            np.testing.assert_allclose(
                loss, 1.40145969, rtol=1e-06, verbose=True
            )
        elif cur_rank == 6 or cur_rank == 7:
            np.testing.assert_allclose(
                loss, 3.1383359, rtol=1e-06, verbose=True
            )

    def run_test_case(self):
        self.test_basic()


if __name__ == '__main__':
    TestSemiAutoParallelGlobalInput().run_test_case()
