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
HIDDEN_SIZE = 8
global_mesh = dist.ProcessMesh(
    [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=['pp', 'dp', 'mp']
)
mesh0 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['dp', 'mp'])
mesh1 = dist.ProcessMesh([[4, 5], [6, 7]], dim_names=['dp', 'mp'])


class MlpModel(paddle.nn.Layer):
    def __init__(self, variable_initial_values, run_single_process=False):
        super().__init__()
        self.w0 = self.create_parameter(
            shape=[HIDDEN_SIZE, HIDDEN_SIZE],
            default_initializer=paddle.nn.initializer.Assign(
                variable_initial_values[0]
            ),
        )
        self.w1 = self.create_parameter(
            shape=[HIDDEN_SIZE, HIDDEN_SIZE],
            default_initializer=paddle.nn.initializer.Assign(
                variable_initial_values[1]
            ),
        )
        if run_single_process is False:
            self.w0 = dist.shard_tensor(
                self.w0,
                mesh0,
                [dist.Replicate(), dist.Shard(1)],
            )
            self.w1 = dist.shard_tensor(
                self.w1,
                mesh1,
                [dist.Replicate(), dist.Shard(0)],
            )
        self.run_single_process = run_single_process

    def forward(self, input1, input2):
        x = input1 + input2
        # x: [bs, seq_len, hidden]
        # forward on mesh0
        y = paddle.matmul(x, self.w0)
        # forward on mesh1
        if self.run_single_process is False:
            y = dist.reshard(y, mesh1, [dist.Shard(0), dist.Shard(2)])
        z = paddle.matmul(y, self.w1)
        return z


class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=8):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples
        self.inputs1 = [
            np.random.uniform(size=[self.seq_len, self.hidden]).astype(
                "float32"
            )
            for _ in range(num_samples)
        ]
        self.inputs2 = [
            np.random.uniform(size=[self.seq_len, self.hidden]).astype(
                "float32"
            )
            for _ in range(num_samples)
        ]
        self.labels = [
            np.array(index, dtype="float32") for index in range(num_samples)
        ]

    def __getitem__(self, index):
        return {
            "inputs": [self.inputs1[index], self.inputs2[index]],
            "label": self.labels[index],
        }

    def __len__(self):
        return self.num_samples


def create_dataloader():
    dataset = RandomDataset(SEQ_LEN, HIDDEN_SIZE)
    sampler = BatchSampler(
        dataset,
        batch_size=2,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
    )
    return dataloader


def get_variable_initial_value(var_num=2):
    res = []
    for i in range(var_num):
        res.append(
            paddle.uniform(
                shape=[HIDDEN_SIZE, HIDDEN_SIZE],
                dtype=paddle.float32,
                min=-0.0001,
                max=0.0001,
            )
        )
    return res


def loss_fn(logits, label):
    # logits: [bs, seq_len, hidden], label: [bs]
    loss = paddle.nn.MSELoss(reduction="sum")
    logits = paddle.sum(logits, axis=[1, 2])
    return loss(logits, label)


class TestSemiAutoParallelMultiInputs:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._run_static = eval(os.getenv("run_static"))
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        paddle.set_device(self._backend)
        self.dataloader = create_dataloader()
        self.variable_initial_values = get_variable_initial_value()
        self.single_process_loss = self.get_single_process_loss()

    def get_single_process_loss(self):
        model = MlpModel(
            variable_initial_values=self.variable_initial_values,
            run_single_process=True,
        )
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=model.parameters()
        )
        for step, data in enumerate(self.dataloader()):
            input1, input2 = data["inputs"]
            logits = model(input1, input2)
            label = data["label"]
            loss = loss_fn(logits, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
        return loss.numpy()

    def test_basic(self):
        model = MlpModel(variable_initial_values=self.variable_initial_values)
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=model.parameters()
        )
        dist_dataloader = dist.shard_dataloader(
            dataloader=self.dataloader,
            meshes=[mesh0, mesh1],  # or [[mesh0, mesh0], mesh1]
            shard_dims="dp",
            input_keys=["inputs", "label"],
        )
        cur_rank = paddle.distributed.get_rank()
        if self._run_static:
            dist_model = dist.to_static(model, dist_dataloader, loss_fn, opt)

            for step, data in enumerate(dist_dataloader()):
                input1, input2 = data["inputs"]
                label = data["label"]
                loss = dist_model(input1, input2, label)

            if cur_rank in [5, 7]:
                loss = paddle.to_tensor(loss)
                group = paddle.distributed.new_group([5, 7])
                dist.all_reduce(loss, group=group)
        else:
            dist_opt = dist.shard_optimizer(opt)
            for step, data in enumerate(dist_dataloader()):
                input1, input2 = data["inputs"]
                logits = model(input1, input2)
                label = data["label"]
                loss = loss_fn(logits, label)
                loss.backward()
                dist_opt.step()
                dist_opt.clear_grad()
        if cur_rank in [5, 7]:
            np.testing.assert_allclose(
                loss.numpy(), self.single_process_loss, rtol=1e-06, verbose=True
            )

    def run_test_case(self):
        self.test_basic()


if __name__ == '__main__':
    TestSemiAutoParallelMultiInputs().run_test_case()
