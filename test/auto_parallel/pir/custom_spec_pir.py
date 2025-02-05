# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.dist_input_spec import (
    DistributedInputSpec,
)
from paddle.io import BatchSampler, DataLoader, Dataset

np.random.seed(1127)
paddle.seed(1127)
random.seed(1127)

mesh = dist.ProcessMesh([0, 1], dim_names=["dp"])


class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples
        self.mode = "A"

    def __getitem__(self, index):
        if self.mode == "A":
            input = np.random.uniform(size=[self.seq_len, self.hidden]).astype(
                "float32"
            )
        else:
            input = np.random.normal(
                size=[self.seq_len * 2, self.hidden]
            ).astype("float32")
        label = np.random.randint(0, 2, size=[128]).astype("int64")
        return input, label

    def __len__(self):
        return self.num_samples


class MlpModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.w0 = self.create_parameter(shape=[1024, 4096])
        self.w1 = self.create_parameter(shape=[1024, 4096])

    def forward(self, x):
        y1 = paddle.matmul(x, self.w0)
        y2 = paddle.matmul(x, self.w1)
        z = y1 + y2
        return z


class TestCustomSpec:
    def get_input_spec(self, dist_dataloader):
        dist_dataloader._dataloader.mode = "A"
        input1, label1 = next(dist_dataloader())
        dist_dataloader._dataloader.mode = "B"
        input2, label2 = next(dist_dataloader())

        inputs_spec1 = [DistributedInputSpec.from_dtensor(input1, "input0")]
        inputs_spec2 = [DistributedInputSpec.from_dtensor(input2, "input0")]
        labels_spec = [DistributedInputSpec.from_dtensor(label1, "label0")]

        return [inputs_spec1, labels_spec], [inputs_spec2, labels_spec]

    def run_test(self):
        model = MlpModel()
        loss_func = paddle.nn.CrossEntropyLoss()

        dataset = RandomDataset(128, 1024, 40)
        sampler = BatchSampler(
            dataset,
            batch_size=4,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
        )
        dist_dataloader = dist.shard_dataloader(
            dataloader=dataloader, meshes=mesh, shard_dims="dp"
        )

        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=model.parameters()
        )
        opt2 = copy.deepcopy(opt)

        input_spec1, input_spec2 = self.get_input_spec(dist_dataloader)

        model.w0.stop_gradient = True
        model.w1.stop_gradient = False
        dist_model1 = dist.to_static(
            model, dist_dataloader, loss_func, opt, input_spec=input_spec1
        )
        dist_model1.train()

        model.w0.stop_gradient = False
        model.w1.stop_gradient = True
        dist_model2 = dist.to_static(
            model, dist_dataloader, loss_func, opt2, input_spec=input_spec2
        )
        dist_model2.train()

        datasets_modes = ["A", "A", "B", "A", "B"]
        for mode in datasets_modes:
            dist_dataloader._dataloader.mode = mode
            input, label = next(iter(dist_dataloader))
            if mode == "A":
                before_w0 = dist_model1.state_dict("param")['w0'].mean().numpy()
                before_w1 = dist_model1.state_dict("param")['w1'].mean().numpy()
                loss = dist_model1(input, label)
                after_w0 = dist_model1.state_dict("param")['w0'].mean().numpy()
                after_w1 = dist_model1.state_dict("param")['w1'].mean().numpy()
                assert np.equal(before_w0, after_w0).all()
                assert not np.equal(before_w1, after_w1).all()
            else:
                before_w0 = dist_model2.state_dict("param")['w0'].mean().numpy()
                before_w1 = dist_model2.state_dict("param")['w1'].mean().numpy()
                loss = dist_model2(input, label)
                after_w0 = dist_model2.state_dict("param")['w0'].mean().numpy()
                after_w1 = dist_model2.state_dict("param")['w1'].mean().numpy()
                assert not np.equal(before_w0, after_w0).all()
                assert np.equal(before_w1, after_w1).all()


if __name__ == '__main__':
    TestCustomSpec().run_test()
