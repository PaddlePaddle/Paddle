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
import paddle.distributed as dist  # 导入飞桨分布式训练模块
from paddle.io import BatchSampler, DataLoader, Dataset

# 声明拓扑：定义计算资源
mesh = dist.ProcessMesh([0, 1, 2, 3], dim_names=['dp'])


class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len, self.hidden]).astype(
            "float32"
        )
        label = np.random.uniform(size=[self.seq_len, self.hidden]).astype(
            'float32'
        )
        return (input, label)

    def __len__(self):
        return self.num_samples


class MlpModel(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        self.w0 = self.create_parameter(shape=[1024, 4096])
        self.w1 = self.create_parameter(shape=[4096, 1024])

    def forward(self, x):
        # 切分张量: 标记输入数据沿第0维切分
        dist.shard_tensor(x, mesh, [dist.Shard(0)])
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z


with paddle.LazyGuard():
    model = MlpModel()
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
opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
loss_fn = paddle.nn.MSELoss()

for step, (inputs, labels) in enumerate(dataloader):
    logits = model(inputs)
    loss = loss_fn(logits, labels)
    loss.backward()
    opt.step()
    opt.clear_grad()
