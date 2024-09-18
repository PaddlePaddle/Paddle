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

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed.auto_parallel.high_level_api import (
    ToDistributedConfig,
    to_distributed,
)

BATCH_SIZE = 4
BATCH_NUM = 3
IMAGE_SIZE = 16
HIDDEN_SIZE = 4096
CLASS_NUM = 8


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class DemoNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.linear_0 = nn.Linear(IMAGE_SIZE, HIDDEN_SIZE)
        self.linear_1 = nn.Linear(HIDDEN_SIZE, IMAGE_SIZE)
        self.gelu_0 = nn.GELU()
        self.linear_2 = nn.Linear(IMAGE_SIZE, HIDDEN_SIZE)
        self.linear_3 = nn.Linear(HIDDEN_SIZE, CLASS_NUM)
        self.gelu_1 = nn.GELU()

    def forward(self, x):
        out = self.relu(x)
        out = self.linear_0(out)
        out = self.gelu_0(out)
        out = self.linear_1(out)
        out = self.linear_2(out)
        out = self.gelu_1(out)
        out = self.linear_3(out)
        return out


# create mesh
mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

# create dataset and dataloader
images = np.random.rand(BATCH_SIZE * BATCH_NUM, IMAGE_SIZE).astype('float32')
labels = np.random.rand(BATCH_SIZE * BATCH_NUM, CLASS_NUM).astype('float32')
dataset = RandomDataset(images, labels, BATCH_SIZE * BATCH_NUM)
loader = paddle.io.DataLoader(dataset, batch_size=BATCH_SIZE)
# create demonet & opt & loss_fn
model = DemoNet()
loss_fn = nn.MSELoss()
opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=model.parameters())

# shard dataloader
dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
# config: input_spec
image_spec = paddle.static.InputSpec([4, 16], 'float32', 'image', True)
dist_config = ToDistributedConfig()
dist_config.input_spec = [image_spec]
# wrap model by using **to_distributed**
dist_model = to_distributed(model, mesh, dist_config)
# dist_model = model

dist_model.train()
for batch_id, (image, label) in enumerate(dist_loader()):
    # dynamic
    print(f"batch: {batch_id}, image is {image}, label is {label}")
    logits = dist_model(image)
    loss = loss_fn(logits, label)
    print(f"loss is {loss}")
    loss.backward()
    opt.step()
    opt.clear_grad()
    # if batch_id == 2:
    #     breakpoint()
