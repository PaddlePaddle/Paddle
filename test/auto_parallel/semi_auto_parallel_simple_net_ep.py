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

import os
import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.io import BatchSampler, DataLoader


class Config:
    def __init__(self):
        self.batch_num = 5
        self.batch_size = 4
        self.input_size = 32
        self.hidden_size = 16
        self.class_num = 10
        self.run_ep = False
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.expert_mesh_list = []
        self.expert_mesh_list.append(dist.ProcessMesh([0], dim_names=["x"]))
        self.expert_mesh_list.append(dist.ProcessMesh([1], dim_names=["x"]))


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples, return_dict=False):
        self.images = images
        self.labels = labels
        self.num_samples = self.images.shape[0]
        self.return_dict = return_dict

    def __getitem__(self, idx):
        if self.return_dict:
            return {
                "image": self.images[idx],
                "label": self.labels[idx],
            }
        else:
            return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.class_num = config.class_num
        self.down_proj = nn.Linear(
            self.hidden_size, self.class_num, bias_attr=False
        )

    def redistribute_expert(self, mesh, placements):
        # place the experts on different devices
        self.down_proj.weight = dist.shard_tensor(
            self.down_proj.weight, mesh, placements
        )

    def forward(self, x):
        return self.down_proj(x)


class DemoLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(
            config.input_size, config.hidden_size, bias_attr=False
        )
        self.gate.weight = dist.shard_tensor(
            self.gate.weight, config.mesh, [dist.Replicate()]
        )

        self.experts = nn.LayerList()
        self.experts.append(MLP(config))
        self.experts.append(MLP(config))
        if config.run_ep:
            for i, expert in enumerate(self.experts):
                expert.redistribute_expert(
                    config.expert_mesh_list[i], [dist.Replicate()]
                )

    def forward(self, x):
        h = self.gate(x)
        if self.config.run_ep:
            local_val_list = dist.auto_parallel.api.moe_sub_mesh_tensors(
                h, self.config.mesh, 0, [dist.Shard(0)]
            )
        else:
            local_val_list = paddle.split(h, num_or_sections=2, axis=0)
        expert_out_list = []
        for i, expert in enumerate(self.experts):
            local_val = local_val_list[i]
            expert_out_list.append(expert(local_val))
        if self.config.run_ep:
            out = dist.auto_parallel.api.moe_global_mesh_tensor(
                expert_out_list, self.config.mesh, [dist.Shard(0)], 0
            )
        else:
            out = paddle.stack(expert_out_list, axis=0)
            out = out.reshape((-1, self.config.class_num))
        return out


class Criterion(nn.Layer):
    def __init__(self):
        super().__init__()
        self.loss_func = paddle.nn.MSELoss()

    def forward(self, logits, labels):
        loss = self.loss_func(logits, labels)
        return loss


class TestSimpleNetForEP:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))

        paddle.set_device(self._backend)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def create_optimizer(self, model, lr_scheduler=None):
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01,
            parameters=model.parameters(),
            grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
        )
        return optimizer

    def create_data_loader(self, config):
        nsamples = config.batch_size * config.batch_num
        images = np.random.rand(nsamples, config.input_size).astype('float32')
        labels = np.random.rand(nsamples, config.class_num).astype('float32')
        train_dataset = RandomDataset(images, labels, config.batch_size)
        train_sampler = BatchSampler(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
        )
        return train_dataloader

    def build(self, config):
        model = DemoLayer(config)
        dataloader = self.create_data_loader(config)
        optimizer = self.create_optimizer(model)
        criterion = Criterion()
        return model, dataloader, criterion, optimizer

    def train(self, config, model, train_dataloader, criterion, optimizer):
        tr_loss = float(0)
        global_step = 0
        model.train()

        losses = []
        for step, inputs in enumerate(train_dataloader()):
            inputs, labels = inputs
            logits = model(inputs)
            tr_loss = criterion(logits, labels)

            tr_loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            losses.append(tr_loss.numpy())

        return losses

    def run_ep(self):
        self.set_seed(self._seed)
        config = Config()
        config.run_ep = True
        model, train_dataloader, criterion, optimizer = self.build(config)

        dist_dataloader = dist.shard_dataloader(
            train_dataloader, config.mesh, shard_dims="x"
        )
        loss = self.train(config, model, dist_dataloader, criterion, optimizer)

        return loss

    def run_replicate(self):
        self.set_seed(self._seed)
        config = Config()
        config.run_ep = False
        model, train_dataloader, criterion, optimizer = self.build(config)

        loss = self.train(config, model, train_dataloader, criterion, optimizer)
        return loss

    def test_ep_demo_net(self):
        ep_loss = self.run_ep()
        replicate_loss = self.run_replicate()
        np.testing.assert_allclose(ep_loss, replicate_loss, rtol=1e-6)

    def run_test_case(self):
        self.test_ep_demo_net()


if __name__ == "__main__":
    TestSimpleNetForEP().run_test_case()
