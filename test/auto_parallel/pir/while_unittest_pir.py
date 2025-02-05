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
from paddle.io import DataLoader, Dataset

os.environ["FLAGS_embedding_deterministic"] = "1"
os.environ["FLAGS_cudnn_deterministic"] = "1"

mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
data_world_size = mesh.get_dim_size("x")
dim = 3


def loss_fn(x, label):
    return x


class RandomDataset(Dataset):
    def __init__(self, num_samples=10):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        inputs = paddle.ones(dim, dtype="float32")
        input_type = paddle.ones([], dtype="int64") * idx % 2
        label = paddle.ones(1, dtype="int64")
        return {"inputs": [inputs, input_type], "label": label}

    def __len__(self):
        return self.num_samples


class Layer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter(
            shape=[dim, dim],
            default_initializer=paddle.nn.initializer.Assign(
                0.05 * paddle.ones([dim, dim])
            ),
        )

    def forward(self, x):
        return paddle.matmul(x, self.w)


class DemoModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.layer_a = Layer()

    def forward(self, inputs, input_type):
        tar = paddle.ones_like(input_type) + 3
        inputs = self.layer_a(inputs)
        while not paddle.equal(input_type, tar).all():
            inputs = self.layer_a(inputs)
            input_type = input_type + 1

        return inputs.mean()


class TestWhileDemo:
    def init_env(self):
        paddle.seed(2024)
        np.random.seed(2024)
        random.seed(2024)

    def create_data_loader(self):
        dataset = RandomDataset()
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        dist_dataloader = dist.shard_dataloader(
            dataloader=dataloader,
            meshes=mesh,
            shard_dims="x",
            input_keys=["inputs", "label"],
            is_dataset_splitted=True,
        )
        return dist_dataloader

    def test_dynamic(self, dist_dataloader):
        dy_layer = DemoModel()
        opt_dy = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=dy_layer.parameters()
        )
        dist_opt = dist.shard_optimizer(opt_dy)
        dy_loss_list = []
        for step, data in enumerate(dist_dataloader()):
            [inputs, input_type], _ = data["inputs"], data["label"]
            loss = dy_layer(inputs, input_type)
            loss.backward()
            dist_opt.step()
            dist_opt.clear_grad()
            dy_loss_list.append(loss.numpy())
        dy_loss = np.array(dy_loss_list)
        dy_loss = np.mean(dy_loss)
        return dy_loss

    def test_dynamic2static(self, dist_dataloader):
        paddle.disable_static()
        paddle.base.set_flags({"FLAGS_enable_pir_api": 1})
        dy2static_layer = DemoModel()
        dy2static_opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=dy2static_layer.parameters()
        )
        static_dp_loss_list = []
        dist_model = dist.to_static(
            dy2static_layer, dist_dataloader, loss_fn, dy2static_opt
        )
        dist_model.train()
        for step, data in enumerate(dist_dataloader()):
            loss = dist_model(data["inputs"], data["label"])
            static_dp_loss_list.append(loss)
        dy2static_losses = np.array(static_dp_loss_list)
        pd_partial_loss = paddle.to_tensor(dy2static_losses)
        pd_loss_list = []
        dist.all_gather(pd_loss_list, pd_partial_loss)
        np_dy2static_loss_list = [loss.numpy() for loss in pd_loss_list]
        np_dy2static_loss = np.array(np_dy2static_loss_list)
        np_dy2static_loss = np.mean(np_dy2static_loss)
        return np_dy2static_loss

    def test_while(self):
        self.init_env()
        dist_dataloader = self.create_data_loader()
        dynamic_loss = self.test_dynamic(dist_dataloader)

        self.init_env()
        dist_dataloader = self.create_data_loader()
        dy2static_loss = self.test_dynamic2static(dist_dataloader)

        np.testing.assert_allclose(dynamic_loss, dy2static_loss, atol=1e-8)


if __name__ == "__main__":
    TestWhileDemo().test_while()
