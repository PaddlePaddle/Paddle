# -*- coding: UTF-8 -*-

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

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.nn import Linear, ReLU

seed = 2022
epoch = 2
linear_size = 1000

np.random.seed(seed)
paddle.seed(seed)


class MLP(paddle.nn.Layer):
    def __init__(self, linear_size=1000):
        super().__init__()

        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
        self._linear3 = Linear(linear_size, 10)
        self._relu = ReLU()

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        y = self._relu(y)
        return y


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples=200, linear_size=1000):
        self.num_samples = num_samples
        self.linear_size = linear_size

    def __getitem__(self, idx):
        img = np.random.rand(self.linear_size).astype('float32')
        return img

    def __len__(self):
        return self.num_samples


def optimizer_setting(model):
    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=0.00001,
        weight_decay=0.00001,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
    )

    return optimizer


def train_mlp(
    model,
    use_sharding_stage1=False,
    accumulate_grad=False,
    test_dp=False,
):
    optimizer = optimizer_setting(model=model)

    strategy = fleet.DistributedStrategy()
    level = 'O1'
    custom_white_list = [
        "matmul_v2",
        "elementwise_add",
        "relu",
        "reduce_mean",
    ]

    if use_sharding_stage1:
        if test_dp:
            dp_degree = 2
            sharding_degree = 2
        else:
            dp_degree = 1
            sharding_degree = 4
    else:
        dp_degree = -1
        sharding_degree = 1
        strategy.heter_ccl_mode = True

    hybrid_configs = {
        "dp_degree": dp_degree,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": sharding_degree,
    }
    strategy.hybrid_configs = hybrid_configs

    fleet.init(is_collective=True, strategy=strategy)

    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer)

    paddle.seed(2023)
    np.random.seed(2023)
    train_loader = paddle.io.DataLoader(
        RandomDataset(),
        batch_size=100,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    model.to(device="gpu")

    losses = []
    for eop in range(epoch):
        model.train()

        for batch_id, data in enumerate(train_loader()):
            data.stop_gradient = True

            with paddle.amp.auto_cast(
                True,
                level=level,
                dtype="bfloat16",
                custom_white_list=custom_white_list,
            ):
                out = model(data)
                loss = paddle.mean(out)

            losses.append(loss)

            loss.backward()
            if not accumulate_grad:
                optimizer.step()
                optimizer.clear_grad()

        if accumulate_grad:
            optimizer.step()
            optimizer.clear_grad()

    return losses


def test_stage1_dp():
    if not paddle.amp.is_bfloat16_supported():
        return
    paddle.distributed.init_parallel_env()

    mlp = MLP()
    state_dict = mlp.state_dict()

    # stage1 + dp vs pure stage1
    mlp1 = MLP()
    mlp2 = MLP()
    mlp1.set_state_dict(state_dict)
    mlp2.set_state_dict(state_dict)
    o1_losses = train_mlp(
        mlp1,
        use_sharding_stage1=True,
    )
    o2_losses = train_mlp(mlp2, use_sharding_stage1=True, test_dp=True)
    for i in range(len(o1_losses)):
        o1_loss = o1_losses[i].detach()
        o2_loss = o2_losses[i].detach()
        np.testing.assert_array_equal(o1_loss, o2_loss)

    # stage1 + dp vs pure dp
    mlp3 = MLP()
    mlp4 = MLP()
    mlp3.set_state_dict(state_dict)
    mlp4.set_state_dict(state_dict)
    o1_losses = train_mlp(
        mlp3,
        use_sharding_stage1=True,
        test_dp=True,
    )
    o2_losses = train_mlp(
        mlp4,
        use_sharding_stage1=False,
    )
    for i in range(len(o1_losses)):
        o1_loss = o1_losses[i].detach()
        o2_loss = o2_losses[i].detach()
        np.testing.assert_array_equal(o1_loss, o2_loss)

    return


if __name__ == '__main__':
    test_stage1_dp()
