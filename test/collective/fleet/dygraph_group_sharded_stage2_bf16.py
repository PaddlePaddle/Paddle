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
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import (
    GroupShardedOptimizerStage2,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage2 import (
    GroupShardedStage2,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import (
    GroupShardedScaler,
)
from paddle.distributed.fleet.utils import mix_precision_utils
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


def optimizer_setting(model, use_pure_bf16, use_main_grad):
    if use_main_grad:
        assert use_pure_bf16
        model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=0.00001,
        weight_decay=0.00001,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
        multi_precision=use_pure_bf16,
    )
    if use_main_grad:
        optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)

    return optimizer


def train_mlp(
    model,
    sharding_stage,
    use_pure_bf16=False,
    accumulate_grad=False,
    use_main_grad=False,
    test_scaler=False,
    use_storage=True,
):
    if sharding_stage != "dp":
        group = paddle.distributed.new_group([0, 1], backend="nccl")
    scaler = None
    if test_scaler:
        assert sharding_stage == 2
        assert not accumulate_grad
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        scaler = GroupShardedScaler(scaler)
    optimizer = optimizer_setting(
        model=model, use_pure_bf16=use_pure_bf16, use_main_grad=use_main_grad
    )
    if use_pure_bf16:
        level = 'O2'
        custom_white_list = None
        model = paddle.amp.decorate(models=model, dtype="bfloat16", level=level)
    else:
        level = 'O1'
        custom_white_list = [
            "matmul_v2",
            "elementwise_add",
            "relu",
            "reduce_mean",
        ]

    if sharding_stage == 2:
        optimizer = GroupShardedOptimizerStage2(
            params=optimizer._parameter_list, optim=optimizer, group=group
        )
        if use_storage:
            buffer_max_size = 2**21
        else:
            buffer_max_size = 0
        model = GroupShardedStage2(
            model, optimizer, group=group, buffer_max_size=buffer_max_size
        )
    else:
        model = paddle.DataParallel(model)

    paddle.seed(2023)
    np.random.seed(2023)
    train_loader = paddle.io.DataLoader(
        RandomDataset(),
        batch_size=100,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    if sharding_stage == 2:
        model.to(device="gpu")

    if not use_pure_bf16:
        for param in model.parameters():
            t = paddle.cast(
                paddle.cast(param, dtype='bfloat16'), dtype='float32'
            )
            param.set_value(t)

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

            if test_scaler:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.clear_grad()
            else:
                loss.backward()
                if not accumulate_grad:
                    optimizer.step()
                    optimizer.clear_grad()

        if accumulate_grad:
            optimizer.step()
            optimizer.clear_grad()

    return losses


def test_stage2_bf16():
    if not paddle.amp.is_bfloat16_supported():
        return
    paddle.distributed.init_parallel_env()
    mlp = MLP()
    state_dict = mlp.state_dict()

    # stage2 bf16 O1 vs stage2 bf16 O2 main_grad
    mlp1 = MLP()
    mlp2 = MLP()
    mlp1.set_state_dict(state_dict)
    mlp2.set_state_dict(state_dict)
    o1_losses = train_mlp(mlp1, sharding_stage=2, use_pure_bf16=False)
    o2_losses = train_mlp(
        mlp2, sharding_stage=2, use_pure_bf16=True, use_main_grad=True
    )
    for i in range(len(o1_losses)):
        o1_32_loss = paddle.cast(o1_losses[i], dtype='float32').detach()
        o2_32_loss = paddle.cast(o2_losses[i], dtype='float32').detach()
        np.testing.assert_array_equal(o1_32_loss, o2_32_loss)

    # stage2 scaler test
    mlp3 = MLP()
    mlp3.set_state_dict(state_dict)
    train_mlp(
        mlp3,
        sharding_stage=2,
        use_pure_bf16=True,
        use_main_grad=True,
        test_scaler=True,
    )

    # not fuse grad test
    mlp4 = MLP()
    mlp4.set_state_dict(state_dict)
    o2_losses_no_storage = train_mlp(
        mlp4,
        sharding_stage=2,
        use_pure_bf16=True,
        use_main_grad=True,
        use_storage=False,
    )
    for i in range(len(o2_losses_no_storage)):
        o2_loss_no_storage = paddle.cast(
            o2_losses_no_storage[i], dtype='float32'
        ).detach()
        o2_32_loss = paddle.cast(o2_losses[i], dtype='float32').detach()
        np.testing.assert_array_equal(o2_loss_no_storage, o2_32_loss)

    # grad accumulation test
    mlp5 = MLP()
    mlp6 = MLP()
    mlp5.set_state_dict(state_dict)
    mlp6.set_state_dict(state_dict)
    o1_losses_grad_acc = train_mlp(
        mlp5, sharding_stage=2, use_pure_bf16=False, accumulate_grad=True
    )
    o2_losses_grad_acc = train_mlp(
        mlp6,
        sharding_stage=2,
        use_pure_bf16=True,
        use_main_grad=True,
        accumulate_grad=True,
    )
    for i in range(len(o2_losses_grad_acc)):
        o2_loss_grad_acc = paddle.cast(
            o2_losses_grad_acc[i], dtype='float32'
        ).detach()
        o1_loss_grad_acc = paddle.cast(
            o1_losses_grad_acc[i], dtype='float32'
        ).detach()
        np.testing.assert_array_equal(o2_loss_grad_acc, o1_loss_grad_acc)

    return


if __name__ == '__main__':
    test_stage2_bf16()
