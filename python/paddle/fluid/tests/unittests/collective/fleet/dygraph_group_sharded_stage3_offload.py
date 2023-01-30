# -*- coding: UTF-8 -*-

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
<<<<<<< HEAD

import paddle
import paddle.fluid as fluid
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage3 import (
    GroupShardedStage3,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import (
    GroupShardedScaler,
)
from paddle.nn import Linear
=======
import argparse
import ast
import time
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
from paddle.distributed import fleet
from paddle.fluid.dygraph import nn
from paddle.fluid.framework import _test_eager_guard

from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage3 import GroupShardedStage3
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import GroupShardedScaler
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

epoch = 10
paddle.seed(2022)
np.random.seed(2022)
base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4


class MLP(fluid.Layer):
<<<<<<< HEAD
    def __init__(self, linear_size=1000, param_attr=None, bias_attr=None):
        super().__init__()

        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
        # test for trainable & untrainable offload
        self._linear2.weight.stop_gradient = False
        self._linear2.bias.stop_gradient = False
=======

    def __init__(self, linear_size=1000, param_attr=None, bias_attr=None):
        super(MLP, self).__init__()

        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._linear3 = Linear(linear_size, 10)

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        return y


<<<<<<< HEAD
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples=2000, linear_size=1000):
        self.num_samples = num_samples
        self.linear_size = linear_size

    def __getitem__(self, idx):
        img = np.random.rand(self.linear_size).astype('float32')
        label = np.ones(1).astype('int64')
        return img, label

    def __len__(self):
        return self.num_samples
=======
def reader_decorator(linear_size=1000):

    def __reader__():
        for _ in range(100):
            img = np.random.rand(linear_size).astype('float32')
            label = np.ones(1).astype('int64')
            yield img, label

    return __reader__
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def optimizer_setting(model, use_pure_fp16, opt_group=False):
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
<<<<<<< HEAD
    optimizer = paddle.optimizer.AdamW(
        parameters=[{"params": model.parameters()}]
        if opt_group
        else model.parameters(),
        learning_rate=0.001,
        weight_decay=0.00001,
        grad_clip=clip,
        multi_precision=use_pure_fp16,
    )
=======
    optimizer = paddle.optimizer.AdamW(parameters=[{
        "params": model.parameters()
    }] if opt_group else model.parameters(),
                                       learning_rate=0.001,
                                       weight_decay=0.00001,
                                       grad_clip=clip,
                                       multi_precision=use_pure_fp16)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    return optimizer


<<<<<<< HEAD
def train_mlp(
    model,
    use_pure_fp16=False,
    accumulate_grad=False,
    offload=False,
    batch_size=100,
    convert2cpu=False,
):
=======
def train_mlp(model,
              use_pure_fp16=False,
              accumulate_grad=False,
              offload=False,
              batch_size=100,
              convert2cpu=False):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    group = paddle.distributed.new_group([0, 1])
    optimizer = optimizer_setting(model=model, use_pure_fp16=use_pure_fp16)

    if use_pure_fp16:
<<<<<<< HEAD
        model = paddle.amp.decorate(
            models=model, level='O2', save_dtype='float32'
        )
        scaler = paddle.amp.GradScaler(init_loss_scaling=32768)
        scaler = GroupShardedScaler(scaler)

    model = GroupShardedStage3(
        model,
        optimizer=optimizer,
        group=group,
        offload=offload,
        segment_size=2**15,
    )

    paddle.seed(2023)
    np.random.seed(2023)
    train_loader = paddle.io.DataLoader(
        RandomDataset(),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
=======
        model = paddle.amp.decorate(models=model,
                                    level='O2',
                                    save_dtype='float32')
        scaler = paddle.amp.GradScaler(init_loss_scaling=32768)
        scaler = GroupShardedScaler(scaler)

    model = GroupShardedStage3(model,
                               optimizer=optimizer,
                               group=group,
                               offload=offload,
                               segment_size=2**15)

    train_reader = paddle.batch(reader_decorator(),
                                batch_size=batch_size,
                                drop_last=True)

    train_loader = paddle.io.DataLoader.from_generator(capacity=32,
                                                       use_double_buffer=True,
                                                       iterable=True,
                                                       return_list=True,
                                                       use_multiprocess=True)
    train_loader.set_sample_list_generator(train_reader)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    for eop in range(epoch):
        model.train()
        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True
<<<<<<< HEAD
            with paddle.amp.auto_cast(use_pure_fp16, level='O2'):
                out = model(img)
                loss = paddle.nn.functional.cross_entropy(
                    input=out, label=label
                )
=======
            with paddle.amp.auto_cast(True, level='O2'):
                out = model(img)
                loss = paddle.nn.functional.cross_entropy(input=out,
                                                          label=label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            avg_loss = paddle.mean(x=loss.cast(dtype=paddle.float32))

            if accumulate_grad:
                avg_loss = avg_loss / 5

            if not use_pure_fp16:
                avg_loss.backward()
            else:
                scaler.scale(avg_loss).backward()

            if not accumulate_grad:
                if not use_pure_fp16:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.clear_grad()
        if accumulate_grad:
            if not use_pure_fp16:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.clear_grad()
    if not convert2cpu:
        model.get_all_parameters()
    else:
        model.get_all_parameters(convert2cpu)
    return model.parameters()


def test_stage3_offload():
    paddle.distributed.init_parallel_env()
<<<<<<< HEAD
    mlp, mlp1, mlp2, mlp3, mlp4, mlp5, mlp6 = (
        MLP(),
        MLP(),
        MLP(),
        MLP(),
        MLP(),
        MLP(),
        MLP(),
    )
=======
    mlp, mlp1, mlp2, mlp3, mlp4, mlp5, mlp6 = MLP(), MLP(), MLP(), MLP(), MLP(
    ), MLP(), MLP()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    state_dict = mlp.state_dict()
    mlp1.set_state_dict(state_dict)
    mlp2.set_state_dict(state_dict)
    mlp3.set_state_dict(state_dict)
    mlp4.set_state_dict(state_dict)
    mlp5.set_state_dict(state_dict)
    mlp6.set_state_dict(state_dict)

    # fp32 offload
    stage3_params = train_mlp(mlp1, use_pure_fp16=False)
    stage3_params_offload = train_mlp(mlp2, use_pure_fp16=False, offload=True)
    for i in range(len(stage3_params)):
<<<<<<< HEAD
        np.testing.assert_allclose(
            stage3_params[i].numpy(),
            stage3_params_offload[i].numpy(),
            rtol=1e-6,
            atol=1e-8,
        )
=======
        np.testing.assert_allclose(stage3_params[i].numpy(),
                                   stage3_params_offload[i].numpy(),
                                   rtol=1e-6,
                                   atol=1e-8)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # fp16 offload
    stage3_params = train_mlp(mlp3, use_pure_fp16=True)
    stage3_params_offload = train_mlp(mlp4, use_pure_fp16=True, offload=True)
    for i in range(len(stage3_params)):
<<<<<<< HEAD
        np.testing.assert_allclose(
            stage3_params[i].numpy(),
            stage3_params_offload[i].numpy(),
            rtol=1e-2,
            atol=1e-2,
        )

    # fp32 accumulate grad offload
    stage3_params = train_mlp(
        mlp5, use_pure_fp16=False, batch_size=20, accumulate_grad=True
    )
    stage3_params_offload = train_mlp(
        mlp6,
        use_pure_fp16=False,
        accumulate_grad=True,
        offload=True,
        batch_size=20,
        convert2cpu=True,
    )
    for i in range(len(stage3_params)):
        np.testing.assert_allclose(
            stage3_params[i].numpy(),
            stage3_params_offload[i].numpy(),
            rtol=1e-6,
            atol=1e-8,
        )
=======
        np.testing.assert_allclose(stage3_params[i].numpy(),
                                   stage3_params_offload[i].numpy(),
                                   rtol=1e-2,
                                   atol=1e-2)

    # fp32 accumulate grad offload
    stage3_params = train_mlp(mlp5,
                              use_pure_fp16=False,
                              batch_size=20,
                              accumulate_grad=True)
    stage3_params_offload = train_mlp(mlp6,
                                      use_pure_fp16=False,
                                      accumulate_grad=True,
                                      offload=True,
                                      batch_size=20,
                                      convert2cpu=True)
    for i in range(len(stage3_params)):
        np.testing.assert_allclose(stage3_params[i].numpy(),
                                   stage3_params_offload[i].numpy(),
                                   rtol=1e-6,
                                   atol=1e-8)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    return


if __name__ == '__main__':
<<<<<<< HEAD
    test_stage3_offload()
=======
    with _test_eager_guard():
        test_stage3_offload()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
