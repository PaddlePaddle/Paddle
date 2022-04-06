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

import os
import shutil
import tempfile
import numpy as np
import argparse
import ast
import time
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
from paddle.distributed import fleet
from paddle.fluid.dygraph import nn
from paddle.fluid.framework import _test_eager_guard

from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import GroupShardedOptimizerStage2
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage2 import GroupShardedStage2
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage3 import GroupShardedStage3
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import GroupShardedScaler

epoch = 10
paddle.seed(2022)
np.random.seed(2022)
base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4


class MLP(fluid.Layer):
    def __init__(self, linear_size=1000, param_attr=None, bias_attr=None):
        super(MLP, self).__init__()

        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
        self._linear3 = Linear(linear_size, 10)

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        return y


def reader_decorator(linear_size=1000):
    def __reader__():
        for _ in range(100):
            img = np.random.rand(linear_size).astype('float32')
            label = np.ones(1).astype('int64')
            yield img, label

    return __reader__


def optimizer_setting(model, use_pure_fp16, opt_group=False):
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.Momentum(
        parameters=[{
            "params": list(model.parameters())
        }] if opt_group else list(model.parameters()),
        learning_rate=0.001,
        weight_decay=0.00001,
        grad_clip=clip,
        multi_precision=use_pure_fp16)

    return optimizer


def train_mlp(model,
              sharding_stage,
              use_pure_fp16=False,
              accumulate_grad=False,
              batch_size=100,
              opt_group=False,
              sync_comm=False,
              test_minimize=False,
              save_model=False):
    group = paddle.distributed.new_group([0, 1])
    if opt_group:
        optimizer = optimizer_setting(
            model=model, use_pure_fp16=use_pure_fp16, opt_group=opt_group)
    else:
        optimizer = optimizer_setting(model=model, use_pure_fp16=use_pure_fp16)

    if use_pure_fp16:
        model = paddle.amp.decorate(
            models=model, level='O2', save_dtype='float32')
        scaler = paddle.amp.GradScaler(init_loss_scaling=32768)
        scaler = GroupShardedScaler(scaler)
    if sharding_stage == 2:
        optimizer = GroupShardedOptimizerStage2(
            params=optimizer._parameter_list, optim=optimizer, group=group)
        model = GroupShardedStage2(
            model, optimizer, group=group, buffer_max_size=2**21)
    elif sharding_stage == 3:
        model = GroupShardedStage3(
            model,
            optimizer=optimizer,
            group=group,
            sync_comm=sync_comm,
            segment_size=2**15)

    # check optimizer.minimize() error
    if test_minimize:
        try:
            optimizer.minimize()
        except:
            print(
                "====== Find sharding_stage3_optimizer.minimize() error ======")
        return

    train_reader = paddle.batch(
        reader_decorator(), batch_size=batch_size, drop_last=True)

    train_loader = paddle.io.DataLoader.from_generator(
        capacity=32,
        use_double_buffer=True,
        iterable=True,
        return_list=True,
        use_multiprocess=True)
    train_loader.set_sample_list_generator(train_reader)

    for eop in range(epoch):
        model.train()
        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True
            with paddle.amp.auto_cast(True, level='O2'):
                out = model(img)
                loss = paddle.nn.functional.cross_entropy(
                    input=out, label=label)
            avg_loss = paddle.mean(x=loss.cast(dtype=paddle.float32))

            if batch_size == 20:
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
    if sharding_stage == 3:
        model.get_all_parameters()

    if save_model:
        return model, optimizer
    return model.parameters()


def test_stage2_stage3():
    paddle.distributed.init_parallel_env()
    mlp, mlp1, mlp2, mlp3, mlp4, mlp5, mlp6, mlp7, mlp8, mlp9, mlp10 = MLP(
    ), MLP(), MLP(), MLP(), MLP(), MLP(), MLP(), MLP(), MLP(), MLP(), MLP()
    state_dict = mlp.state_dict()
    mlp1.set_state_dict(state_dict)
    mlp2.set_state_dict(state_dict)
    mlp3.set_state_dict(state_dict)
    mlp4.set_state_dict(state_dict)
    mlp5.set_state_dict(state_dict)
    mlp6.set_state_dict(state_dict)
    mlp7.set_state_dict(state_dict)
    mlp8.set_state_dict(state_dict)
    mlp9.set_state_dict(state_dict)
    mlp10.set_state_dict(state_dict)

    # fp32 
    stage2_params = train_mlp(
        mlp1, sharding_stage=2, use_pure_fp16=False, opt_group=False)
    stage3_params = train_mlp(
        mlp2, sharding_stage=3, use_pure_fp16=False, opt_group=False)

    for i in range(len(stage2_params)):
        np.testing.assert_allclose(
            stage2_params[i].numpy(),
            stage3_params[i].numpy(),
            rtol=1e-6,
            atol=1e-6)

    # fp32 accumulate grad
    stage3_params = train_mlp(
        mlp3,
        sharding_stage=3,
        use_pure_fp16=False,
        accumulate_grad=True,
        opt_group=True)
    stage3_params_add = train_mlp(
        mlp4,
        sharding_stage=3,
        use_pure_fp16=False,
        accumulate_grad=True,
        batch_size=20,
        opt_group=True)
    for i in range(len(stage3_params)):
        np.testing.assert_allclose(
            stage3_params[i].numpy(),
            stage3_params_add[i].numpy(),
            rtol=1e-6,
            atol=1e-4)

    # fp16
    stage2_params = train_mlp(
        mlp5, sharding_stage=2, use_pure_fp16=True, opt_group=False)
    stage3_params = train_mlp(
        mlp6, sharding_stage=3, use_pure_fp16=True, opt_group=False)
    for i in range(len(stage2_params)):
        np.testing.assert_allclose(
            stage2_params[i].numpy(),
            stage3_params[i].numpy(),
            rtol=1e-4,
            atol=1e-3)

    # fp16 sync_comm
    stage3_params = train_mlp(
        mlp7, sharding_stage=3, use_pure_fp16=True, opt_group=False)
    stage3_params_re = train_mlp(
        mlp8,
        sharding_stage=3,
        use_pure_fp16=True,
        opt_group=False,
        sync_comm=True)
    for i in range(len(stage3_params)):
        np.testing.assert_allclose(
            stage3_params[i].numpy(), stage3_params_re[i].numpy(), rtol=1e-6)

    # save/load model
    output_dir = tempfile.mkdtemp()
    model_file = os.path.join(output_dir, "model.pdmodel")
    optimizer_file = os.path.join(output_dir, "model.pdopt")
    model_stage3, optimizer_stage3 = train_mlp(
        mlp9,
        sharding_stage=3,
        use_pure_fp16=False,
        opt_group=False,
        save_model=True)
    paddle.save(model_stage3.state_dict(), model_file)
    paddle.save(optimizer_stage3.state_dict(), optimizer_file)
    m_state_dict = paddle.load(model_file)
    opt_state_dict = paddle.load(optimizer_file)
    model_stage3.set_state_dict(m_state_dict)
    optimizer_stage3.set_state_dict(opt_state_dict)
    shutil.rmtree(output_dir)

    # check optimizer.minimize() error
    train_mlp(
        mlp10,
        sharding_stage=3,
        use_pure_fp16=False,
        opt_group=False,
        test_minimize=True)


if __name__ == '__main__':
    with _test_eager_guard():
        test_stage2_stage3()
