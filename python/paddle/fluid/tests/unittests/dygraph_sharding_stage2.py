# -*- coding: UTF-8 -*-

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import ast
import time
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
from paddle.distributed import fleet
from paddle.fluid.dygraph import nn

from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import DygraphShardingOptimizer
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.sharding_optimizer_stage2 import ShardingOptimizerStage2
from paddle.distributed.fleet.meta_parallel.sharding.sharding_stage2 import ShardingStage2

seed = 2021
epoch = 2
batch_size = 32

strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {
    "dp_degree": 2,
    "mp_degree": 1,
    "pp_degree": 1,
    "sharding_degree": 1
}
fleet.init(is_collective=True, strategy=strategy)

np.random.seed(seed)
paddle.seed(seed)


class MLP(fluid.Layer):
    def __init__(self, param_attr=None, bias_attr=None):
        super(MLP, self).__init__()

        self._linear1 = Linear(10000, 10000)
        self._linear2 = Linear(10000, 10000)
        self._linear3 = Linear(10000, 10)

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        return y


def reader_decorator():
    def __reader__():
        for _ in range(100):
            img = np.random.rand(10000).astype('float32')
            label = np.ones(1).astype('int64')
            yield img, label

    return __reader__


def optimizer_setting(model, use_pure_fp16, stage=1):
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=0.001,
        weight_decay=0.00001,
        grad_clip=clip,
        multi_precision=use_pure_fp16)

    return optimizer


def train_mlp(model,
              sharding_stage,
              use_pure_fp16=False,
              all_test=False,
              accumulate_grad=False):
    if sharding_stage == 1:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_check_parallel_group()
    else:
        group = paddle.distributed.new_group([0, 1])
    optimizer = optimizer_setting(
        model=model, use_pure_fp16=use_pure_fp16, stage=sharding_stage)

    if use_pure_fp16:
        model, optimizer = paddle.amp.decorate(
            models=model,
            optimizers=optimizer,
            level='O2',
            save_dtype='float32')

    if sharding_stage == 2:
        optimizer = ShardingOptimizerStage2(
            params=model.parameters(), optim=optimizer, group=group)
        if all_test:
            model = ShardingStage2(
                model, optimizer, group=group, accumulate_grads=accumulate_grad)
        else:
            model = ShardingStage2(model, optimizer, group=group)
    else:
        optimizer = fleet.distributed_optimizer(optimizer)
        model = fleet.distributed_model(model)

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

            with paddle.amp.auto_cast(enable=use_pure_fp16, level='O2'):
                out = model(img)
                loss = paddle.nn.functional.cross_entropy(
                    input=out, label=label)

            avg_loss = paddle.mean(x=loss.cast(dtype=paddle.float32))
            avg_loss.backward()

            if accumulate_grad and batch_id == 2:
                model.grad_scale()
                optimizer.step()
                model.clear_gradients()
                return model.parameters()

            if not accumulate_grad:
                optimizer.step()

                if sharding_stage == 2:
                    model.clear_gradients()
                else:
                    optimizer.clear_grad()

            if all_test and batch_id == 2:
                return model.parameters()

    if sharding_stage == 2:
        model.to(device="gpu")

    return model.parameters()


def test_stage1_stage2():
    mlp = MLP()
    state_dict = mlp.state_dict()
    mlp1 = MLP()
    mlp2 = MLP()
    mlp3 = MLP()
    mlp4 = MLP()
    mlp1.set_state_dict(state_dict)
    mlp2.set_state_dict(state_dict)
    mlp3.set_state_dict(state_dict)
    mlp4.set_state_dict(state_dict)
    stage1_params = train_mlp(mlp, sharding_stage=1, use_pure_fp16=False)
    stage2_params = train_mlp(mlp, sharding_stage=2, use_pure_fp16=False)
    for i in range(len(stage1_params)):
        np.testing.assert_allclose(
            stage1_params[i].numpy(), stage2_params[i].numpy(), rtol=1e-6)

    stage2_params = train_mlp(
        mlp3, sharding_stage=2, use_pure_fp16=True, all_test=True)
    stage2_accumulate_grad = train_mlp(
        mlp4,
        sharding_stage=2,
        use_pure_fp16=True,
        all_test=True,
        accumulate_grad=True)
    for i in range(len(stage2_params)):
        for j in range(len(stage2_accumulate_grad)):
            if stage2_params[i].name == stage2_accumulate_grad[j].name:
                np.testing.assert_allclose(
                    stage2_params[i].numpy(),
                    stage2_accumulate_grad[j].numpy(),
                    rtol=1e-6)

    return


if __name__ == '__main__':
    test_stage1_stage2()
