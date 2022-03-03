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

from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.sharding_optimizer_stage2 import ShardingOptimizerStage2
from paddle.distributed.fleet.meta_parallel.sharding.sharding_stage2 import ShardingStage2
from paddle.distributed.fleet.meta_parallel.sharding.sharding_utils import ShardingScaler

from dygraph_sharding_stage2 import MLP, reader_decorator, optimizer_setting

seed = 2021
epoch = 2
batch_size = 32
linear_size = 1000

np.random.seed(seed)
paddle.seed(seed)


def train_mlp(model, offload=False):
    optimizer = optimizer_setting(model=model, use_pure_fp16=True)

    model = paddle.amp.decorate(models=model, level='O2', save_dtype='float32')
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    scaler = ShardingScaler(scaler)

    optimizer = ShardingOptimizerStage2(
        params=model.parameters(), optim=optimizer, offload=offload)
    model = ShardingStage2(model, optimizer, buffer_max_size=2**21)

    train_reader = paddle.batch(
        reader_decorator(linear_size), batch_size=batch_size, drop_last=True)

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
            scaler.scale(avg_loss).backward()

            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad()

    for dtype in optimizer.param_storages:
        for dst_rank, param_storage in optimizer.param_storages[dtype].items():
            param_storage.to(device="gpu", dtype=dtype)

    return model.parameters()


def test_sharding_stage2_offload():
    mlp = MLP(linear_size)
    mlp_offload = MLP(linear_size)
    mlp_offload.set_state_dict(mlp.state_dict())

    mlp_params = train_mlp(mlp, offload=False)
    mlp_offload_params = train_mlp(mlp_offload, offload=True)

    for i in range(len(mlp_params)):
        np.testing.assert_allclose(
            mlp_params[i].numpy(),
            mlp_offload_params[i].numpy(),
            rtol=5e-3,
            atol=5e-3)
    return


if __name__ == '__main__':
    test_sharding_stage2_offload()
