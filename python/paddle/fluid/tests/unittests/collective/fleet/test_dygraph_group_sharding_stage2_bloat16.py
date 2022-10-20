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
import numpy as np
import tempfile
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.framework import _test_eager_guard

from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import GroupShardedOptimizerStage2
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage2 import GroupShardedStage2
from paddle.distributed.sharding import group_sharded_parallel

seed = 2022
epoch = 2
linear_size = 1000

np.random.seed(seed)
paddle.seed(seed)


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


def optimizer_setting(model, opt_group=False):
    optimizer = paddle.optimizer.Adam(parameters=[{
        "params": model.parameters(),
    }] if opt_group else model.parameters(),
                                      learning_rate=0.001,
                                      weight_decay=0.00001)
    return optimizer


def train_mlp(model,
              sharding_stage,
              batch_size=100,
              use_bfloat16=False,
              opt_group=False):
    optimizer = optimizer_setting(model=model, opt_group=opt_group)
    if use_bfloat16:
        model = paddle.amp.decorate(models=model, level='O2', dtype='bfloat16')

    if sharding_stage != "dp":
        group = paddle.distributed.new_group([0, 1], backend="nccl")

    scaler = paddle.amp.GradScaler(init_loss_scaling=32768.0)

    if sharding_stage == 2:
        model, optimizer, scaler = group_sharded_parallel(model,
                                                          optimizer,
                                                          "os_g",
                                                          scaler=scaler)
    else:
        model = paddle.DataParallel(model)

    train_reader = paddle.batch(reader_decorator(),
                                batch_size=batch_size,
                                drop_last=True)

    train_loader = paddle.io.DataLoader.from_generator(capacity=32,
                                                       use_double_buffer=True,
                                                       iterable=True,
                                                       return_list=True,
                                                       use_multiprocess=True)

    train_loader.set_sample_list_generator(train_reader)

    if sharding_stage == 2:
        model.to(device="gpu")

    for epo in range(epoch):
        model.train()

        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True

            if use_bfloat16:
                with paddle.no_grad():
                    img = paddle.cast(img, dtype=paddle.bfloat16)

            with paddle.amp.auto_cast(use_bfloat16,
                                      level='O2',
                                      dtype='bfloat16'):
                out = model(img)
                loss = paddle.nn.functional.cross_entropy(input=out,
                                                          label=label)
                avg_loss = paddle.mean(loss)
                avg_loss.backward()

            optimizer.step()
            optimizer.clear_grad()

    paddle.device.cuda.synchronize()

    return model.parameters()


def test_dp_stage2():
    paddle.distributed.init_parallel_env()
    mlp = MLP()
    state_dict = mlp.state_dict()
    mlp1 = MLP()
    mlp2 = MLP()
    mlp1.set_state_dict(state_dict)
    mlp2.set_state_dict(state_dict)

    # DP VS stage2
    dp_params = train_mlp(mlp1,
                          sharding_stage="dp",
                          use_bfloat16=False,
                          opt_group=False)
    stage2_params = train_mlp(mlp2,
                              sharding_stage=2,
                              use_bfloat16=True,
                              opt_group=False)
    stage2_params = [
        paddle.cast(param, paddle.float32) for param in stage2_params
    ]
    for i in range(len(dp_params)):
        np.testing.assert_allclose(dp_params[i].numpy(),
                                   stage2_params[i].numpy(),
                                   atol=0)

    return


if __name__ == '__main__':
    with _test_eager_guard():
        test_dp_stage2()
