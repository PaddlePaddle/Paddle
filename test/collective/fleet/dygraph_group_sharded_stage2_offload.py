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
from dygraph_group_sharded_stage2 import MLP, RandomDataset, optimizer_setting

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

seed = 2021
epoch = 2
batch_size = 32
linear_size = 1000

np.random.seed(seed)
paddle.seed(seed)


def train_mlp(model, offload=False, test=False):
    optimizer = optimizer_setting(model=model, use_pure_fp16=True)

    model = paddle.amp.decorate(models=model, level='O2', save_dtype='float32')
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    scaler = GroupShardedScaler(scaler)

    dp_group = (
        None
        if not test
        else paddle.distributed.new_group(
            list(range(paddle.distributed.get_world_size()))
        )
    )
    optimizer = GroupShardedOptimizerStage2(
        params=optimizer._parameter_list,
        optim=optimizer,
        offload=offload,
        dp_group=dp_group,
    )
    model = GroupShardedStage2(
        model, optimizer, buffer_max_size=2**21, dp_group=dp_group
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

    for eop in range(epoch):
        model.train()

        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True

            with paddle.amp.auto_cast(True, level='O2'):
                out = model(img)
                loss = paddle.nn.functional.cross_entropy(
                    input=out, label=label
                )

            avg_loss = paddle.mean(x=loss.cast(dtype=paddle.float32))
            scaler.scale(avg_loss).backward()

            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad()

    for dtype in optimizer.param_storages:
        for dst_rank, param_storage in optimizer.param_storages[dtype].items():
            param_storage.to(
                device="xpu" if paddle.is_compiled_with_xpu() else "gpu",
                dtype=dtype,
            )

    return model.parameters()


def test_sharding_stage2_offload():
    paddle.distributed.init_parallel_env()
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
            atol=5e-3,
        )

    # just to test assert error for the rate of coverage
    try:
        train_mlp(mlp_offload, offload=True, test=True)
    except Exception as e:
        assert isinstance(e, AssertionError)


if __name__ == '__main__':
    test_sharding_stage2_offload()
