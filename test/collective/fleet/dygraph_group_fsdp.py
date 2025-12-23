# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.distributed.fsdp.fully_shard import fully_shard
from paddle.distributed.sharding import group_sharded_parallel


class Model(nn.Layer):
    def __init__(self):
        super().__init__()
        self.first_stage = nn.Linear(4096, 4096, bias_attr=False)
        self.center_stage = nn.Linear(4096, 4096)
        self.center_stage.weight.stop_gradient = True
        self.center_stage.bias.stop_gradient = True
        self.final_stage = nn.Linear(4096, 2, bias_attr=False)
        paddle.distributed.init_parallel_env()
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.hybrid_configs = {
            "sharding_degree": 2,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=dist_strategy)

    def forward(self, x):
        x = self.first_stage(x)
        x = self.center_stage(x)
        x = self.final_stage(x)
        return x


def train_mlp(
    model,
    use_fsdp=True,
    data=None,
    use_pure_bf16=True,
):
    model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
    clip = paddle.nn.ClipGradByGlobalNorm(0.5)
    optimizer = optimizer = paddle.optimizer.AdamW(
        learning_rate=0.001,
        parameters=model.parameters(),
        grad_clip=clip,
        multi_precision=use_pure_bf16,
    )
    optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)

    if use_fsdp:
        model = fully_shard(model)
    else:
        model, optimizer, _ = group_sharded_parallel(
            model=model,
            optimizer=optimizer,
            level="p_g_os",
            sync_buffers=False,
        )

    losses = []
    for i in range(20):
        model.train()
        img = data[i]
        with paddle.amp.auto_cast(level='O1'):
            out = model(img)
            loss = out.mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

    return losses


def test_fsdp_api():
    # test sharding with fsdp api
    paddle.seed(2025)
    np.random.seed(2025)
    data = [paddle.randn([8, 4096]) for i in range(20)]
    model = Model()
    loss_fsdp = train_mlp(model, use_fsdp=True, data=data)

    # test sharding with group_sharded_parallel
    paddle.seed(2025)
    np.random.seed(2025)
    data = [paddle.randn([8, 4096]) for i in range(20)]
    model = Model()
    loss = train_mlp(model, use_fsdp=False, data=data)
    assert loss == loss_fsdp

    # test sharding with fsdp api with fp32
    paddle.seed(2025)
    np.random.seed(2025)
    data = [paddle.randn([8, 4096]) for i in range(20)]
    model = Model()
    loss = train_mlp(model, use_fsdp=True, data=data, use_pure_bf16=False)


if __name__ == '__main__':
    test_fsdp_api()
