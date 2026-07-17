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
        self.center_stage = nn.Linear(4096, 4096, bias_attr=False)
        self.final_stage = nn.Linear(4096, 2, bias_attr=False)

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
    enable_tensor_fusion_and_overlap=True,
):
    if use_fsdp:
        model = fully_shard(
            model,
            enable_tensor_fusion_and_overlap=enable_tensor_fusion_and_overlap,
        )
    model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
    optimizer = paddle.optimizer.AdamW(
        learning_rate=0.001,
        parameters=model.parameters(),
        multi_precision=use_pure_bf16,
    )
    optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)

    if not use_fsdp:
        model, optimizer, _ = group_sharded_parallel(
            model=model,
            optimizer=optimizer,
            level="os",
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


class TransformerLayer(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias_attr=False)

    def forward(self, x):
        return self.linear(x)


class SharedLinear(nn.Layer):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return paddle.matmul(x, self.weight)


class TestModel(nn.Layer):
    def __init__(self, hidden_size=16, num_layers=4):
        super().__init__()
        self.input_embeddings = nn.Linear(
            hidden_size, hidden_size, bias_attr=False
        )
        self.layers = nn.LayerList(
            [TransformerLayer(hidden_size) for _ in range(num_layers)]
        )
        self.shared_linear = SharedLinear(self.input_embeddings.weight)
        self.final_stage = nn.Linear(hidden_size, 2, bias_attr=False)

    def get_input_embeddings(self):
        return self.input_embeddings

    def forward(self, x):
        x = self.input_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.shared_linear(x)
        x = self.final_stage(x)
        return x


def test_fsdp_api():
    # test sharding with fsdp api
    paddle.seed(2025)
    np.random.seed(2025)
    paddle.distributed.init_parallel_env()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "sharding_degree": 2,
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)

    data = [paddle.randn([8, 4096]) for i in range(20)]
    sharding_model = Model()
    fsdp_model = Model()
    sharding_model.set_state_dict(fsdp_model.state_dict())

    sharding_loss = train_mlp(sharding_model, use_fsdp=False, data=data)
    fsdp_loss = train_mlp(fsdp_model, use_fsdp=True, data=data)
    assert fsdp_loss == sharding_loss
    # test sharding with fsdp api with fp32 and without overlap and tie_weight
    data = [paddle.randn([8, 16]) for i in range(20)]
    model = TestModel()
    loss = train_mlp(
        model,
        use_fsdp=True,
        data=data,
        use_pure_bf16=False,
        enable_tensor_fusion_and_overlap=False,
    )


if __name__ == '__main__':
    test_fsdp_api()
