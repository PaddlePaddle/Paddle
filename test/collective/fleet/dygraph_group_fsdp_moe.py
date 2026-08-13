# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.distributed.fsdp.fully_shard import fully_shard

HIDDEN = 16
INTER = 32
NUM_EXPERTS = 4
NUM_LAYERS = 2
STEPS = 10
TOKENS = 8
LEARNING_RATE = 0.001


class EPAllGather(PyLayer):
    @staticmethod
    def forward(ctx, x, group=None):
        ctx.group = group
        parts = []
        paddle.distributed.all_gather(parts, x, group=group)
        return paddle.concat(parts, axis=0)

    @staticmethod
    def backward(ctx, dy):
        group = ctx.group
        out = paddle.empty(
            [dy.shape[0] // group.nranks, *dy.shape[1:]], dtype=dy.dtype
        )
        paddle.distributed.reduce_scatter(
            out, dy, op=paddle.distributed.ReduceOp.SUM, group=group
        )
        return out


class EPReduceScatter(PyLayer):
    @staticmethod
    def forward(ctx, x, group=None):
        ctx.group = group
        out = paddle.empty(
            [x.shape[0] // group.nranks, *x.shape[1:]], dtype=x.dtype
        )
        paddle.distributed.reduce_scatter(
            out, x, op=paddle.distributed.ReduceOp.SUM, group=group
        )
        return out

    @staticmethod
    def backward(ctx, dy):
        parts = []
        paddle.distributed.all_gather(parts, dy, group=ctx.group)
        return paddle.concat(parts, axis=0)


class StandardMLPExpert(nn.Layer):
    def __init__(self, hidden, inter):
        super().__init__()
        self.up_proj = nn.Linear(hidden, inter, bias_attr=False)
        self.down_proj = nn.Linear(inter, hidden, bias_attr=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.up_proj(x)))


class MoEBlock(nn.Layer):
    def __init__(self, hidden, inter, num_experts, ep_group, ep_rank):
        super().__init__()
        self.ep_group = ep_group
        self.gate = nn.Linear(hidden, num_experts, bias_attr=False)
        assert num_experts % ep_group.nranks == 0
        per_rank = num_experts // ep_group.nranks
        self.local_ids = list(
            range(ep_rank * per_rank, (ep_rank + 1) * per_rank)
        )
        self.experts = nn.LayerList(
            [StandardMLPExpert(hidden, inter) for _ in self.local_ids]
        )

    def forward(self, x):
        tokens = EPAllGather.apply(x, group=self.ep_group)
        logits = self.gate(tokens)
        weights = F.softmax(logits, axis=-1)
        choice = paddle.argmax(logits, axis=-1)

        out = paddle.zeros_like(tokens)
        for slot, expert_id in enumerate(self.local_ids):
            mask = choice == expert_id
            if not bool(mask.any()):
                continue
            idx = paddle.nonzero(mask).flatten()
            picked = paddle.gather(tokens, idx, axis=0)
            expert_out = self.experts[slot](picked) * paddle.gather(
                weights[:, expert_id : expert_id + 1], idx, axis=0
            )
            out = paddle.scatter(out, idx, expert_out, overwrite=False)
        return EPReduceScatter.apply(out, group=self.ep_group)


class TransformerLayer(nn.Layer):
    def __init__(self, hidden, inter, num_experts, ep_group, ep_rank):
        super().__init__()
        self.attn = nn.Linear(hidden, hidden, bias_attr=False)
        self.moe = MoEBlock(hidden, inter, num_experts, ep_group, ep_rank)

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.moe(x)


class MoEModel(nn.Layer):
    def __init__(self, ep_group, ep_rank):
        super().__init__()
        self.embed = nn.Linear(HIDDEN, HIDDEN, bias_attr=False)
        self.layers = nn.LayerList(
            [
                TransformerLayer(HIDDEN, INTER, NUM_EXPERTS, ep_group, ep_rank)
                for _ in range(NUM_LAYERS)
            ]
        )
        self.head = nn.Linear(HIDDEN, 2, bias_attr=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def init_dist():
    world_size = paddle.distributed.get_world_size()
    assert world_size in (2, 4), (
        "bitwise loss comparison runs on 2 cards (moe_sharding_degree=1) "
        "or 4 cards (moe_sharding_degree=2)"
    )
    ep_degree = 2
    moe_sharding_degree = world_size // ep_degree

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "order": [
            "dp",
            "pp",
            "moe_sharding",
            "ep",
            "sharding",
            "sep",
            "cp",
            "mp",
        ],
        "sharding_degree": world_size,
        "ep_degree": ep_degree,
        "moe_sharding_degree": moe_sharding_degree,
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
    }
    sharding = strategy.hybrid_configs["sharding_configs"]
    sharding.split_param = True
    sharding.tensor_fusion = False
    sharding.comm_overlap = False
    sharding.use_reduce_avg = False
    fleet.init(is_collective=True, strategy=strategy)


def tag_experts(model, group):
    for layer in model.layers:
        for param in layer.moe.experts.parameters():
            param.is_moe_param = True
            param.expert = True
            param.color = {"color": "moe_expert", "group": group}


def build_models(hcg):
    ep_group = hcg.get_expert_parallel_group()
    ep_rank = hcg.get_expert_parallel_rank()

    paddle.seed(2026)
    stage1_model = MoEModel(ep_group, ep_rank)
    paddle.seed(2026)
    fsdp_model = MoEModel(ep_group, ep_rank)
    stage1_model.set_state_dict(fsdp_model.state_dict())

    expert_group = hcg.get_moe_sharding_parallel_group()
    tag_experts(stage1_model, expert_group)
    tag_experts(fsdp_model, expert_group)
    return stage1_model, fsdp_model


def build_optimizer(model):
    optimizer = paddle.optimizer.AdamW(
        learning_rate=LEARNING_RATE,
        parameters=model.parameters(),
        weight_decay=0.0,
        multi_precision=True,
    )
    return mix_precision_utils.MixPrecisionOptimizer(optimizer)


def train_stage1(model, data):
    model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
    optimizer = build_optimizer(model)
    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer)
    return train(model, optimizer, data)


def train_fsdp(model, data, enable_overlap):
    model = fully_shard(
        model,
        enable_tensor_fusion_and_overlap=enable_overlap,
    )
    model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
    optimizer = build_optimizer(model)
    return train(model, optimizer, data)


def train(model, optimizer, data):
    loss_md5s = []
    for x in data:
        model.train()
        with paddle.amp.auto_cast(level="O1", dtype="bfloat16"):
            loss = model(x).mean()
        loss_md5s.append(loss._md5sum())
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
    return loss_md5s


def run_loss_comparison(hcg):
    stage1_model, fsdp_model = build_models(hcg)

    paddle.seed(2026)
    data = [paddle.randn([TOKENS, HIDDEN]) for _ in range(STEPS)]

    stage1_loss_md5s = train_stage1(stage1_model, data)
    enable_overlap = hcg.get_moe_sharding_parallel_world_size() == 1
    fsdp_loss_md5s = train_fsdp(fsdp_model, data, enable_overlap)

    assert fsdp_loss_md5s == stage1_loss_md5s, (
        "10-step loss MD5 sequence diverged: "
        f"stage1={stage1_loss_md5s}, fsdp={fsdp_loss_md5s}"
    )


def run():
    paddle.distributed.init_parallel_env()
    init_dist()
    hcg = fleet.get_hybrid_communicate_group()
    # 4 cards => moe_sharding_degree=2, expert grads really go through reduce_scatter
    assert (
        hcg.get_moe_sharding_parallel_world_size()
        == paddle.distributed.get_world_size() // 2
    )
    run_loss_comparison(hcg)


if __name__ == "__main__":
    run()
