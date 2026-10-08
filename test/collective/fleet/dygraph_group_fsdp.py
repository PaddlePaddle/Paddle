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
import paddle.nn.functional as F
from paddle import nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.distributed.fsdp import fully_shard_fusion
from paddle.distributed.fsdp.fully_shard import fully_shard
from paddle.optimizer.muon import MuonParamInfo

HIDDEN = 16
INTER = 32
NUM_EXPERTS = 4
NUM_LAYERS = 6
STEPS = 5
TOKENS = 64
ACCUM_STEPS = 2
MUON_MATS = 4
MUON_LAYERS = 2
MUON_STEPS = 1


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


class SharedLinear(nn.Layer):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return paddle.matmul(x, self.weight)


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
        self.scale = self.create_parameter(
            shape=[HIDDEN],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.scale.stop_gradient = True
        self.shared_head = SharedLinear(self.embed.weight)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.shared_head(x)
        return self.head(x)


def init_moe_dist(ep_degree):
    world_size = paddle.distributed.get_world_size()
    assert world_size % ep_degree == 0
    assert ep_degree > 1, "ep_degree=1 falls back to the non-MoE topology"
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
    fleet.init(is_collective=True, strategy=strategy)


def tag_experts(model, group):
    for layer in model.layers:
        for param in layer.moe.experts.parameters():
            param.is_moe_param = True
            param.expert = True
            param.color = {"color": "moe_expert", "group": group}


def build_moe_models(hcg):
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


def build_moe_optimizer(model):
    optimizer = paddle.optimizer.AdamW(
        learning_rate=0.001,
        parameters=[p for p in model.parameters() if p.trainable],
        weight_decay=0.0,
        multi_precision=True,
    )
    return mix_precision_utils.MixPrecisionOptimizer(optimizer)


def train_moe(model, optimizer, data, accum_steps=1):
    loss_md5s = []
    for i in range(0, len(data), accum_steps):
        for x in data[i : i + accum_steps]:
            model.train()
            with paddle.amp.auto_cast(level="O1", dtype="bfloat16"):
                loss = model(x).mean()
            loss_md5s.append(loss._md5sum())
            loss.backward()
        optimizer.step()
        optimizer.clear_grad()
    return loss_md5s


def run_moe(ep_degree):
    paddle.distributed.init_parallel_env()
    init_moe_dist(ep_degree)
    hcg = fleet.get_hybrid_communicate_group()
    assert (
        hcg.get_moe_sharding_parallel_world_size()
        == paddle.distributed.get_world_size() // ep_degree
    )

    stage1_model, fsdp_model = build_moe_models(hcg)
    paddle.seed(2026)
    data = [paddle.randn([TOKENS, HIDDEN]) for _ in range(STEPS)]

    stage1_model = mix_precision_utils.MixPrecisionLayer(
        stage1_model, dtype="bfloat16"
    )
    stage1_optimizer = build_moe_optimizer(stage1_model)
    stage1_loss_md5s = train_moe(
        fleet.distributed_model(stage1_model),
        fleet.distributed_optimizer(stage1_optimizer),
        data,
        accum_steps=ACCUM_STEPS,
    )

    for enable_overlap in (True, False):
        if fsdp_model is None:
            _, fsdp_model = build_moe_models(hcg)
        fsdp_model = fully_shard(
            fsdp_model, enable_tensor_fusion_and_overlap=enable_overlap
        )
        fsdp_model = mix_precision_utils.MixPrecisionLayer(
            fsdp_model, dtype="bfloat16"
        )
        fsdp_loss_md5s = train_moe(
            fsdp_model,
            build_moe_optimizer(fsdp_model),
            data,
            accum_steps=ACCUM_STEPS,
        )
        fsdp_model = None

        assert fsdp_loss_md5s == stage1_loss_md5s, (
            f"loss MD5 sequence diverged (ep_degree={ep_degree}, "
            f"accum_steps={ACCUM_STEPS}, "
            f"enable_overlap={enable_overlap}): "
            f"stage1={stage1_loss_md5s}, fsdp={fsdp_loss_md5s}"
        )


class MuonStackedLayer(TransformerLayer):
    """Layer whose weights are 3D stacks, like Muon expert params.

    Subclasses TransformerLayer only so FSDP treats it as a unit:
    ``is_fsdp_unit`` matches class names along the MRO.
    """

    def __init__(self, hidden, inter, num_mats):
        nn.Layer.__init__(self)
        self.attn = nn.Linear(hidden, hidden, bias_attr=False)
        self.w_up = self.create_parameter(shape=[num_mats, hidden, inter])
        self.w_down = self.create_parameter(shape=[num_mats, inter, hidden])

    def forward(self, x):
        x = x + self.attn(x)
        for i in range(self.w_up.shape[0]):
            h = F.silu(paddle.matmul(x, self.w_up[i]))
            x = x + paddle.matmul(h, self.w_down[i])
        return x


class MuonModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(HIDDEN, HIDDEN, bias_attr=False)
        self.layers = nn.LayerList(
            [
                MuonStackedLayer(HIDDEN, INTER, MUON_MATS)
                for _ in range(MUON_LAYERS)
            ]
        )
        self.head = nn.Linear(HIDDEN, 2, bias_attr=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def tag_muon_params(model):
    info_map = {}
    for layer in model.layers:
        for param in (layer.w_up, layer.w_down):
            param.use_muon = True
            info_map[param.name] = MuonParamInfo(use_muon=True)
    return info_map


def train_muon(model, info_map, ns_per_matrix, data):
    model = fully_shard(model)
    fsdp_context = model._fsdp_context
    model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
    optimizer = paddle.optimizer.Muon(
        learning_rate=0.001,
        parameters=[p for p in model.parameters() if p.trainable],
        weight_decay=0.0,
        muon_param_info_map=info_map,
        ns_per_matrix=ns_per_matrix,
        multi_precision=True,
    )
    optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)
    losses = []
    for x in data:
        model.train()
        with paddle.amp.auto_cast(level="O1", dtype="bfloat16"):
            loss = model(x).mean()
        losses.append(float(loss.astype("float32")))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
    return losses, fsdp_context, optimizer._inner_opt


def muon_groups_of(fsdp_context):
    return [
        group
        for group in fsdp_context.buffer_manager.buffer_groups
        if group.use_muon and group.grads_buffer is not None
    ]


def gather_muon_params(fsdp_context):
    """Every Muon group's full params buffer, gathered on all ranks."""
    buffers = {}
    for gid, group in enumerate(fsdp_context.buffer_manager.buffer_groups):
        if not group.use_muon or group.grads_buffer is None:
            continue
        params_buffer = group.params_buffer
        buffer = params_buffer.data_buffer
        if params_buffer.is_sharded:
            parts = []
            paddle.distributed.all_gather(parts, buffer, group=group.fsdp_group)
            buffer = paddle.concat(parts)
        buffers[gid] = buffer.astype("float32").numpy()
    return buffers


def run_muon_shard():
    """Matrix-aligned Muon sharding must match the dense element-shard path.

    Reuses the topology ``run_moe`` already initialized. These params are not
    tagged as experts, so their FSDP group is the sharding group and spans
    every rank, which is what makes a matrix-aligned shard possible on 2 cards.

    Both must reach the same weights. One step, compared bitwise: they start
    from the same weights, so an exact match pins down the shard mapping, the
    grad reduction and the owner gather/scatter. Running longer would not be
    bitwise comparable -- the sharded buffer is all-gathered before each
    forward, and the resulting GEMM layout moves the loss by ~1e-5 per step
    even when the weights are identical.
    """
    paddle.seed(2026)
    matrix_model = MuonModel()
    paddle.seed(2026)
    dense_model = MuonModel()
    dense_model.set_state_dict(matrix_model.state_dict())
    matrix_info = tag_muon_params(matrix_model)
    dense_info = tag_muon_params(dense_model)

    paddle.seed(2026)
    data = [paddle.randn([TOKENS, HIDDEN]) for _ in range(MUON_STEPS)]

    # Baseline: hide the matrix-aligned shard plan so the Muon groups fall back
    # to the dense element-shard + owner-gather path. ns_per_matrix is passed by
    # hand, since only the matrix-aligned path turns it on by itself.
    origin_shard_numel = fully_shard_fusion._muon_3d_shard_numel
    fully_shard_fusion._muon_3d_shard_numel = lambda *args, **kwargs: None
    try:
        dense_losses, dense_ctx, _ = train_muon(
            dense_model, dense_info, True, data
        )
    finally:
        fully_shard_fusion._muon_3d_shard_numel = origin_shard_numel

    dense_groups = muon_groups_of(dense_ctx)
    assert dense_groups
    for group in dense_groups:
        # Element-sharded like AdamW: no matrix granularity, an owner does the
        # per-step gather, and the buffer is still sharded across ranks.
        assert group.muon_shard_numel is None
        assert group.muon_owner_rank is not None
        assert group.params_buffer.is_sharded
    dense_params = gather_muon_params(dense_ctx)

    matrix_losses, matrix_ctx, matrix_opt = train_muon(
        matrix_model, matrix_info, False, data
    )

    matrix_groups = muon_groups_of(matrix_ctx)
    assert len(matrix_groups) == len(dense_groups)
    for group in matrix_groups:
        # Every weight here is [MUON_MATS, HIDDEN, INTER] or its transpose, so
        # the shard unit is one matrix; a silent fallback leaves this None.
        assert group.muon_shard_numel == HIDDEN * INTER, (
            f"expected matrix-aligned sharding, got {group.muon_shard_numel}"
        )
        assert group.params_buffer.is_sharded
        assert group.muon_owner_rank is None
    # The matrix-aligned path must switch Newton-Schulz to per-matrix by itself.
    assert matrix_opt._ns_per_matrix
    matrix_params = gather_muon_params(matrix_ctx)

    assert matrix_losses == dense_losses, (
        f"the two paths did not start from the same weights: "
        f"matrix={matrix_losses}, dense={dense_losses}"
    )
    assert sorted(matrix_params) == sorted(dense_params)
    for gid, expected in dense_params.items():
        np.testing.assert_array_equal(
            matrix_params[gid],
            expected,
            err_msg=(
                f"matrix-sharded Muon diverged from the dense path, group {gid}"
            ),
        )


if __name__ == '__main__':
    run_moe(2)
    run_muon_shard()
