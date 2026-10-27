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


import os
import shutil
import tempfile

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.distributed.fsdp import fully_shard_fusion
from paddle.distributed.fsdp._fsdp_context import register_fsdp_context
from paddle.distributed.fsdp.fully_shard import (
    MixedPrecisionPolicy,
    fully_shard,
)
from paddle.optimizer.muon import Muon, MuonParamInfo

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
MUON_MATS_DOWN = 3
VOCAB = 24
CE_CHUNK = 16


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
            with paddle.amp.auto_cast(level="O1", dtype="float16"):
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
        stage1_model, dtype="float16"
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
            fsdp_model, dtype="float16"
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

    check_clip_attrs(hcg)


def check_clip_attrs(hcg):
    """Fused buffers carry the clip routing attrs of the params they hold."""
    _, model = build_moe_models(hcg)
    for layer in model.layers:
        for param in layer.moe.experts.parameters():
            param.no_sync = True
    optimizer = build_moe_optimizer(model)
    model = fully_shard(
        model, optimizer=optimizer, grad_accum_steps=ACCUM_STEPS
    )
    assert optimizer._fsdp_state_dict_ctx is model._fsdp_context
    for gid, group in enumerate(
        model._fsdp_context.buffer_manager.buffer_groups
    ):
        data_buffer = group.params_buffer.data_buffer
        for attr, default in (
            ("need_clip", True),
            ("no_sync", False),
            ("is_distributed", False),
            ("is_firstly_shared", True),
        ):
            values = {getattr(p, attr, default) for p in group.params}
            assert len(values) == 1, f"bucket {gid} mixes {attr}={values}"
            expected = values.pop()
            if (
                attr == "is_firstly_shared"
                and group.grads_buffer is not None
                and not group.grads_buffer.is_sharded
            ):
                # Replicated grads are counted on rank 0 of fsdp_group only.
                rank = paddle.distributed.get_rank(group.fsdp_group)
                expected = expected and (
                    group.fsdp_group.nranks == 1
                    or group.muon_owner_rank is not None
                    or rank == 0
                )
            assert getattr(data_buffer, attr) == expected, (
                f"bucket {gid}: {attr}={getattr(data_buffer, attr)}, "
                f"expected {expected}"
            )
        assert data_buffer.no_sync == group.is_expert_param, gid


class MuonStackedLayer(TransformerLayer):
    """Layer whose weights are 3D stacks, like Muon expert params.

    Subclasses TransformerLayer only so FSDP treats it as a unit:
    ``is_fsdp_unit`` matches class names along the MRO.
    """

    def __init__(self, hidden, inter, num_mats, num_mats_down=None, low=None):
        nn.Layer.__init__(self)
        self.attn = nn.Linear(hidden, hidden, bias_attr=False)
        self.w_up = self.create_parameter(shape=[num_mats, hidden, inter])
        self.w_down = self.create_parameter(
            shape=[num_mats_down or num_mats, inter, hidden]
        )
        if low is not None:
            self.w_low = self.create_parameter(
                shape=[num_mats_down or num_mats, hidden, hidden], dtype=low
            )

    def forward(self, x):
        x = x + self.attn(x)
        for i in range(self.w_up.shape[0]):
            h = F.silu(paddle.matmul(x, self.w_up[i]))
            x = x + paddle.matmul(h, self.w_down[i % self.w_down.shape[0]])
        w_low = getattr(self, "w_low", None)
        if w_low is not None:
            for i in range(w_low.shape[0]):
                h = paddle.matmul(x.astype(w_low.dtype), w_low[i])
                x = x + F.silu(h.astype(x.dtype))
        return x


class MuonModel(nn.Layer):
    def __init__(self, num_mats_down=None, low=None):
        super().__init__()
        self.embed = nn.Linear(HIDDEN, HIDDEN, bias_attr=False)
        self.layers = nn.LayerList(
            [
                MuonStackedLayer(
                    HIDDEN, INTER, MUON_MATS, num_mats_down, low=low
                )
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
        stacks = [layer.w_up, layer.w_down]
        w_low = getattr(layer, "w_low", None)
        if w_low is not None:
            stacks.append(w_low)
        for param in stacks:
            param.use_muon = True
            info_map[param.name] = MuonParamInfo(use_muon=True)
    return info_map


def train_muon(model, info_map, data, probe=None):
    """Train with Muon under FSDP, with per-matrix Newton-Schulz.

    ``probe(step, fsdp_context)`` runs after every step, for tests that need to
    inspect per-step buffer state.
    """
    model = fully_shard(model)
    fsdp_context = model._fsdp_context
    model = mix_precision_utils.MixPrecisionLayer(model, dtype="float16")
    optimizer = paddle.optimizer.Muon(
        learning_rate=0.001,
        parameters=[p for p in model.parameters() if p.trainable],
        weight_decay=0.0,
        muon_param_info_map=info_map,
        ns_per_matrix=True,
        multi_precision=True,
    )
    optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)
    losses = []
    for step, x in enumerate(data):
        model.train()
        with paddle.amp.auto_cast(level="O1", dtype="float16"):
            loss = model(x).mean()
        losses.append(float(loss.astype("float32")))
        loss.backward()
        optimizer.step()
        if probe is not None:
            probe(step, fsdp_context)
        optimizer.clear_grad()
    return losses, fsdp_context


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

    origin_shard_numel = fully_shard_fusion._muon_3d_shard_numel
    fully_shard_fusion._muon_3d_shard_numel = lambda *args, **kwargs: None
    try:
        dense_losses, dense_ctx = train_muon(dense_model, dense_info, data)
    finally:
        fully_shard_fusion._muon_3d_shard_numel = origin_shard_numel

    dense_groups = muon_groups_of(dense_ctx)
    assert dense_groups
    for group in dense_groups:
        assert group.muon_shard_numel is None
        assert group.muon_owner_rank is not None
        assert group.params_buffer.is_sharded
    dense_params = gather_muon_params(dense_ctx)

    matrix_losses, matrix_ctx = train_muon(matrix_model, matrix_info, data)

    matrix_groups = muon_groups_of(matrix_ctx)
    assert len(matrix_groups) == len(dense_groups)
    for group in matrix_groups:
        assert group.muon_shard_numel == HIDDEN * INTER, (
            f"expected matrix-aligned sharding, got {group.muon_shard_numel}"
        )
        assert group.params_buffer.is_sharded
        assert group.muon_owner_rank is None
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

    # fully_shard(optimizer=Muon) rejects trainable params missing from
    # muon_param_info_map; here embed/head are untagged.
    model = MuonModel()
    optimizer = paddle.optimizer.Muon(
        parameters=model.parameters(),
        muon_param_info_map=tag_muon_params(model),
    )
    try:
        fully_shard(model, optimizer=optimizer)
    except ValueError as e:
        assert "misses trainable params" in str(e), e
    else:
        raise AssertionError("untagged trainable params were not rejected")


def run_muon_replicated():
    paddle.seed(2026)
    replicated_model = MuonModel(MUON_MATS_DOWN)
    paddle.seed(2026)
    dense_model = MuonModel(MUON_MATS_DOWN)
    dense_model.set_state_dict(replicated_model.state_dict())
    replicated_info = tag_muon_params(replicated_model)
    dense_info = tag_muon_params(dense_model)

    paddle.seed(2026)
    data = [paddle.randn([TOKENS, HIDDEN]) for _ in range(2)]

    origin_shard_numel = fully_shard_fusion._muon_3d_shard_numel
    fully_shard_fusion._muon_3d_shard_numel = lambda *args, **kwargs: None
    try:
        dense_losses, dense_ctx = train_muon(
            dense_model, dense_info, data[:MUON_STEPS]
        )
    finally:
        fully_shard_fusion._muon_3d_shard_numel = origin_shard_numel
    dense_params = gather_muon_params(dense_ctx)

    snapshot = {}

    def probe(step, fsdp_context):
        if step + 1 == MUON_STEPS:
            snapshot.update(gather_muon_params(fsdp_context))

    replicated_losses, ctx = train_muon(
        replicated_model, replicated_info, data, probe=probe
    )

    groups = muon_groups_of(ctx)
    assert len(groups) == MUON_LAYERS, groups
    my_rank = paddle.distributed.get_rank()
    for group in groups:
        assert group.muon_shard_numel is None
        assert not group.params_buffer.is_sharded
        assert group.muon_owner_rank is not None
        if group.fsdp_group.ranks[group.muon_owner_rank] != my_rank:
            assert group.grads_buffer._data_released

    assert replicated_losses[:MUON_STEPS] == dense_losses, (
        f"the two paths did not start from the same weights: "
        f"replicated={replicated_losses[:MUON_STEPS]}, dense={dense_losses}"
    )
    assert sorted(snapshot) == sorted(dense_params)
    for gid, expected in dense_params.items():
        np.testing.assert_array_equal(
            snapshot[gid],
            expected,
            err_msg=(
                f"replicated Muon diverged from the dense path, group {gid}"
            ),
        )

    ctx.set_grad_accum_steps(ACCUM_STEPS)
    for group in ctx.buffer_manager.buffer_groups:
        grads_buffer = group.grads_buffer
        assert grads_buffer.grad_accum_steps == ACCUM_STEPS
        assert grads_buffer.grad_scale == 1.0 / (
            grads_buffer.grad_div * ACCUM_STEPS
        )
    ctx.all_gather_params()
    for group in ctx.buffer_manager.buffer_groups:
        for param in group.params:
            assert param._is_initialized(), f"{param.name} not materialized"


def tag_adamh_params(model):
    return {
        param.name: MuonParamInfo(use_muon=False, use_hyperball=True)
        for param in model.parameters()
        if param.trainable
    }


def build_adamh_optimizer(model, info_map):
    return paddle.optimizer.Muon(
        learning_rate=0.01,
        parameters=[p for p in model.parameters() if p.trainable],
        weight_decay=0.0,
        muon_param_info_map=info_map,
        multi_precision=True,
    )


def train_adamh(model, optimizer, data, grad_scale=1.0):
    losses = []
    for i in range(0, len(data), ACCUM_STEPS):
        for x in data[i : i + ACCUM_STEPS]:
            model.train()
            with paddle.amp.auto_cast(level="O1", dtype="float16"):
                loss = model(x).mean()
            losses.append(float(loss.astype("float32")))
            loss.backward()
        if grad_scale != 1.0:
            for param in model.parameters():
                if getattr(param, "main_grad", None) is not None:
                    param.main_grad.scale_(grad_scale)
        optimizer.step()
        optimizer.clear_grad()
    return losses


def run_adamh_shard():
    w = paddle.randn([HIDDEN, INTER])
    u = paddle.randn([HIDDEN, INTER])
    np.testing.assert_allclose(
        Muon._hyperball_apply(w.flatten(), u.flatten(), 0.1).numpy(),
        Muon._hyperball_apply(w, u, 0.1).flatten().numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    paddle.seed(2026)
    base_model = MuonModel()
    init_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    paddle.seed(2026)
    data = [paddle.randn([TOKENS, HIDDEN]) for _ in range(STEPS - 1)]

    register_fsdp_context(None)
    base_optimizer = build_adamh_optimizer(
        base_model, tag_adamh_params(base_model)
    )
    base_model = mix_precision_utils.MixPrecisionLayer(
        base_model, dtype="float16"
    )
    base_losses = train_adamh(
        base_model,
        mix_precision_utils.MixPrecisionOptimizer(base_optimizer),
        data,
        grad_scale=1.0 / ACCUM_STEPS,
    )
    base_params = [p.numpy() for p in base_model.parameters()]

    for enable_overlap in (True, False):
        fsdp_model = MuonModel()
        fsdp_model.set_state_dict(init_state)
        optimizer = build_adamh_optimizer(
            fsdp_model, tag_adamh_params(fsdp_model)
        )
        fsdp_model = fully_shard(
            fsdp_model,
            enable_tensor_fusion_and_overlap=enable_overlap,
            optimizer=optimizer,
            grad_accum_steps=ACCUM_STEPS,
        )
        fsdp_context = fsdp_model._fsdp_context

        sharded = [
            group
            for group in fsdp_context.buffer_manager.buffer_groups
            if group.hyperball_sharded
        ]
        assert sharded, "no 2D param took the sharded Hyperball path"
        for group in sharded:
            assert len(group.params) == 1
            assert len(group.params[0].shape) == 2
            assert not group.use_muon
            data_buffer = group.params_buffer.data_buffer
            assert data_buffer._hyperball_group is group.fsdp_group
        for layer in fsdp_model.layers:
            for param in (layer.w_up, layer.w_down):
                assert not param.use_hyperball_sharded, param.name
                assert param.use_muon, param.name

        fsdp_model = mix_precision_utils.MixPrecisionLayer(
            fsdp_model, dtype="float16"
        )
        fsdp_losses = train_adamh(
            fsdp_model,
            mix_precision_utils.MixPrecisionOptimizer(optimizer),
            data,
        )
        np.testing.assert_allclose(
            fsdp_losses,
            base_losses,
            rtol=1e-6,
            err_msg=f"AdamW+Hyperball losses diverged (overlap={enable_overlap})",
        )
        fsdp_context.all_gather_params()
        for param, expected in zip(fsdp_model.parameters(), base_params):
            np.testing.assert_allclose(
                param.numpy(),
                expected,
                rtol=1e-6,
                atol=1e-7,
                err_msg=f"{param.name} diverged (overlap={enable_overlap})",
            )


def build_muon_checkpoint_run(num_mats_down):
    paddle.seed(2026)
    model = MuonModel(num_mats_down, low="bfloat16")
    info_map = tag_muon_params(model)
    model = fully_shard(model)
    fsdp_context = model._fsdp_context
    model = mix_precision_utils.MixPrecisionLayer(model, dtype="float16")
    inner = paddle.optimizer.Muon(
        learning_rate=0.001,
        parameters=[p for p in model.parameters() if p.trainable],
        weight_decay=0.0,
        muon_param_info_map=info_map,
        multi_precision=True,
    )
    optimizer = mix_precision_utils.MixPrecisionOptimizer(inner)

    def train_step(x):
        model.train()
        with paddle.amp.auto_cast(level="O1", dtype="float16"):
            loss = model(x).mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

    def state_dict():
        state = dict(model.sharded_state_dict())
        state.update(optimizer.sharded_state_dict())
        return state

    return fsdp_context, optimizer, inner, train_step, state_dict


def check_muon_checkpoint(num_mats_down, ckpt_path, dense):
    """Muon state saved through FSDP must restore a run that steps the same."""
    paddle.seed(2026)
    data = [paddle.randn([TOKENS, HIDDEN]) for _ in range(3)]

    ctx, _, inner, train_step, state_dict = build_muon_checkpoint_run(
        num_mats_down
    )
    train_step(data[0])
    train_step(data[1])
    assert inner._fsdp_state_dict_ctx is ctx, "Muon never bound FSDP"
    groups = muon_groups_of(ctx)
    matrix_sharded = all(g.muon_shard_numel is not None for g in groups)
    assert matrix_sharded == (not dense and num_mats_down is None)

    state = state_dict()
    muon_names = {p.name for g in groups for p in g.params}
    held = [
        name
        for name in inner._accumulators[inner._moment_acc_str]
        if name in muon_names
    ]
    for name in held:
        key = f"{ctx._muon_state_descs[name][0]}.moment1_0"
        assert key in state, f"Muon state {key} missing from the checkpoint"
        assert np.abs(state[key].local_tensor.numpy()).max() > 0, key
    num_held = paddle.to_tensor([len(held)])
    paddle.distributed.all_reduce(num_held)
    if matrix_sharded:
        assert int(num_held) >= len(muon_names), int(num_held)
    else:
        assert int(num_held) == len(muon_names), int(num_held)
    masters = [name for name in inner._master_weights if name in muon_names]
    for name in masters:
        key = f"{ctx._muon_state_descs[name][0]}.w_0"
        assert key in state, f"Muon master weight {key} missing"
    num_masters = paddle.to_tensor([len(masters)])
    paddle.distributed.all_reduce(num_masters)
    assert int(num_masters), "the bf16 stack contributed no master weight"
    saved = {
        key: sharded.local_tensor.astype("float32").numpy()
        for key, sharded in state.items()
    }
    paddle.distributed.save_state_dict(state, ckpt_path)
    paddle.distributed.barrier()

    train_step(data[2])
    expected = gather_muon_params(ctx)

    ctx, optimizer, inner, train_step, state_dict = build_muon_checkpoint_run(
        num_mats_down
    )
    # Bind the wrapper, not the inner Muon: FSDP has to unwrap it.
    ctx.bind_optimizer(optimizer)
    optimizer.init_state_for_load()
    state = state_dict()
    assert sorted(state) == sorted(saved), sorted(set(state) ^ set(saved))
    paddle.distributed.load_state_dict(state, ckpt_path)
    for key, sharded in state.items():
        np.testing.assert_array_equal(
            sharded.local_tensor.astype("float32").numpy(),
            saved[key],
            err_msg=key,
        )
    train_step(data[2])
    resumed = gather_muon_params(ctx)
    assert sorted(resumed) == sorted(expected)
    for gid, value in expected.items():
        buffer = ctx.buffer_manager.buffer_groups[gid].params_buffer
        low = buffer.data_buffer.dtype != paddle.float32
        np.testing.assert_allclose(
            resumed[gid],
            value,
            rtol=2e-2 if low else 1e-3,
            atol=2e-3 if low else 1e-4,
            err_msg=f"resumed Muon diverged, group {gid}",
        )


def run_muon_checkpoint():
    ckpt_root = [None]
    if paddle.distributed.get_rank() == 0:
        ckpt_root = [tempfile.mkdtemp()]
    paddle.distributed.broadcast_object_list(ckpt_root, src=0)
    ckpt_root = ckpt_root[0]
    try:
        check_muon_checkpoint(None, os.path.join(ckpt_root, "matrix"), False)
        origin_shard_numel = fully_shard_fusion._muon_3d_shard_numel
        fully_shard_fusion._muon_3d_shard_numel = lambda *args, **kwargs: None
        try:
            check_muon_checkpoint(None, os.path.join(ckpt_root, "dense"), True)
        finally:
            fully_shard_fusion._muon_3d_shard_numel = origin_shard_numel
        check_muon_checkpoint(
            MUON_MATS_DOWN, os.path.join(ckpt_root, "replicated"), False
        )
    finally:
        paddle.distributed.barrier()
        if paddle.distributed.get_rank() == 0:
            shutil.rmtree(ckpt_root, ignore_errors=True)


class FusedLinearCE(PyLayer):
    """Chunked linear + cross-entropy, mirroring the fused LM-head loss.

    Like the real kernel, every grad is computed in forward, chunk by chunk,
    and the weight grad is kept in fp32. Without FSDP it is added straight to
    ``main_grad``. Under FSDP it is handed to the grad hook through
    ``_fused_kernel_fp32_grad``, and the grad returned through autograd is
    deliberately zero: the run can only match the baseline if the hook takes
    the stashed tensor.
    """

    @staticmethod
    def forward(ctx, x, weight, labels, fsdp):
        # The LM head returned the weight; FSDP must not have re-sharded it.
        assert weight._is_initialized(), "exposed param was re-sharded early"
        num_tokens = x.shape[0]
        grad_x = []
        grad_w = paddle.zeros(weight.shape, dtype="float32")
        loss = paddle.zeros([], dtype="float32")
        for start in range(0, num_tokens, CE_CHUNK):
            x_c = x[start : start + CE_CHUNK]
            label_c = labels[start : start + CE_CHUNK]
            logits = paddle.matmul(x_c, weight, transpose_y=True)
            loss += F.cross_entropy(logits, label_c, reduction="sum")
            dlogits = F.softmax(logits, axis=-1) - F.one_hot(label_c, VOCAB)
            grad_x.append(paddle.matmul(dlogits, weight) / num_tokens)
            grad_w += paddle.matmul(dlogits, x_c, transpose_x=True) / num_tokens
        ctx.save_for_backward(paddle.concat(grad_x), grad_w)
        ctx.weight = weight
        ctx.fsdp = fsdp
        return loss / num_tokens

    @staticmethod
    def backward(ctx, dy):
        grad_x, grad_w = ctx.saved_tensor()
        grad_x = grad_x * dy
        grad_w = grad_w * dy
        weight = ctx.weight
        if ctx.fsdp:
            stashed = getattr(weight, "_fused_kernel_fp32_grad", None)
            weight._fused_kernel_fp32_grad = (stashed or []) + [grad_w]
            return grad_x, paddle.zeros_like(weight), None
        if weight.main_grad is None:
            weight.main_grad = paddle.zeros(weight.shape, dtype="float32")
        weight.main_grad.add_(grad_w)
        return grad_x, None, None


class DenseUnit(TransformerLayer):
    # Subclasses TransformerLayer only so FSDP treats it as a unit.
    def __init__(self, hidden):
        nn.Layer.__init__(self)
        self.attn = nn.Linear(hidden, hidden, bias_attr=False)

    def forward(self, x):
        return x + F.silu(self.attn(x))


class FusedLMHead(nn.Layer):
    """Returns ``(hidden, weight)`` and leaves the projection to the loss."""

    def __init__(self, hidden, vocab):
        super().__init__()
        self.weight = self.create_parameter(shape=[vocab, hidden])

    def forward(self, x):
        return x, self.weight


class FusedLinearModel(nn.Layer):
    # No embedding: a non-unit layer ahead of the units would share the
    # non-unit buffer with the head, which already diverges under overlap on
    # develop regardless of the fused path.
    def __init__(self):
        super().__init__()
        self.layers = nn.LayerList([DenseUnit(HIDDEN) for _ in range(2)])
        self.mtp_proj = nn.Linear(HIDDEN, HIDDEN, bias_attr=False)
        self.lm_head = FusedLMHead(HIDDEN, VOCAB)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        hidden, weight = self.lm_head(x)
        return hidden, self.mtp_proj(hidden), weight


def train_fused_linear(model, data, fsdp):
    optimizer = build_moe_optimizer(model)
    lm_weight = model._layers.lm_head.weight
    losses = []
    for i in range(0, len(data), ACCUM_STEPS):
        for x, labels in data[i : i + ACCUM_STEPS]:
            model.train()
            hidden, mtp_hidden, weight = model(x)
            # Main + MTP head: two fused calls feed the same weight.
            loss = FusedLinearCE.apply(
                hidden, weight, labels, fsdp
            ) + FusedLinearCE.apply(mtp_hidden, weight, labels, fsdp)
            losses.append(float(loss))
            loss.backward()
            if fsdp:
                assert lm_weight._fused_kernel_fp32_grad is None
        optimizer.step()
        optimizer.clear_grad()
    return losses


def run_fused_linear():
    """Exposed LM-head param + fp32 fused grad must match a non-FSDP run.

    Reuses the topology ``run_moe`` already initialized. Every rank feeds the
    same data, so FSDP's averaged grad equals the local grad of a plain
    single-card baseline.
    """
    paddle.seed(2026)
    base_model = FusedLinearModel()
    paddle.seed(2026)
    data = [
        (
            paddle.randn([TOKENS, HIDDEN]),
            paddle.randint(0, VOCAB, [TOKENS]),
        )
        for _ in range(STEPS - 1)
    ]
    init_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    # The optimizer steps on whichever FSDP context was registered last; drop
    # the one left by the earlier runs so the baseline steps on its own params.
    register_fsdp_context(None)
    base_model = mix_precision_utils.MixPrecisionLayer(
        base_model, dtype="float16"
    )
    base_losses = train_fused_linear(base_model, data, fsdp=False)
    base_params = [p.numpy() for p in base_model.parameters()]

    for enable_overlap in (True, False):
        fsdp_model = FusedLinearModel()
        fsdp_model.set_state_dict(init_state)
        fsdp_model = fully_shard(
            fsdp_model, enable_tensor_fusion_and_overlap=enable_overlap
        )
        fsdp_context = fsdp_model._fsdp_context
        fsdp_model = mix_precision_utils.MixPrecisionLayer(
            fsdp_model, dtype="float16"
        )
        fsdp_losses = train_fused_linear(fsdp_model, data, fsdp=True)
        np.testing.assert_allclose(
            fsdp_losses,
            base_losses,
            rtol=1e-6,
            err_msg=f"fused linear losses diverged (overlap={enable_overlap})",
        )
        fsdp_context.all_gather_params()
        for param, expected in zip(fsdp_model.parameters(), base_params):
            np.testing.assert_allclose(
                param.numpy(),
                expected,
                rtol=1e-6,
                atol=1e-7,
                err_msg=f"{param.name} diverged (overlap={enable_overlap})",
            )


def train_grad_comm(comm_dtype, data):
    _, model = build_moe_models(fleet.get_hybrid_communicate_group())
    opt = paddle.optimizer.AdamW(
        learning_rate=0.001, parameters=model.parameters(), weight_decay=0.0
    )
    model = fully_shard(
        model,
        optimizer=opt,
        grad_accum_steps=ACCUM_STEPS,
        mp_policy=MixedPrecisionPolicy(grad_comm_dtype=comm_dtype),
    )
    ctx = model._fsdp_context
    sharded = [
        g.grads_buffer
        for g in ctx.buffer_manager.buffer_groups
        if g.grads_buffer is not None and g.grads_buffer.is_sharded
    ]
    assert sharded, "no sharded grad buffer"
    for buf in sharded:
        assert (buf.comm_dtype is not None) == (comm_dtype is not None)
    losses, grads = [], []
    for i in range(0, len(data), ACCUM_STEPS):
        for x in data[i : i + ACCUM_STEPS]:
            loss = (model(x) ** 2).mean()
            losses.append(float(loss))
            loss.backward()
        opt.step()
        grads.append([buf.data_buffer.numpy() for buf in sharded])
        opt.clear_grad()
    assert not ctx.comm_manager.grad_reduce_queue
    assert all(buf._rs_comm_buffer is None for buf in sharded)
    return losses, grads


def run_grad_comm_dtype():
    """bf16 grad reduce-scatter stays close to the fp32 one."""
    for bad in (paddle.float16, paddle.int32, "bfloat16"):
        try:
            fully_shard(
                nn.Linear(HIDDEN, HIDDEN),
                mp_policy=MixedPrecisionPolicy(grad_comm_dtype=bad),
            )
        except ValueError as e:
            assert "grad_comm_dtype" in str(e), e
        else:
            raise AssertionError(f"grad_comm_dtype={bad} was not rejected")
    paddle.seed(7 + paddle.distributed.get_rank())
    data = [paddle.randn([TOKENS, HIDDEN]) for _ in range(STEPS - 1)]
    ref_losses, ref_grads = train_grad_comm(None, data)
    new_losses, new_grads = train_grad_comm(paddle.bfloat16, data)
    assert ref_losses[:ACCUM_STEPS] == new_losses[:ACCUM_STEPS]
    inexact = 0
    for a, b in zip(ref_grads[0], new_grads[0]):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-2 * np.abs(a).max())
        inexact += int(not np.array_equal(a, b))
    assert inexact, "bf16 reduce-scatter left grads exact"
    np.testing.assert_allclose(new_losses, ref_losses, rtol=1e-2)


if __name__ == '__main__':
    run_moe(2)
    run_fused_linear()
    run_grad_comm_dtype()
    run_muon_shard()
    run_muon_replicated()
    run_adamh_shard()
    run_muon_checkpoint()
