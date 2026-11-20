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
"""FSDP with tp=2 + sequence parallel must match a plain tp+sp run.

Sequence-parallel params (the LayerNorms and the row-parallel bias) see only
the local tokens, so their grads are partial over tp. The baseline sums them
with ``register_sequence_parallel_allreduce_hooks``; under FSDP the sum is done
on the reduced grad buffers. tp-sharded weights must stay local.
"""

from unittest import mock

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import sequence_parallel_utils as spu
from paddle.distributed.fsdp import fully_shard_fusion
from paddle.distributed.fsdp.fully_shard import fully_shard

MP_DEGREE = 2
HIDDEN = 16
INTER = 32
NUM_LAYERS = 2
TOKENS = 64
STEPS = 4


class TransformerLayer(nn.Layer):
    # Named so FSDP treats each layer as a unit.
    def __init__(self, fc1, fc2):
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN)
        spu.mark_as_sequence_parallel_parameter(self.norm.weight)
        spu.mark_as_sequence_parallel_parameter(self.norm.bias)
        self.up = spu.ColumnSequenceParallelLinear(
            HIDDEN,
            INTER,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Assign(fc1)
            ),
            has_bias=True,
            gather_output=False,
        )
        # has_bias marks the row-parallel bias as sequence parallel.
        self.down = spu.RowSequenceParallelLinear(
            INTER,
            HIDDEN,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Assign(fc2)
            ),
            has_bias=True,
            input_is_parallel=True,
        )

    def forward(self, x):
        return x + self.down(F.silu(self.up(self.norm(x))))


class SPModel(nn.Layer):
    def __init__(self, seed):
        super().__init__()
        rng = np.random.RandomState(seed)
        mp_rank = fleet.get_hybrid_communicate_group().get_model_parallel_rank()
        per_rank = INTER // MP_DEGREE
        cols = slice(mp_rank * per_rank, (mp_rank + 1) * per_rank)

        def init(*shape):
            return rng.uniform(-0.5, 0.5, shape).astype("float32")

        self.embed = nn.Linear(
            HIDDEN,
            HIDDEN,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Assign(init(HIDDEN, HIDDEN))
            ),
            bias_attr=False,
        )
        layers = []
        for _ in range(NUM_LAYERS):
            fc1, fc2 = init(HIDDEN, INTER), init(INTER, HIDDEN)
            layers.append(TransformerLayer(fc1[:, cols], fc2[cols, :]))
        self.layers = nn.LayerList(layers)
        self.final_norm = nn.LayerNorm(HIDDEN)
        spu.mark_as_sequence_parallel_parameter(self.final_norm.weight)
        spu.mark_as_sequence_parallel_parameter(self.final_norm.bias)
        self.head = nn.Linear(
            HIDDEN,
            2,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Assign(init(HIDDEN, 2))
            ),
            bias_attr=False,
        )

    def forward(self, x):
        x = spu.ScatterOp.apply(self.embed(x))
        for layer in self.layers:
            x = layer(x)
        x = spu.GatherOp.apply(self.final_norm(x))
        return (self.head(x) ** 2).mean()


def build_optimizer(model):
    return paddle.optimizer.AdamW(
        learning_rate=0.01,
        parameters=model.parameters(),
        weight_decay=0.0,
    )


def train_baseline(data):
    model = SPModel(seed=2026)
    spu.register_sequence_parallel_allreduce_hooks(model, 1, False)
    opt = build_optimizer(model)
    losses, grads = [], []
    for x in data:
        loss = model(x)
        losses.append(float(loss))
        loss.backward()
        grads.append([p.grad.numpy() for p in model.parameters()])
        opt.step()
        opt.clear_grad()
    return model, losses, grads


def fsdp_grads(model):
    """Per-param grads read back from the unsharded FSDP grad buffers."""
    bm = model._fsdp_context.buffer_manager
    grads = []
    for p in model.parameters():
        buf = bm.buffer_groups[bm.param_to_buffer_id[p.name]].grads_buffer
        assert not buf.is_sharded, "fsdp_degree=1 keeps grads whole"
        off = buf.param_offsets[p.name]
        grad = buf.data_buffer._slice(off, off + int(np.prod(p.shape)))
        grads.append(grad.numpy().reshape(p.shape))
    return grads


def train_fsdp(data):
    model = SPModel(seed=2026)
    opt = build_optimizer(model)
    model = fully_shard(model, optimizer=opt)
    bm = model._fsdp_context.buffer_manager
    assert bm.tp_group.nranks == MP_DEGREE
    assert bm.sp_grad_slices, "no sequence-parallel grad slice collected"
    losses, grads = [], []
    for x in data:
        loss = model(x)
        losses.append(float(loss))
        loss.backward()
        opt.step()
        # The reduced grads stay in the buffers until clear_grad.
        grads.append(fsdp_grads(model))
        opt.clear_grad()
    model._fsdp_context.all_gather_params()
    return model, losses, grads


def assert_close(actual, expected, what):
    for i, (a, e) in enumerate(zip(actual, expected)):
        np.testing.assert_allclose(
            a, e, rtol=1e-5, atol=1e-6, err_msg=f"{what} #{i} diverged"
        )


def run_sp_grad_sum():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": MP_DEGREE,
        "pp_degree": 1,
        "sharding_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)
    assert paddle.distributed.get_world_size() == MP_DEGREE

    # Same tokens on every rank; ScatterOp gives each rank half of them.
    paddle.seed(2026)
    data = [paddle.randn([TOKENS, HIDDEN]) for _ in range(STEPS)]

    base_model, base_losses, base_grads = train_baseline(data)
    base_params = [p.numpy() for p in base_model.parameters()]
    sp_ids = [
        i
        for i, p in enumerate(base_model.parameters())
        if spu.is_sequence_parallel_parameter(p)
    ]
    # 2 LayerNorms per layer + final LayerNorm + row-parallel bias per layer.
    assert len(sp_ids) == 3 * NUM_LAYERS + 2, sp_ids

    fsdp_model, fsdp_losses, fsdp_grads_ = train_fsdp(data)
    np.testing.assert_allclose(
        fsdp_losses, base_losses, rtol=1e-5, err_msg="losses diverged"
    )
    for step, (got, want) in enumerate(zip(fsdp_grads_, base_grads)):
        assert_close(got, want, f"step {step} grad")
    assert_close(
        [p.numpy() for p in fsdp_model.parameters()], base_params, "param"
    )

    # Without the tp sum only the sp grads go wrong.
    with mock.patch.object(
        fully_shard_fusion, "_sum_slices_over_group", lambda *args: None
    ):
        _, _, unsummed = train_fsdp(data[:1])
    for i, (got, want) in enumerate(zip(unsummed[0], base_grads[0])):
        if i in sp_ids:
            assert not np.allclose(got, want, rtol=1e-5, atol=1e-6), (
                f"sp grad #{i} matches without the tp sum"
            )
        else:
            np.testing.assert_allclose(
                got, want, rtol=1e-5, atol=1e-6, err_msg=f"grad #{i}"
            )


if __name__ == '__main__':
    run_sp_grad_sum()
