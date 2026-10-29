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

"""FSDP with several sharding groups (dp x sharding, 4 cards).

The dense reduce-scatter clones made by ``set_dense_reduce_scatter_streams``
must be created collectively: rank 0 clones [0, 1] and rank 2 clones [2, 3],
and both would get the same gid if each rank only cloned its own group. Every rank
feeds the same data, so the dp replicas stay identical and each run with
streams > 1 must match the streams == 1 run exactly.
"""

from dygraph_group_fsdp import (
    ACCUM_STEPS,
    HIDDEN,
    STEPS,
    TOKENS,
    TransformerLayer,
    build_moe_optimizer,
)

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.distributed.fsdp.fully_shard import fully_shard


class TwoGroupUnit(TransformerLayer):
    # Two dense buffer groups per unit (weight decay on / off), so the second
    # group's reduce-scatter goes through a clone when streams >= 2.
    def __init__(self, hidden):
        nn.Layer.__init__(self)
        self.up = nn.Linear(hidden, hidden, bias_attr=False)
        self.down = nn.Linear(hidden, hidden, bias_attr=False)
        self.down.weight.no_weight_decay = True

    def forward(self, x):
        return x + self.down(F.silu(self.up(x)))


class TwoGroupModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.layers = nn.LayerList([TwoGroupUnit(HIDDEN) for _ in range(2)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def init_dp_sharding_dist(sharding_degree):
    world_size = paddle.distributed.get_world_size()
    assert world_size % sharding_degree == 0
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": world_size // sharding_degree,
        "sharding_degree": sharding_degree,
        "mp_degree": 1,
        "pp_degree": 1,
    }
    strategy.hybrid_configs["sharding_configs"].split_param = True
    fleet.init(is_collective=True, strategy=strategy)


def check_clone_gids(cm, rs_streams):
    """Each gid must name the same rank list on every rank.

    Cloning only the local group would hand rank 0's [0, 1] clone and rank 2's
    [2, 3] clone the same gid, so their NCCL unique ids collide in the store.
    """
    local = {}
    streams = set()
    for group in cm.buffer_manager.buffer_groups:
        buf = group.grads_buffer
        if buf is not None and buf.is_sharded:
            local[buf.fsdp_group.id] = tuple(buf.fsdp_group.ranks)
            streams.add(id(buf.comm_stream))
    # Each communicator used for the dense RS gets its own comm stream.
    assert len(streams) == len(local), (len(streams), local)
    gathered = []
    paddle.distributed.all_gather_object(gathered, local)
    gid_to_ranks = {}
    for per_rank in gathered:
        for gid, ranks in per_rank.items():
            assert gid_to_ranks.setdefault(gid, ranks) == ranks, (
                f"gid {gid} names both {gid_to_ranks[gid]} and {ranks}"
            )
    # Two sharding groups, each reduce-scattered through min(streams, 2) gids
    # (two dense groups per unit).
    assert len(gid_to_ranks) == 2 * min(rs_streams, 2), gid_to_ranks


def train(init_state, data, rs_streams):
    model = TwoGroupModel()
    model.set_state_dict(init_state)
    model = fully_shard(model)
    model._fsdp_context.set_dense_reduce_scatter_streams(rs_streams)
    check_clone_gids(model._fsdp_context.comm_manager, rs_streams)
    model = mix_precision_utils.MixPrecisionLayer(model, dtype="float16")
    optimizer = build_moe_optimizer(model)
    losses = []
    for i in range(0, len(data), ACCUM_STEPS):
        for x in data[i : i + ACCUM_STEPS]:
            model.train()
            with paddle.amp.auto_cast(level="O1", dtype="float16"):
                loss = model(x).mean()
            losses.append(loss._md5sum())
            loss.backward()
        optimizer.step()
        optimizer.clear_grad()
    return losses


def check_guards(init_state):
    model = TwoGroupModel()
    model.set_state_dict(init_state)
    ctx = fully_shard(model)._fsdp_context
    try:
        ctx.set_dense_reduce_scatter_streams(0)
    except ValueError:
        pass
    else:
        raise AssertionError("streams=0 should raise ValueError")
    ctx.set_dense_reduce_scatter_streams(2)
    try:
        ctx.set_dense_reduce_scatter_streams(3)
    except RuntimeError:
        pass
    else:
        raise AssertionError("setting streams twice should raise")


def run_dense_rs_streams_multi_group():
    paddle.distributed.init_parallel_env()
    init_dp_sharding_dist(sharding_degree=2)
    hcg = fleet.get_hybrid_communicate_group()
    assert hcg.get_sharding_parallel_world_size() == 2
    assert hcg.get_data_parallel_world_size() == 2

    paddle.seed(2026)
    init_state = {k: v.clone() for k, v in TwoGroupModel().state_dict().items()}
    paddle.seed(2026)
    data = [paddle.randn([TOKENS, HIDDEN]) for _ in range(STEPS - 1)]

    check_guards(init_state)
    base_losses = train(init_state, data, rs_streams=1)
    for rs_streams in (2, 3):
        losses = train(init_state, data, rs_streams=rs_streams)
        assert losses == base_losses, (
            f"rs_streams={rs_streams}: {losses} != {base_losses}"
        )


if __name__ == '__main__':
    run_dense_rs_streams_multi_group()
