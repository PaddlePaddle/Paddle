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

import socket
from collections import Counter
from dataclasses import dataclass

import paddle

_OUTER_STRATEGIES = ("no_shard", "optim_grads_params")


@dataclass
class OuterLayout:
    group: object
    rank: int  # this rank's index in group

    @property
    def is_replica(self):
        return self.rank != 0

    def sync_grad(self, grad):
        paddle.distributed.all_reduce(
            grad,
            op=paddle.distributed.ReduceOp.SUM,
            group=self.group,
            sync_op=True,
        )


def check_outer_strategy(name, strategy, hsdp_degree, default):
    if hsdp_degree is None or hsdp_degree == 1:
        if strategy is not None:
            raise ValueError(f"{name} needs hsdp_degree > 1.")
        return None
    strategy = default if strategy is None else strategy
    if strategy not in _OUTER_STRATEGIES:
        raise ValueError(f"Invalid {name}={strategy!r}.")
    return strategy


def build_hybrid_groups(
    sharding_group,
    expert_groups,
    hsdp_degree,
    dense_strategy,
    expert_strategy,
):
    """Return (dense inner group, {id(inner): OuterLayout},
    {id(expert group): expert inner group})."""
    outers, expert_inner = {}, {}
    if dense_strategy is None:
        return sharding_group, outers, expert_inner
    inner_size, dense_inner = _build_dense_groups(
        sharding_group, hsdp_degree, dense_strategy, outers
    )
    if expert_strategy == "no_shard":
        expert_inner = _build_expert_groups(
            sharding_group, expert_groups, inner_size, outers
        )
    return dense_inner, outers, expert_inner


def _build_dense_groups(sharding_group, hsdp_degree, dense_strategy, outers):
    ranks = list(sharding_group.ranks)
    if len(ranks) != paddle.distributed.get_world_size():
        raise NotImplementedError("Hybrid sharding needs one sharding group.")
    if (
        hsdp_degree < 1
        or len(ranks) % hsdp_degree
        or len(ranks) // hsdp_degree < 2
    ):
        raise ValueError(f"Invalid hsdp_degree={hsdp_degree}.")
    inner_size = len(ranks) // hsdp_degree
    hosts = []
    paddle.distributed.all_gather_object(
        hosts, socket.gethostname(), group=sharding_group
    )
    for i in range(0, len(ranks), inner_size):
        block = set(hosts[i : i + inner_size])
        if len(block) != 1:
            raise ValueError(f"Inner group spans hosts {sorted(block)}.")
    if dense_strategy != "no_shard":
        return inner_size, sharding_group
    idx = ranks.index(paddle.distributed.get_rank())
    # Every rank creates every group, in the same order.
    inners = [
        paddle.distributed.new_group(ranks[i : i + inner_size])
        for i in range(0, len(ranks), inner_size)
    ]
    inner = inners[idx // inner_size]
    outer = [
        paddle.distributed.new_group(ranks[i::inner_size])
        for i in range(inner_size)
    ][idx % inner_size]
    outers[id(inner)] = OuterLayout(outer, paddle.distributed.get_rank(outer))
    return inner_size, inner


def _build_expert_groups(sharding_group, expert_groups, inner_size, outers):
    """Experts unsharded, replicated over their expert group; needs one
    expert rank per inner group."""
    ranks = list(sharding_group.ranks)
    block = {r: ranks.index(r) // inner_size for r in ranks}
    counts = set()
    for group in expert_groups:
        per_block = Counter(block[r] for r in group.ranks)
        counts.update(per_block.values())
    # 0: no experts, 1: supported, 2: unsupported.
    if not counts:
        local = 0
    elif counts == {1} and len(expert_groups) == 1:
        local = 1
    else:
        local = 2
    lo, hi = paddle.to_tensor(local), paddle.to_tensor(local)
    paddle.distributed.all_reduce(
        lo, op=paddle.distributed.ReduceOp.MIN, group=sharding_group
    )
    paddle.distributed.all_reduce(
        hi, op=paddle.distributed.ReduceOp.MAX, group=sharding_group
    )
    if int(hi) == 0:
        return {}
    if int(lo) != 1 or int(hi) != 1:
        raise NotImplementedError("Unsupported expert layout for no_shard.")
    # Every rank creates every one-rank group, in the same order.
    singles = [paddle.distributed.new_group([r]) for r in ranks]
    inner = singles[ranks.index(paddle.distributed.get_rank())]
    (outer,) = expert_groups
    outers[id(inner)] = OuterLayout(outer, paddle.distributed.get_rank(outer))
    return {id(outer): inner}
