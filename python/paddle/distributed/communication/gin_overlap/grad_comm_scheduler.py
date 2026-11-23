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

"""Builds the GinGradCommScheduler of a sharding-stage1 optimizer: picks and checks the buckets, lays out the windows,
the recv pool and the host windows, moves each bucket's grad_storage into its send window, and hands the tables to
the C++ scheduler (paddle/fluid/distributed/collective/gin_overlap/grad_comm_scheduler.h).

Layout
  - Each group (sharding group for dense buckets, moe_sharding group for expert buckets) has a send window (the
    grad_storage of its buckets, 256 B aligned each), a recv window (the recv pool) and a host window (the u64
    counters).
  - Recv pool: dense = [intra-node segment][inter-node segment], expert = [expert segment]. Each segment is a ring:
    buckets take one slot each in the fixed release order and wrap to the start when the rest does not fit. The
    predecessor distance d of a slot (1..n, n = the bucket itself one step earlier) gives the credit target.
  - Host window, dense group: one signal counter per dense bucket; 16 intra-node credits; one inter-node credit per
    node; 16 intra-node arrival counters per dense bucket; the dense 2D-gather arrival counters. Expert group: one
    signal counter per expert bucket; one expert credit per group rank; the expert 2D-gather arrival counters.
  - The layout must be identical on all ranks (dense over the sharding group, expert over the moe_sharding group);
    init checks it with a crc.
Buckets have one index across both kinds: dense buckets 0..nd-1, expert buckets nd..nd+ne-1
(``bucket._gin_bucket_index``). The layout uses the index within the kind.
"""

import os
import statistics
import zlib
from collections import Counter

import paddle
import paddle.distributed as dist
from paddle.base import core
from paddle.distributed.fleet.utils.log_util import logger
from paddle.utils import strtobool

_LIB = getattr(core, "gin_overlap", None)

INTRA_COPY = _LIB.INTRA_COPY if _LIB is not None else None
INTRA_SUM = _LIB.INTRA_SUM if _LIB is not None else None
PUT = _LIB.PUT if _LIB is not None else None
FINAL_SUM = _LIB.FINAL_SUM if _LIB is not None else None
GATHER = _LIB.GATHER if _LIB is not None else None
GinGradCommScheduler = _LIB.GinGradCommScheduler if _LIB is not None else None

_BUILD_HINT = (
    "this Paddle was built without gin_overlap (WITH_GIN_OVERLAP): it needs WITH_DISTRIBUTE, WITH_GPU and WITH_NCCL, "
    "CUDA >= 13.0, NCCL >= 2.30.7 with nccl_device.h, and a CUDA arch >= 9.0"
)

# GIN contexts requested for each device communicator (a hint).
GIN_CONTEXTS = 8
# Size of the mark ring of release().
NUM_WINDOW_MARKS = 1024
# Tail marks per step = number of buckets + TAIL_EXTRA_MARKS: one for the hand-over, one per bucket reported after
# it, one per release_async() after it.
TAIL_EXTRA_MARKS = 1024
# Intra-node arrival counters per dense bucket (ranks per node <= 16).
_MAX_RANKS_PER_NODE = 16

# The GinGradCommScheduler of this process, None when gin_overlap is off. Set once by build_grad_comm_scheduler (one
# sharding optimizer per process). Read it on this module each time (grad_comm_scheduler.ACTIVE_SCHEDULER): a
# ``from ... import ACTIVE_SCHEDULER`` copies the value at import time, which is None.
ACTIVE_SCHEDULER = None


def _sharding_configs():
    from paddle.distributed import fleet

    return fleet.fleet._user_defined_strategy.hybrid_configs["sharding_configs"]


def _align(n, a=256):
    """n rounded up to a multiple of a. The actual requirements: dense send ranges and intra-node recv slots 16 B
    (float4 loads / stores of the sum kernels), counters 8 B (u64 atomics), window base and size 4096 B
    (kNCCLWindowAlignment; ncclMemAlloc and AllocHostWindow already pad the size). 256 is only a convenient
    multiple of the first two."""
    return (n + a - 1) // a * a


def _all_ranks_same(checks, world):
    """Collective: whether every (obj, group) of checks is identical on all ranks of its group (crc32 of repr(obj),
    one all_reduce MAX per group), combined with an all_reduce MIN over world."""
    all_same = True
    for obj, group in checks:
        crc = zlib.crc32(repr(obj).encode()) & 0x7FFFFFFF
        crc_max_and_neg_min = paddle.to_tensor([crc, -crc], dtype="int64")
        dist.all_reduce(crc_max_and_neg_min, op=dist.ReduceOp.MAX, group=group)
        all_same = (
            int(crc_max_and_neg_min[0]) == -int(crc_max_and_neg_min[1])
            and all_same
        )
    all_same_flag = paddle.to_tensor([int(all_same)], dtype="int32")
    dist.all_reduce(all_same_flag, op=dist.ReduceOp.MIN, group=world)
    return int(all_same_flag[0]) == 1


# ============================================================ unsupported configurations (raise at init)
_NO_HOOK_COLORS = ("dense_weight_no_hook", "moe_weight_no_hook")


def _check_compat(opt, where):
    """Configurations that raise for every sharding optimizer."""
    if strtobool(os.getenv("FLAGS_pp_check_naninf", "0")):
        raise RuntimeError(
            f"gin_overlap ({where}) does not support FLAGS_pp_check_naninf=1: the check scans a whole bucket before "
            "its communication, but gin_overlap already sends buckets during the backward pass, so it would read "
            "partial sums written by the side streams and may report a false inf/nan"
        )
    if dist.in_auto_parallel_align_mode():
        raise RuntimeError(
            f"gin_overlap ({where}) does not support align mode (FLAGS_enable_auto_parallel_align_mode=True)"
        )
    hcg = opt._hcg
    if hcg.get_data_parallel_world_size() > 1:
        raise RuntimeError(
            f"gin_overlap ({where}) does not support dp_degree > 1 (got {hcg.get_data_parallel_world_size()}): "
            "the gradients would still need a data-parallel all_reduce after the sharding reduce"
        )
    if getattr(opt, "comm_overlap", False) or getattr(opt, "pp_overlap", False):
        raise RuntimeError(
            f"gin_overlap ({where}) cannot be combined with sharding_configs.comm_overlap or "
            "pp_configs.sharding_comm_overlap: both move the gradient communication of the same buckets"
        )
    if getattr(opt, "sd_release_grads", False):
        raise RuntimeError(
            f"gin_overlap ({where}) does not support release_gradients: grad_storage is reallocated every step, "
            "but a bucket's grad_storage is its range of a send window registered once at init"
        )


def _is_expert_param(p):
    """Expert parameter: is_moe_param, or color "moe_expert". Expert parameters are reduced over the moe_sharding
    group."""
    color = getattr(p, "color", None)
    return getattr(p, "is_moe_param", False) or (
        isinstance(color, dict) and color.get("color") == "moe_expert"
    )


def _skip_reason(b, act, sync_ids=()):
    """Why bucket b cannot be taken over; None = it can."""
    if id(b) in sync_ids:
        return (
            "shared-parameter bucket (no-hook color, in _sync_only_buffer_ids)"
        )
    if b._act != act:
        return f"act={b._act} (only {act} is supported for this kind of bucket)"
    if b._comm_group is None or getattr(b._comm_group, "nranks", 1) <= 1:
        return (
            f"the communication group has {getattr(b._comm_group, 'nranks', 1)} rank: there is nothing to overlap "
            "(usually expert parallelism covers every GPU, so experts are not sharded)"
        )
    if b._release_grads or b._free_grads_in_comm:
        return f"release_grads={b._release_grads} free_grads_in_comm={b._free_grads_in_comm}"
    no_hook_colors = [
        p.color["color"]
        for p in b._params
        if isinstance(getattr(p, "color", None), dict)
        and p.color.get("color") in _NO_HOOK_COLORS
    ]
    if no_hook_colors:
        return (
            f'parameters with the no-hook color "{no_hook_colors[0]}": they are scaled as experts but classified '
            "into dense buckets"
        )
    if b.grad_storage is None:
        return "grad_storage is None"
    if b.grad_storage.dtype != paddle.float32 or not b.use_main_grad:
        return f"gradients are not fp32 main_grad (dtype={b.grad_storage.dtype} use_main_grad={b.use_main_grad})"
    if not b.need_reduce_scale_sync():
        return "all parameters of the bucket are frozen (need_reduce_scale_sync() is False), it is never sent"
    if not b._scale_after_comm and not b._use_reduce_avg:
        return (
            f"scale_after_comm={b._scale_after_comm} use_reduce_avg={b._use_reduce_avg}: the bucket is scaled in "
            "place before every _comm_grads call, and reduce_gradients calls _comm_grads again for buckets "
            "reported in the backward pass"
        )
    return None


def _release_order(opt, buckets):
    """Fixed release order (bucket indices within the kind), an estimate of the order in which buckets fill up in
    the backward pass: descending by the smallest index of the bucket's parameters in opt._parameter_list, ties
    broken by the larger bucket index. Affects only performance; all ranks get the same order. Raises on an empty
    bucket or a bucket parameter that is not in opt._parameter_list."""
    position = {p.name: i for i, p in enumerate(opt._parameter_list)}
    first_position = []
    for bucket in buckets:
        if not bucket._params:
            raise RuntimeError(
                "gin_overlap: a bucket has no parameters, so it has no place in the release order"
            )
        missing = [p.name for p in bucket._params if p.name not in position]
        if missing:
            raise RuntimeError(
                f"gin_overlap: {len(missing)} parameters of the bucket of {bucket._params[0].name} are not in the "
                f"optimizer's _parameter_list (e.g. {missing[:3]}), so the bucket has no place in the release order"
            )
        first_position.append(min(position[p.name] for p in bucket._params))
    return sorted(range(len(buckets)), key=lambda i: (-first_position[i], -i))


def _split_buckets(buckets, act, world, moe_group, sync_ids, where):
    """Splits buckets into (expert, dense). Raises on an unsupported bucket: one with a _skip_reason, one that mixes
    expert and non-expert parameters, a dense bucket outside the sharding group and an expert bucket outside the
    moe_sharding group (moe_group None = no expert parallelism)."""
    expert, dense, unsupported = [], [], []
    for bucket in buckets:
        reason = _skip_reason(bucket, act, sync_ids)
        if reason is None:
            expert_flags = [_is_expert_param(p) for p in bucket._params]
            if all(expert_flags):
                if (
                    moe_group is not None
                    and bucket._comm_group.id == moe_group.id
                ):
                    expert.append(bucket)
                    continue
                reason = "expert bucket outside the moe_sharding group" + (
                    " (no expert parallelism, so there is no moe_sharding group)"
                    if moe_group is None
                    else ""
                )
            elif any(expert_flags):
                reason = "mixes expert and non-expert parameters"
            elif bucket._comm_group.id != world.id:
                reason = (
                    "dense bucket outside the sharding group (its parameters carry their own color['group']); dense "
                    "buckets must all be in one group"
                )
            else:
                dense.append(bucket)
                continue
        unsupported.append((bucket, reason))
    if unsupported:
        names_by_reason = {}
        for bucket, reason in unsupported:
            names_by_reason.setdefault(reason, []).append(
                bucket._params[0].name
            )
        lines = [
            f"{len(names)}: {reason} (e.g. {names[:2]})"
            for reason, names in names_by_reason.items()
        ]
        raise RuntimeError(
            f"gin_overlap ({where}): {len(unsupported)}/{len(buckets)} buckets are not supported. With gin_overlap on "
            "every bucket must be taken over (no bucket falls back to NCCL).\n  "
            + "\n  ".join(lines)
        )
    return expert, dense


# ============================================================ layout
def _topology(group, what):
    """(group members per node, GPUs per node). GPUs per node = PADDLE_LOCAL_SIZE, or the number of visible devices
    when it is unset or not positive. The rank list must be ascending and every node must hold the same number of
    members."""
    ranks = list(group.ranks)
    local_size_env = os.getenv("PADDLE_LOCAL_SIZE", "").strip()
    try:
        local_size = int(local_size_env) if local_size_env else 0
    except ValueError:
        local_size = 0
    if local_size <= 0:
        local_size = max(1, paddle.device.cuda.device_count())
    if ranks != sorted(ranks):
        raise RuntimeError(
            f"gin_overlap: the rank list of the {what} group is not ascending ({ranks[:16]}...), so the node layout "
            "cannot be derived"
        )
    per_node = Counter(r // local_size for r in ranks)
    sizes = set(per_node.values())
    if len(sizes) != 1:
        raise RuntimeError(
            f"gin_overlap: the {what} group has a different number of members on different nodes (GPUs per node "
            f"PADDLE_LOCAL_SIZE={local_size}, members per node {dict(per_node)})"
        )
    ranks_per_node = sizes.pop()
    if (
        ranks_per_node < 1
        or ranks_per_node > _MAX_RANKS_PER_NODE
        or group.nranks % ranks_per_node
    ):
        raise RuntimeError(
            f"gin_overlap: the {what} group of {group.nranks} ranks has {ranks_per_node} members per node; it must "
            f"divide the group size and be at most {_MAX_RANKS_PER_NODE}"
        )
    return ranks_per_node, local_size


class _GroupLayout:
    """Node layout of one group: size, rank, ranks_per_node, rank_in_node, num_nodes, node_index."""

    def __init__(self, group, what):
        self.group, self.what = group, what
        self.size, self.rank = group.nranks, group.rank
        self.ranks_per_node, self.local_size = _topology(group, what)
        self.num_nodes = self.size // self.ranks_per_node
        self.rank_in_node = self.rank % self.ranks_per_node
        self.node_index = self.rank // self.ranks_per_node


def _send_layout(numels):
    """One 256 B aligned range per bucket (the sum kernels need 16 B, see _align). Returns (byte offsets, total
    bytes)."""
    offsets, total = [], 0
    for nbytes in [n * 4 for n in numels]:
        offsets.append(total)
        total += _align(nbytes)
    return offsets, total


def _recv_pool_ring(order, slot_bytes, size):
    """Ring layout of one recv pool segment: buckets take contiguous ranges in `order`; one that does not fit
    (cursor + slot > size) wraps to 0. Predecessor = among the earlier holders of the range of this bucket's slot,
    the one with the largest sequence number (step * n + position); the predecessor distance d is taken in step 2
    (1..n). Returns ({bucket: offset}, {bucket: d}); buckets with a slot of 0 bytes are absent."""
    num_buckets, offset, cursor = len(order), {}, 0
    for bucket in order:
        nbytes = slot_bytes[bucket]
        if nbytes <= 0:
            continue
        if nbytes > size:
            raise RuntimeError(
                f"gin_overlap: the slot of bucket {bucket} is {nbytes} B, larger than its segment ({size} B)"
            )
        if cursor + nbytes > size:
            cursor = 0
        offset[bucket] = cursor
        cursor += nbytes
    # occupied: sorted (begin, end, sequence number of the holder) ranges of the segment
    occupied, distance = [], {}
    for step in (0, 1):
        for position, bucket in enumerate(order):
            if bucket not in offset:
                continue
            begin = offset[bucket]
            end = begin + slot_bytes[bucket]
            sequence = step * num_buckets + position
            remaining, predecessor = [], -1
            for held_begin, held_end, holder in occupied:
                if held_end <= begin or held_begin >= end:
                    remaining.append((held_begin, held_end, holder))
                    continue
                predecessor = max(predecessor, holder)
                if held_begin < begin:
                    remaining.append((held_begin, begin, holder))
                if held_end > end:
                    remaining.append((end, held_end, holder))
            remaining.append((begin, end, sequence))
            occupied = sorted(remaining)
            if step == 1:
                if predecessor < 0:
                    raise RuntimeError(
                        f"gin_overlap: no predecessor found for bucket {bucket} (internal error)"
                    )
                distance[bucket] = sequence - predecessor
    return offset, distance


def _recv_pool_sizes(lower_bounds, bound_texts, dense_gb, expert_gb):
    """Sizes (bytes, multiples of 4 KB) of the three segments. The dense pool is split between the intra-node and
    inter-node segments in proportion to their lower bounds."""
    sizes = list(lower_bounds)
    dense_bound = lower_bounds[0] + lower_bounds[1]
    if dense_bound > 0:
        dense_pool = int(dense_gb * 2**30)
        if dense_pool < dense_bound:
            raise RuntimeError(
                f"gin_overlap_dense_recv_pool_gb={dense_gb:g} is too small: the lower bounds of the intra-node and "
                f"inter-node segments add up to {dense_bound / 2**30:.3f} GiB ({dense_bound} B); {bound_texts[0]}; "
                f"{bound_texts[1]}"
            )
        sizes[0] = (lower_bounds[0] * dense_pool // dense_bound) // 4096 * 4096
        sizes[1] = (lower_bounds[1] * dense_pool // dense_bound) // 4096 * 4096
        for i in (0, 1):
            if sizes[i] < lower_bounds[i]:
                raise RuntimeError(
                    f"gin_overlap_dense_recv_pool_gb={dense_gb:g}: after the split, {bound_texts[i]} is not met "
                    f"(got {sizes[i] / 2**30:.3f} GiB)"
                )
    if lower_bounds[2] > 0:
        expert_pool = int(expert_gb * 2**30) // 4096 * 4096
        if expert_pool < lower_bounds[2]:
            raise RuntimeError(
                f"gin_overlap_expert_recv_pool_gb={expert_gb:g} is too small: {bound_texts[2]}"
            )
        sizes[2] = expert_pool
    return sizes


class _Layout:
    """Window, recv pool and host window layout of both groups (see the module docstring). Collective: computes
    lower bounds with all_reduce and checks with a crc that every rank has the same layout."""

    def __init__(self, dense, expert, dense_numels, expert_numels, order, cfg):
        self.dense, self.expert = dense, expert
        self.dense_numels, self.expert_numels = (
            list(dense_numels),
            list(expert_numels),
        )
        self.dense_order = [int(i) for i in order[0]]
        self.expert_order = [int(i) for i in order[1]]
        num_dense, num_expert = len(self.dense_numels), len(self.expert_numels)
        if sorted(self.dense_order) != list(range(num_dense)) or sorted(
            self.expert_order
        ) != list(range(num_expert)):
            raise RuntimeError(
                f"gin_overlap: the release order is not a permutation of the bucket ids ({num_dense} dense / "
                f"{num_expert} expert)"
            )
        for numel in self.dense_numels:
            if numel % dense.size or (numel // dense.size) % 4:
                raise RuntimeError(
                    f"gin_overlap: dense bucket size {numel} must be a multiple of 4 x the sharding group size "
                    f"{dense.size}"
                )
        for numel in self.expert_numels:
            if numel % expert.size:
                raise RuntimeError(
                    f"gin_overlap: expert bucket size {numel} is not divisible by the moe_sharding group size "
                    f"{expert.size}"
                )
        self.dense_send, self.dense_send_bytes = _send_layout(self.dense_numels)
        self.expert_send, self.expert_send_bytes = _send_layout(
            self.expert_numels
        )
        self._recv_pool(cfg)
        self._host_windows()

    def _recv_pool(self, cfg):
        dense, expert = self.dense, self.expert
        ranks_per_node, num_nodes = dense.ranks_per_node, dense.num_nodes
        expert_size = max(expert.size, 1) if self.expert_numels else 1
        dense_shard_numels = [n // dense.size for n in self.dense_numels]
        expert_shard_numels = [n // expert_size for n in self.expert_numels]
        slot_bytes = [
            # intra-node segment: the rows pushed by the other ranks_per_node - 1 ranks
            [
                _align((ranks_per_node - 1) * num_nodes * shard_numel * 4)
                if ranks_per_node > 1
                else 0
                for shard_numel in dense_shard_numels
            ],
            # inter-node segment: num_nodes rows, this rank's row stays empty
            [
                _align(num_nodes * shard_numel * 4) if num_nodes > 1 else 0
                for shard_numel in dense_shard_numels
            ],
            # expert segment: expert_size rows, this rank's row stays empty
            [
                _align(expert_size * shard_numel * 4) if expert_size > 1 else 0
                for shard_numel in expert_shard_numels
            ],
        ]
        largest_bucket = [
            max(range(len(slots)), key=lambda i: slots[i]) if slots else -1
            for slots in slot_bytes
        ]
        local_bounds = [
            _align(max(slots, default=0), 4096) for slots in slot_bytes
        ]
        dense_bounds = paddle.to_tensor(local_bounds[:2], dtype="int64")
        dist.all_reduce(dense_bounds, op=dist.ReduceOp.MAX, group=dense.group)
        expert_bounds = paddle.to_tensor(local_bounds[2:], dtype="int64")
        if self.expert_numels:
            dist.all_reduce(
                expert_bounds, op=dist.ReduceOp.MAX, group=expert.group
            )
        lower_bounds = [
            int(dense_bounds[0]),
            int(dense_bounds[1]),
            int(expert_bounds[0]),
        ]
        names = (
            "dense intra-node segment",
            "dense inter-node segment",
            "expert segment",
        )
        bound_texts = [
            f"{names[i]} lower bound {lower_bounds[i] / 2**30:.3f} GiB (largest bucket on this rank "
            f"{'dense' if i < 2 else 'expert'}#{largest_bucket[i]}, slot {local_bounds[i] / 2**30:.3f} GiB)"
            for i in range(3)
        ]
        segment_bytes = _recv_pool_sizes(
            lower_bounds,
            bound_texts,
            cfg.gin_overlap_dense_recv_pool_gb,
            cfg.gin_overlap_expert_recv_pool_gb,
        )
        intra_offsets, intra_distance = _recv_pool_ring(
            self.dense_order, slot_bytes[0], segment_bytes[0]
        )
        inter_offsets, inter_distance = _recv_pool_ring(
            self.dense_order, slot_bytes[1], segment_bytes[1]
        )
        expert_offsets, expert_distance = _recv_pool_ring(
            self.expert_order, slot_bytes[2], segment_bytes[2]
        )
        # Senders compute the write offsets on their peers from their own table, so it must match on all ranks.
        dense_table = (
            segment_bytes[0], segment_bytes[1], ranks_per_node, num_nodes, self.dense_order, slot_bytes[0],
            slot_bytes[1], intra_offsets, intra_distance, inter_offsets, inter_distance,
        )  # fmt: skip
        checks = [(dense_table, dense.group)]
        if self.expert_numels:
            expert_table = (
                segment_bytes[2], expert_size, self.expert_order, slot_bytes[2], expert_offsets, expert_distance,
            )  # fmt: skip
            checks.append((expert_table, expert.group))
        if not _all_ranks_same(checks, dense.group):
            raise RuntimeError(
                "gin_overlap: the recv pool layout (segment sizes / release order / offsets / predecessors) differs "
                "between ranks (dense must match over the sharding group, expert over the moe_sharding group); "
                "check the recv pool sizes and the bucket split on every rank"
            )
        self.segment_bytes = segment_bytes
        self.lower_bounds = lower_bounds
        self.segment_slots = slot_bytes
        # start of each segment: intra-node / inter-node in the dense recv window, expert in the expert recv window
        self.base = [0, segment_bytes[0], 0]
        self.offsets = (intra_offsets, inter_offsets, expert_offsets)
        self.distance = (intra_distance, inter_distance, expert_distance)
        self.dense_recv_bytes = segment_bytes[0] + segment_bytes[1]
        self.expert_recv_bytes = segment_bytes[2]

    def _host_windows(self):
        """Host window layout of both groups, each from 0, every area 256 B aligned (the counters need 8 B for the
        u64 atomics; the window size is padded to 4096 B, kNCCLWindowAlignment, which AllocHostWindow also pads to).
        dense group: signal (one counter per dense bucket at 8*idx, counting the inter-node puts) | credit_intra (16
        counters, one per node-local sender) | credit_inter (one counter per node) | intra_arrive (16 counters per
        dense bucket at intra_arrive + (idx*16 + l)*8, written by node-local sender l) | gather arrive (8 B per
        dense bucket) | gather intra (16 counters per dense bucket).
        expert group: signal (one counter per expert bucket at 8*idx) | credit_expert (one counter per group rank) |
        gather arrive | gather intra, sized by the expert buckets."""
        num_dense, num_expert = len(self.dense_numels), len(self.expert_numels)
        credit_intra = _align(8 * num_dense, 256)
        credit_inter = _align(credit_intra + 8 * 16, 256)
        dense_credit_end = credit_inter + 8 * max(self.dense.num_nodes, 1)
        intra_arrive = _align(dense_credit_end, 256)
        gather_arrive = _align(intra_arrive + 8 * 16 * num_dense, 256)
        gather_intra = _align(gather_arrive + 8 * max(num_dense, 1), 256)
        self.intra_arrive = intra_arrive
        self.gather_dense = (gather_arrive, gather_intra)
        self.dense_host_bytes = _align(
            gather_intra + 8 * 16 * max(num_dense, 1), 4096
        )
        expert_size = self.expert.size if num_expert else 0
        credit_expert = _align(8 * num_expert, 256)
        expert_credit_end = credit_expert + 8 * max(expert_size, 1)
        expert_gather_arrive = _align(expert_credit_end, 256)
        expert_gather_intra = _align(
            expert_gather_arrive + 8 * max(num_expert, 1), 256
        )
        self.gather_expert = (expert_gather_arrive, expert_gather_intra)
        self.expert_host_bytes = _align(
            expert_gather_intra + 8 * 16 * max(num_expert, 1), 4096
        )
        self.credit = (credit_intra, credit_inter, credit_expert)

    def bucket_table(self):
        """One (kind, send_offset, signal_offset, shard_numel, arrive_offset, recv_offset_0, recv_offset_1,
        depth_0, depth_1) per bucket, dense buckets first. depth -1 = no slot in that segment."""
        rows = []
        for i, numel in enumerate(self.dense_numels):
            intra_distance = self.distance[0].get(i)
            inter_distance = self.distance[1].get(i)
            rows.append(
                (
                    _LIB.DENSE_BUCKET,
                    self.dense_send[i],
                    8 * i,
                    numel // self.dense.size,
                    self.intra_arrive + i * 16 * 8,
                    self.base[0] + self.offsets[0].get(i, 0),
                    self.base[1] + self.offsets[1].get(i, 0),
                    -1 if intra_distance is None else intra_distance,
                    -1 if inter_distance is None else inter_distance,
                )
            )
        for i, numel in enumerate(self.expert_numels):
            expert_distance = self.distance[2].get(i)
            rows.append(
                (
                    _LIB.EXPERT_BUCKET,
                    self.expert_send[i],
                    8 * i,
                    numel // self.expert.size,
                    0,
                    self.base[2] + self.offsets[2].get(i, 0),
                    0,
                    -1 if expert_distance is None else expert_distance,
                    -1,
                )
            )
        return rows

    def log(self, windows):
        names = (
            "dense intra-node segment",
            "dense inter-node segment",
            "expert segment",
        )
        segment_texts = []
        for i in range(3):
            slots = [x for x in self.segment_slots[i] if x > 0]
            median_slot = statistics.median(slots) if slots else 0
            slots_held = (
                int(self.segment_bytes[i] // median_slot) if median_slot else 0
            )
            segment_texts.append(
                f"{names[i]} {self.segment_bytes[i] / 2**30:.3f} GiB (lower bound "
                f"{self.lower_bounds[i] / 2**30:.3f}, median slot {median_slot / 2**20:.1f} MiB, holds {slots_held})"
            )
        num_buckets = (
            len(self.dense_order),
            len(self.dense_order),
            len(self.expert_order),
        )
        same_step_predecessors = [
            sum(1 for d in self.distance[i].values() if d < num_buckets[i])
            for i in range(3)
        ]
        if dist.get_rank() == 0:
            logger.info(
                f"[gin_overlap] init: {len(self.dense_numels)} dense buckets (sharding group {self.dense.num_nodes} "
                f"nodes x {self.dense.ranks_per_node}), {len(self.expert_numels)} expert buckets; recv pool: "
                + "; ".join(segment_texts)
                + f"; predecessors in the same step: intra-node {same_step_predecessors[0]} / inter-node "
                f"{same_step_predecessors[1]} / expert {same_step_predecessors[2]}; windows {windows}"
            )  # fmt: skip


class _GroupWindows:
    """Windows of one group (collective over the group): send and recv windows from ncclMemAlloc, registered with
    register_comm_buffer and zeroed; the host window from alloc_host_window; then the device communicator, whose LSA
    team must match the node layout."""

    def __init__(self, layout, send_bytes, recv_bytes, host_bytes):
        from paddle.distributed.collective import _nccl_symmetric_empty

        pg = layout.group.process_group
        self.layout = layout
        self.send = _nccl_symmetric_empty(
            [max(send_bytes, 4096) // 4], paddle.float32
        )
        self.send.zero_()
        self.send_window = pg.register_comm_buffer(self.send)
        self.recv = _nccl_symmetric_empty(
            [max(recv_bytes, 4096) // 4], paddle.float32
        )
        self.recv.zero_()
        self.recv_window = pg.register_comm_buffer(self.recv)
        paddle.device.synchronize()
        self.host_window, _, self.host_bytes = pg.alloc_host_window(host_bytes)
        if not (self.send_window and self.recv_window and self.host_window):
            raise RuntimeError(
                f"gin_overlap: window registration of the {layout.what} group returned no window (the loaded NCCL "
                "has no window API)"
            )
        dev_comm = pg.get_or_create_dev_comm(GIN_CONTEXTS)
        dev_comm_usable = int(
            dev_comm["gin_context_count"] >= 1
            and dev_comm["lsa_size"] == layout.ranks_per_node
            and dev_comm["lsa_rank"] == layout.rank_in_node
        )
        # MIN of [usable, -ranks_per_node, ranks_per_node]: all usable, and the max and min of ranks_per_node agree
        usable_and_ranks_per_node = paddle.to_tensor(
            [dev_comm_usable, -layout.ranks_per_node, layout.ranks_per_node],
            dtype="int32",
        )
        dist.all_reduce(
            usable_and_ranks_per_node,
            op=dist.ReduceOp.MIN,
            group=layout.group,
        )
        if int(usable_and_ranks_per_node[0]) != 1 or int(
            -usable_and_ranks_per_node[1]
        ) != int(usable_and_ranks_per_node[2]):
            raise RuntimeError(
                f"gin_overlap: the device communicator of the {layout.what} group is unusable, or its LSA team "
                f"differs from the node layout (GIN contexts {dev_comm['gin_context_count']}, lsa_rank "
                f"{dev_comm['lsa_rank']}, lsa_size {dev_comm['lsa_size']}; from the rank list rank_in_node "
                f"{layout.rank_in_node}, ranks_per_node {layout.ranks_per_node}; GPUs per node {layout.local_size})"
            )
        self.gin_contexts = dev_comm["gin_context_count"]

    def handles(self):
        return (self.send_window, self.recv_window, self.host_window)


# ============================================================ grad_storage -> send window
def _alias_1d(b, window, offset):
    """Replaces grad_storage of a 1D bucket with its range of the send window and re-links each parameter's
    main_grad / _slice_grad. The old grad_storage is copied over, then released."""
    numel = b.grad_storage._numel()
    begin = offset // 4
    new = window.send._slice(begin, begin + numel)
    new.copy_(b.grad_storage, False)
    b.grad_storage = new
    for p in b._params:
        grad_view = b._sharding_param_grad_view[p.name]
        grad_view._grad_buffer = new
        grad_view._link_grad_to_buffer()
    b._record_addr()
    return new.data_ptr()


def _alias_2d(b, window, offset, padded_numel):
    """Moves a Muon 2D bucket into the send window: the range holds padded_numel fp32, the first numel copied from
    the old grad_storage and [numel, padded_numel) zero; each parameter's main_grad is re-sliced at its element
    offset; grad_storage becomes the whole padded_numel range."""
    old = b.grad_storage
    numel = old._numel()
    if numel > padded_numel:
        raise RuntimeError(
            f"gin_overlap: 2D bucket of {b._params[0].name}: {numel} elements, more than its padded size {padded_numel}"
        )
    begin = offset // 4
    new = window.send._slice(begin, begin + padded_numel)
    src = (
        old
        if padded_numel == numel
        else paddle.concat(
            [old, paddle.zeros([padded_numel - numel], dtype=paddle.float32)]
        )
    )
    new.copy_(src, False)
    del src
    for p in b._params:
        main_grad = p.main_grad
        if main_grad is None or main_grad._numel() != p._numel():
            raise RuntimeError(
                f"gin_overlap: 2D bucket: main_grad of {p.name} is missing or does not have {p._numel()} elements"
            )
        eoff = (main_grad.data_ptr() - old.data_ptr()) // 4
        if eoff < 0 or eoff + p._numel() > numel:
            raise RuntimeError(
                f"gin_overlap: 2D bucket: main_grad of {p.name} starts at element {eoff}, outside the bucket's "
                f"{numel} elements"
            )
        new_main_grad = new._slice(eoff, eoff + p._numel())
        new_main_grad.get_tensor()._set_dims(p.shape)
        p.main_grad = new_main_grad
    b.grad_storage = new
    b._record_addr()
    return new.data_ptr()


# ============================================================ Muon 2D gather table
def _gather_table(layout, dense_2d, expert_2d, dense_index, expert_index):
    """One (unified bucket index, owner group rank, arrive_offset, intra_arrive_offset) per Muon 2D bucket.
    arrive_offset = inter-node arrival counter of the bucket on the owner, intra_arrive_offset = start of its 16
    same-node arrival counters. Collective: the table must match over each group (crc)."""
    num_dense = len(layout.dense_numels)
    rows, bucket_tables = [], {"dense": [], "expert": []}
    for kind, buckets, index, group_layout, (gather_arrive, gather_intra) in (
        ("dense", dense_2d, dense_index, layout.dense, layout.gather_dense),
        (
            "expert",
            expert_2d,
            expert_index,
            layout.expert,
            layout.gather_expert,
        ),
    ):
        for bucket in buckets:
            kind_index = index[id(bucket)]
            _, padded_numel, owner = bucket._gin_owner_bucket
            want = (layout.dense_numels if kind == "dense" else layout.expert_numels)[kind_index]  # fmt: skip
            if padded_numel != want:
                raise RuntimeError(
                    f"gin_overlap: 2D {kind} bucket {kind_index} has padded size {padded_numel}, the layout has {want}"
                )
            send = (layout.dense_send if kind == "dense" else layout.expert_send)[kind_index]  # fmt: skip
            unified = kind_index if kind == "dense" else num_dense + kind_index
            rows.append(
                (
                    unified,
                    owner,
                    gather_arrive + 8 * kind_index,
                    gather_intra + kind_index * 16 * 8,
                )
            )
            bucket_tables[kind].append(
                (kind_index, send, padded_numel // group_layout.size, owner)
            )
    checks = [
        (
            (layout.gather_dense, num_dense, layout.dense.ranks_per_node, sorted(bucket_tables["dense"])),
            layout.dense.group,
        )
    ]  # fmt: skip
    if layout.expert_numels:
        checks.append(
            (
                (layout.gather_expert, len(layout.expert_numels), layout.expert.ranks_per_node,
                 sorted(bucket_tables["expert"])),
                layout.expert.group,
            )
        )  # fmt: skip
    if not _all_ranks_same(checks, layout.dense.group):
        raise RuntimeError(
            "gin_overlap: the 2D-bucket table (bucket / send offset / shard size / owner) or the gather counter "
            "offsets differ between ranks (dense over the sharding group, expert over the moe_sharding group)"
        )
    return rows


# ============================================================ build
def _select_v2(opt, world, moe_group):
    where = "DygraphShardingOptimizerV2"
    expert, dense = _split_buckets(
        opt._comm_buffer_list,
        _hook_action().REDUCE_SCATTER,
        world,
        moe_group,
        (),
        where,
    )
    return expert, dense, [], []


def _select_muon(opt, world, moe_group):
    where = "MuonShardingOptimizer"
    if not opt._use_fuse_gradients:
        raise RuntimeError(
            f"gin_overlap ({where}) needs comm_buffer_size_MB > 0: 2D parameters must be in fused buckets"
        )
    if opt.use_group_call_opt:
        raise RuntimeError(
            f"gin_overlap ({where}) cannot be combined with sharding_configs.comm_group_call_opt"
        )
    act = _hook_action()
    sync_ids = opt._sync_only_buffer_ids
    expert_1d, dense_1d = _split_buckets(
        opt._comm_buffer_list, act.REDUCE_SCATTER, world, moe_group, sync_ids, f"{where} / 1D buckets"
    )  # fmt: skip
    expert_2d, dense_2d = _split_buckets(
        opt.comm_buffer_2d, act.REDUCE, world, moe_group, sync_ids, f"{where} / 2D buckets"
    )  # fmt: skip
    dense_size = world.nranks
    expert_size = moe_group.nranks if (expert_1d or expert_2d) else 1
    for bucket in dense_2d:
        numel = bucket.grad_storage._numel()
        # (numel, numel padded to a multiple of 4 x group size, owner's group rank)
        bucket._gin_owner_bucket = (
            numel,
            _align(numel, 4 * dense_size),
            world.ranks.index(bucket._dst),
        )
    for bucket in expert_2d:
        numel = bucket.grad_storage._numel()
        bucket._gin_owner_bucket = (
            numel,
            _align(numel, 4 * expert_size),
            moe_group.ranks.index(bucket._dst),
        )
    return expert_1d + expert_2d, dense_1d + dense_2d, expert_2d, dense_2d


def _hook_action():
    from paddle.distributed.fleet.utils.tensor_fusion_helper import HOOK_ACTION

    return HOOK_ACTION


def build_grad_comm_scheduler(opt):
    """Builds the GinGradCommScheduler of a DygraphShardingOptimizerV2 or MuonShardingOptimizer, after its buckets
    are built, and sets it as ACTIVE_SCHEDULER. Collective over the sharding group and, with expert buckets,
    the moe_sharding group. Raises on an unsupported configuration or bucket.
      - Every bucket is taken over: expert buckets (all parameters expert: is_moe_param or color "moe_expert") in
        the moe_sharding group, dense buckets (no expert parameter) in the sharding group. Muon 2D buckets are
        padded to a multiple of 4 x group size and gathered to their owner after the reduce.
      - grad_storage of each bucket becomes its range of the send window; main_grad is re-linked.
      - Each bucket gets ``_gin_bucket_index`` (unified index) and ``_gin_scheduler``; each parameter gets a backward
        hook that reports its bucket once the bucket is full, i.e. after sharding_configs.accumulate_steps gradients
        per parameter (the bucket's _acc_steps, set when the sharding optimizer creates the bucket). The trainer must
        set sharding_configs.accumulate_steps to its gradient accumulation steps."""
    global ACTIVE_SCHEDULER
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
        DygraphShardingOptimizerV2,
    )

    if _LIB is None:
        raise RuntimeError(
            f"sharding_configs.gin_overlap is on, but {_BUILD_HINT}"
        )
    if ACTIVE_SCHEDULER is not None:
        raise RuntimeError(
            "gin_overlap: a GinGradCommScheduler is already built in this process (one sharding optimizer per "
            "process)"
        )
    is_v2 = isinstance(opt, DygraphShardingOptimizerV2)
    where = "DygraphShardingOptimizerV2" if is_v2 else "MuonShardingOptimizer"
    _check_compat(opt, where)
    cfg = _sharding_configs()
    if not cfg.gin_overlap_timeout > 0:
        raise ValueError(
            f"sharding_configs.gin_overlap_timeout must be positive, got {cfg.gin_overlap_timeout}"
        )
    hcg = opt._hcg
    world = hcg.get_sharding_parallel_group()
    # only the hcg of expert parallelism (EPHybridCommunicateGroup) has a moe_sharding group
    moe_group = (
        hcg.get_moe_sharding_parallel_group()
        if hasattr(hcg, "get_moe_sharding_parallel_group")
        else None
    )
    select = _select_v2 if is_v2 else _select_muon
    expert, dense, expert_2d, dense_2d = select(opt, world, moe_group)
    if not dense:
        raise RuntimeError(
            f"gin_overlap ({where}): there is no dense bucket; check comm_buffer_size_MB > 0 and split_param"
        )
    dense_layout = _GroupLayout(world, "sharding")
    expert_layout = _GroupLayout(moe_group, "moe_sharding") if expert else None
    order = (_release_order(opt, dense), _release_order(opt, expert))
    layout = _Layout(
        dense_layout,
        expert_layout,
        [
            b._gin_owner_bucket[1]
            if getattr(b, "_gin_owner_bucket", None) is not None
            else b.grad_storage._numel()
            for b in dense
        ],
        [
            b._gin_owner_bucket[1]
            if getattr(b, "_gin_owner_bucket", None) is not None
            else b.grad_storage._numel()
            for b in expert
        ],
        order,
        cfg,
    )
    dense_windows = _GroupWindows(
        dense_layout,
        layout.dense_send_bytes,
        layout.dense_recv_bytes,
        layout.dense_host_bytes,
    )
    expert_windows = None
    if expert:
        expert_windows = _GroupWindows(
            expert_layout,
            layout.expert_send_bytes,
            layout.expert_recv_bytes,
            layout.expert_host_bytes,
        )
    dense_index = {id(b): i for i, b in enumerate(dense)}
    expert_index = {id(b): i for i, b in enumerate(expert)}
    gather = _gather_table(
        layout, dense_2d, expert_2d, dense_index, expert_index
    )

    # grad_storage -> send window, before the first backward pass
    num_dense = len(dense)
    for kind, buckets, windows, offsets in (
        ("dense", dense, dense_windows, layout.dense_send),
        ("expert", expert, expert_windows, layout.expert_send),
    ):
        for i, b in enumerate(buckets):
            if getattr(b, "_gin_owner_bucket", None) is not None:
                ptr = _alias_2d(b, windows, offsets[i], b._gin_owner_bucket[1])
            else:
                ptr = _alias_1d(b, windows, offsets[i])
            b._gin_bucket_index = i if kind == "dense" else num_dense + i
            b._gin_grad_ptr = ptr
    paddle.device.synchronize()

    credit_intra, credit_inter, credit_expert = layout.credit
    scheduler = GinGradCommScheduler(
        dense_group=world.process_group,
        expert_group=moe_group.process_group if expert else None,
        dense_windows=dense_windows.handles(),
        expert_windows=expert_windows.handles() if expert else (0, 0, 0),
        buckets=layout.bucket_table(),
        dense_order=layout.dense_order,
        expert_order=[num_dense + i for i in layout.expert_order],
        credit_intra_offset=credit_intra,
        credit_inter_offset=credit_inter,
        credit_expert_offset=credit_expert,
        gather=gather,
        num_window_marks=NUM_WINDOW_MARKS,
        num_tail_marks=num_dense + len(expert) + TAIL_EXTRA_MARKS,
    )
    # keep the windows alive as long as the scheduler
    scheduler_state = (layout, dense_windows, expert_windows)

    hook_kw = {"use_comm": True} if is_v2 else {}
    for b in dense + expert:
        b._gin_scheduler = scheduler
        b._gin_reported = False
        for param in b._params:
            param._register_backward_hook(
                opt._create_backward_hook(b, param, **hook_kw)
            )
    opt._gin_scheduler = scheduler
    opt._gin_scheduler_state = scheduler_state
    opt._gin_buckets = dense + expert
    dist.barrier(group=world)
    windows = "; ".join(
        f"{group_windows.layout.what} group send {group_windows.send._numel() * 4 / 2**30:.3f} + recv "
        f"{group_windows.recv._numel() * 4 / 2**30:.3f} GiB, host {group_windows.host_bytes} B, GIN contexts "
        f"{group_windows.gin_contexts}"
        for group_windows in (dense_windows, expert_windows)
        if group_windows is not None
    )
    layout.log(windows)
    if gather and dist.get_rank() == 0:
        logger.info(
            f"[gin_overlap] init: {len(gather)} Muon 2D buckets gathered to their owners (dense {len(dense_2d)} / "
            f"expert {len(expert_2d)})"
        )
    ACTIVE_SCHEDULER = scheduler
    return scheduler


# ============================================================ calls from FusedCommBuffer and the optimizers
def bucket_ready(bucket, reduce_op):
    """Reports a full bucket (FusedCommBuffer._comm_grads). The scale is the factor of reduce_op: 1/N for AVG, 1 for
    SUM. A second call within the step does nothing."""
    if bucket._gin_reported:
        return
    grad_storage = bucket.grad_storage
    if grad_storage is None or grad_storage.data_ptr() != bucket._gin_grad_ptr:
        raise RuntimeError(
            f"gin_overlap: grad_storage of bucket {bucket._gin_bucket_index} is no longer its send window range"
        )
    scale = (
        1.0 / bucket._comm_group.nranks
        if reduce_op == dist.ReduceOp.AVG
        else 1.0
    )
    bucket._gin_scheduler.bucket_ready(bucket._gin_bucket_index, scale)
    bucket._gin_reported = True


def finish_step(opt):
    """End of reduce_gradients: every bucket went through _comm_grads. Waits for the step (FinishStep)."""
    opt._gin_scheduler.finish_step(
        float(_sharding_configs().gin_overlap_timeout)
    )


def fused_all_gather(buckets):
    """Parameter all_gather of the REDUCE_SCATTER buckets, fused per (communication group, dtype): this rank's
    shards of a batch of buckets are concatenated, sent with one process_group.all_gather and split back into each
    bucket's param_storage. A batch holds at most sharding_configs.param_allgather_fuse_mb MB (at least one bucket).
    param_storage of a bucket is [rank 0 part, rank 1 part, ...], so a bucket with part_numel elements per part
    takes columns [column, column + part_numel) of each row of recv[nranks, total] and row r goes to
    param_storage[r * part_numel, (r + 1) * part_numel)."""
    from paddle.distributed.fleet.utils.tensor_fusion_helper import HOOK_ACTION

    limit = max(1, int(_sharding_configs().param_allgather_fuse_mb)) * 2**20
    groups = {}
    for b in buckets:
        if (
            not b.need_reduce_scale_sync()
            or b._act != HOOK_ACTION.REDUCE_SCATTER
            or b._comm_group.nranks <= 1
        ):
            continue
        groups.setdefault(
            (b._comm_group.id, str(b.param_storage.dtype)), []
        ).append(b)
    for group_buckets in groups.values():
        group = group_buckets[0]._comm_group
        nranks, rank = group.nranks, max(group.rank, 0)
        chunk, nbytes = [], 0
        for b in [*group_buckets, None]:
            if b is not None:
                bucket_bytes = (
                    b.param_storage._numel() * b.param_storage.element_size()
                )
                if not chunk or nbytes + bucket_bytes <= limit:
                    chunk.append(b)
                    nbytes += bucket_bytes
                    continue
            _all_gather_chunk(chunk, group, nranks, rank)
            if b is None:
                break
            chunk, nbytes = [b], bucket_bytes


def _all_gather_chunk(chunk, group, nranks, rank):
    part_numels = [b.param_storage._numel() // nranks for b in chunk]
    for b, part_numel in zip(chunk, part_numels):
        if b.param_storage._numel() != nranks * part_numel:
            raise RuntimeError(
                f"gin_overlap: param_storage of the bucket of {b._params[0].name} has {b.param_storage._numel()} "
                f"elements, not a multiple of the group size {nranks}"
            )
    send = paddle.concat(
        [
            b.param_storage._slice(part_numel * rank, part_numel * (rank + 1))
            for b, part_numel in zip(chunk, part_numels)
        ]
    )
    total = send._numel()
    recv = paddle.empty([nranks * total], dtype=send.dtype)
    group.process_group.all_gather(send, recv).wait()
    column = 0
    for b, part_numel in zip(chunk, part_numels):
        if part_numel:
            for src_rank in range(nranks):
                row = src_rank * total + column
                b.param_storage._slice(
                    src_rank * part_numel, (src_rank + 1) * part_numel
                ).copy_(recv._slice(row, row + part_numel), False)
        column += part_numel
    del send, recv
