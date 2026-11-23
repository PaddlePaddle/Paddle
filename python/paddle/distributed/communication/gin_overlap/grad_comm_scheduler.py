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
import sys
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

# The scheduler of this process (one sharding optimizer per process).
_ACTIVE = None
# Whether the first reduce_gradients has re-checked FusedCommBuffer._comm_grads.
_COMM_GRADS_CHECKED = False


def active_scheduler():
    """The GinGradCommScheduler of this process, or None when gin_overlap is off."""
    return _ACTIVE


def _sharding_configs():
    from paddle.distributed import fleet

    return fleet.fleet._user_defined_strategy.hybrid_configs["sharding_configs"]


def _accumulate_steps():
    """Micro-steps per step: pipeline_configs.accumulate_steps with pipeline parallelism, otherwise
    sharding_configs.accumulate_steps. A bucket is full, and reported, once each of its parameters has received this
    many gradients."""
    from paddle.distributed import fleet

    strategy = fleet.fleet._user_defined_strategy
    if strategy.hybrid_configs["pp_degree"] > 1:
        return max(1, int(strategy.pipeline_configs["accumulate_steps"]))
    return max(1, int(_sharding_configs().accumulate_steps))


def _align(n, a=256):
    return (n + a - 1) // a * a


def _gib(x):
    return format(x / 2**30, ".3f")


def _log(msg):
    if dist.get_rank() == 0:
        logger.info(f"[gin_overlap] {msg}")


def _local_size():
    """GPUs per node: PADDLE_LOCAL_SIZE, or the number of visible devices when it is unset or not positive."""
    v = os.getenv("PADDLE_LOCAL_SIZE", "").strip()
    if v:
        try:
            n = int(v)
        except ValueError:
            n = 0
        if n > 0:
            return n
    return max(1, paddle.device.cuda.device_count())


def _all_ranks_same(checks, world):
    """Collective: whether every (obj, group) of checks is identical on all ranks of its group (crc32 of repr(obj),
    one all_reduce MAX per group), combined with an all_reduce MIN over world."""
    ok = True
    for obj, group in checks:
        h = zlib.crc32(repr(obj).encode()) & 0x7FFFFFFF
        v = paddle.to_tensor([h, -h], dtype="int64")
        dist.all_reduce(v, op=dist.ReduceOp.MAX, group=group)
        ok = int(v[0]) == -int(v[1]) and ok
    v = paddle.to_tensor([int(ok)], dtype="int32")
    dist.all_reduce(v, op=dist.ReduceOp.MIN, group=world)
    return int(v[0]) == 1


# ============================================================ unsupported configurations (raise at init)
_NO_HOOK_COLORS = ("dense_weight_no_hook", "moe_weight_no_hook")


def _check_comm_grads(where):
    """FusedCommBuffer._comm_grads must be Paddle's own (the one that hands buckets to the scheduler), and
    PaddleFleet's accuracy_compatible patch must not be installed: it replaces _comm_grads as a whole."""
    from paddle.distributed.fleet.utils import tensor_fusion_helper as tfh

    cur = tfh.FusedCommBuffer.__dict__.get("_comm_grads")
    wrapped = getattr(cur, "__wrapped__", cur)
    ours = (
        getattr(wrapped, "__module__", "") == tfh.__name__
        and getattr(wrapped, "__qualname__", "")
        == "FusedCommBuffer._comm_grads"
    )
    acp = sys.modules.get("paddlefleet.accuracy_compatible_patch")
    acp_on = acp is not None and bool(
        getattr(acp, "_PADDLE_RUNTIME_PATCHED", False)
    )
    if ours and not acp_on:
        return
    name = (
        getattr(wrapped, "__module__", "?")
        + "."
        + getattr(wrapped, "__qualname__", repr(cur))
    )
    raise RuntimeError(
        f"gin_overlap ({where}) does not support a replaced FusedCommBuffer._comm_grads (for example PaddleFleet's "
        f"accuracy_compatible patch, use_dsv4_accuracy): it is now {name}, accuracy_compatible patch installed: "
        f"{acp_on}. Turn the patch off or turn sharding_configs.gin_overlap off"
    )


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
    _check_comm_grads(where)


def _check_buckets(buckets, where):
    """release_grads / free_grads_in_comm per bucket, and parameters with a no-hook color (the trainer scales them
    as experts, while bucket classification puts them in dense buckets)."""
    bad = [b for b in buckets if b._release_grads or b._free_grads_in_comm]
    if bad:
        raise RuntimeError(
            f"gin_overlap ({where}) does not support release_grads / free_grads_in_comm buckets: {len(bad)}/"
            f"{len(buckets)} buckets reallocate grad_storage every step"
        )
    nh = [
        p.name
        for b in buckets
        for p in b._params
        if isinstance(getattr(p, "color", None), dict)
        and p.color.get("color") in _NO_HOOK_COLORS
    ]
    if nh:
        raise RuntimeError(
            f"gin_overlap ({where}) does not support parameters with a no-hook color: {len(nh)}, e.g. {nh[:3]}. "
            "They are scaled as experts but classified into dense buckets"
        )


def _check_dense_no_expert(dense_buckets, where):
    """Dense buckets must not hold expert parameters (is_moe_param or color "moe_expert"): expert parameters are
    reduced over the moe_sharding group."""
    bad = []
    for b in dense_buckets:
        for p in b._params:
            color = getattr(p, "color", None)
            if getattr(p, "is_moe_param", False):
                bad.append((p.name, "is_moe_param=True"))
            elif isinstance(color, dict) and color.get("color") == "moe_expert":
                bad.append((p.name, 'color="moe_expert"'))
    if bad:
        raise RuntimeError(
            f"gin_overlap ({where}): {len(bad)} expert parameters are in dense buckets (e.g. {bad[:3]}); expert "
            "parameters must be in expert buckets of the moe_sharding group"
        )


def _is_expert_bucket(b):
    def is_expert(p):
        color = getattr(p, "color", None)
        return isinstance(color, dict) and color.get("color") == "moe_expert"

    return all(is_expert(p) for p in b._params)


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


def _bad_bucket_msg(where, bad, total):
    cnt = {}
    for b, r in bad:
        cnt.setdefault(r, []).append(b._params[0].name)
    lines = [f"{len(v)}: {k} (e.g. {v[:2]})" for k, v in cnt.items()]
    return (
        f"gin_overlap ({where}): {len(bad)}/{total} buckets are not supported. With gin_overlap on every bucket "
        "must be taken over (no bucket falls back to NCCL).\n  "
        + "\n  ".join(lines)
    )


def _release_order(opt, buckets):
    """Fixed release order (bucket indices within the kind), an estimate of the order in which buckets fill up in
    the backward pass: descending by the smallest index of the bucket's parameters in opt._parameter_list, ties
    broken by the larger bucket index. Affects only performance; all ranks get the same order."""
    pos = {p.name: i for i, p in enumerate(opt._parameter_list)}
    key = [
        min((pos.get(p.name, 1 << 60) for p in b._params), default=1 << 60)
        for b in buckets
    ]
    return sorted(range(len(buckets)), key=lambda i: (-key[i], -i))


def _moe_sharding_group(hcg):
    try:
        return hcg.get_moe_sharding_parallel_group(), ""
    except AttributeError as e:
        return None, f" (hcg has no moe_sharding group: {e})"


def _split_buckets(buckets, act, world, moe_group, why_no_moe, sync_ids, where):
    """Splits buckets into (expert, dense). Raises on an unsupported bucket, on a dense bucket outside the sharding
    group and on an expert bucket outside the moe_sharding group."""
    expert, dense, bad = [], [], []
    for b in buckets:
        r = _skip_reason(b, act, sync_ids)
        if r is not None:
            bad.append((b, r))
        elif _is_expert_bucket(b):
            if moe_group is None or b._comm_group.id != moe_group.id:
                bad.append(
                    (
                        b,
                        "expert bucket outside the moe_sharding group"
                        + why_no_moe,
                    )
                )
            else:
                expert.append(b)
        elif b._comm_group.id != world.id:
            bad.append(
                (
                    b,
                    "dense bucket outside the sharding group (its parameters carry their own color['group']); dense "
                    "buckets must all be in one group",
                )
            )
        else:
            dense.append(b)
    if bad:
        raise RuntimeError(_bad_bucket_msg(where, bad, len(buckets)))
    return expert, dense


# ============================================================ layout
def _topology(group, what):
    """(group members per node, GPUs per node). The rank list must be ascending and every node must hold the same
    number of members."""
    ranks = list(group.ranks)
    ls = _local_size()
    if ranks != sorted(ranks):
        raise RuntimeError(
            f"gin_overlap: the rank list of the {what} group is not ascending ({ranks[:16]}...), so the node layout "
            "cannot be derived"
        )
    per_node = Counter(r // ls for r in ranks)
    sizes = set(per_node.values())
    if len(sizes) != 1:
        raise RuntimeError(
            f"gin_overlap: the {what} group has a different number of members on different nodes (GPUs per node "
            f"PADDLE_LOCAL_SIZE={ls}, members per node {dict(per_node)})"
        )
    rpn = sizes.pop()
    if rpn < 1 or rpn > _MAX_RANKS_PER_NODE or group.nranks % rpn:
        raise RuntimeError(
            f"gin_overlap: the {what} group of {group.nranks} ranks has {rpn} members per node; it must divide the "
            f"group size and be at most {_MAX_RANKS_PER_NODE}"
        )
    return rpn, ls


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
    """One 256 B aligned range per bucket. Returns (byte offsets, total bytes)."""
    offsets, total = [], 0
    for b in [n * 4 for n in numels]:
        offsets.append(total)
        total += _align(b)
    return offsets, total


def _recv_pool_ring(order, seg, size):
    """Ring layout of one recv pool segment: buckets take contiguous ranges in `order`; one that does not fit
    (cur + slot > size) wraps to 0. Predecessor = among the earlier holders of the range of this bucket's slot, the
    one with the largest sequence number (step * n + position); the predecessor distance d is taken in step 2
    (1..n). Returns ({bucket: offset}, {bucket: d}); buckets with a slot of 0 bytes are absent."""
    n, off, cur = len(order), {}, 0
    for b in order:
        sz = seg[b]
        if sz <= 0:
            continue
        if sz > size:
            raise RuntimeError(
                f"gin_overlap: the slot of bucket {b} is {sz} B, larger than its segment ({size} B)"
            )
        if cur + sz > size:
            cur = 0
        off[b] = cur
        cur += sz
    occ, d = [], {}
    for s in (0, 1):
        for k, b in enumerate(order):
            if b not in off:
                continue
            a, e, q = off[b], off[b] + seg[b], s * n + k
            keep, pq = [], -1
            for x, y, o in occ:
                if y <= a or x >= e:
                    keep.append((x, y, o))
                    continue
                pq = max(pq, o)
                if x < a:
                    keep.append((x, a, o))
                if y > e:
                    keep.append((e, y, o))
            keep.append((a, e, q))
            occ = sorted(keep)
            if s == 1:
                if pq < 0:
                    raise RuntimeError(
                        f"gin_overlap: no predecessor found for bucket {b} (internal error)"
                    )
                d[b] = q - pq
    return off, d


def _recv_pool_sizes(lb, descs, dense_gb, expert_gb):
    """Sizes (bytes, multiples of 4 KB) of the three segments. The dense pool is split between the intra-node and
    inter-node segments in proportion to their lower bounds."""
    P = list(lb)
    Sd = lb[0] + lb[1]
    if Sd > 0:
        Td = int(dense_gb * 2**30)
        if Td < Sd:
            raise RuntimeError(
                f"gin_overlap_dense_recv_pool_gb={dense_gb:g} is too small: the lower bounds of the intra-node and "
                f"inter-node segments add up to {_gib(Sd)} GiB ({Sd} B); {descs[0]}; {descs[1]}"
            )
        P[0] = (lb[0] * Td // Sd) // 4096 * 4096
        P[1] = (lb[1] * Td // Sd) // 4096 * 4096
        for i in (0, 1):
            if P[i] < lb[i]:
                raise RuntimeError(
                    f"gin_overlap_dense_recv_pool_gb={dense_gb:g}: after the split, {descs[i]} is not met "
                    f"(got {_gib(P[i])} GiB)"
                )
    if lb[2] > 0:
        Tm = int(expert_gb * 2**30) // 4096 * 4096
        if Tm < lb[2]:
            raise RuntimeError(
                f"gin_overlap_expert_recv_pool_gb={expert_gb:g} is too small: {descs[2]}"
            )
        P[2] = Tm
    return P


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
        nd, ne = len(self.dense_numels), len(self.expert_numels)
        if sorted(self.dense_order) != list(range(nd)) or sorted(
            self.expert_order
        ) != list(range(ne)):
            raise RuntimeError(
                f"gin_overlap: the release order is not a permutation of the bucket ids ({nd} dense / {ne} expert)"
            )
        for n in self.dense_numels:
            if n % dense.size or (n // dense.size) % 4:
                raise RuntimeError(
                    f"gin_overlap: dense bucket size {n} must be a multiple of 4 x the sharding group size "
                    f"{dense.size}"
                )
        for n in self.expert_numels:
            if n % expert.size:
                raise RuntimeError(
                    f"gin_overlap: expert bucket size {n} is not divisible by the moe_sharding group size "
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
        L, Nn = dense.ranks_per_node, dense.num_nodes
        N = max(expert.size, 1) if self.expert_numels else 1
        dm = [n // dense.size for n in self.dense_numels]
        mm = [n // N for n in self.expert_numels]
        seg = [
            # intra-node segment: the rows pushed by the other L-1 ranks
            [_align((L - 1) * Nn * m * 4) if L > 1 else 0 for m in dm],
            # inter-node segment: Nn rows, this rank's row stays empty
            [_align(Nn * m * 4) if Nn > 1 else 0 for m in dm],
            # expert segment: N rows, this rank's row stays empty
            [_align(N * m * 4) if N > 1 else 0 for m in mm],
        ]
        big = [max(range(len(s)), key=lambda i: s[i]) if s else -1 for s in seg]
        lb_loc = [_align(max(s, default=0), 4096) for s in seg]
        t = paddle.to_tensor(lb_loc[:2], dtype="int64")
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=dense.group)
        tm = paddle.to_tensor(lb_loc[2:], dtype="int64")
        if self.expert_numels:
            dist.all_reduce(tm, op=dist.ReduceOp.MAX, group=expert.group)
        lb = [int(t[0]), int(t[1]), int(tm[0])]
        names = (
            "dense intra-node segment",
            "dense inter-node segment",
            "expert segment",
        )
        descs = [
            f"{names[i]} lower bound {_gib(lb[i])} GiB (largest bucket on this rank "
            f"{'dense' if i < 2 else 'expert'}#{big[i]}, slot {_gib(lb_loc[i])} GiB)"
            for i in range(3)
        ]
        P = _recv_pool_sizes(
            lb,
            descs,
            cfg.gin_overlap_dense_recv_pool_gb,
            cfg.gin_overlap_expert_recv_pool_gb,
        )
        off_i, d_i = _recv_pool_ring(self.dense_order, seg[0], P[0])
        off_n, d_n = _recv_pool_ring(self.dense_order, seg[1], P[1])
        off_m, d_m = _recv_pool_ring(self.expert_order, seg[2], P[2])
        # Senders compute the write offsets on their peers from their own table, so it must match on all ranks.
        dense_layout = (P[0], P[1], L, Nn, self.dense_order, seg[0], seg[1], off_i, d_i, off_n, d_n)  # fmt: skip
        checks = [(dense_layout, dense.group)]
        if self.expert_numels:
            checks.append(
                ((P[2], N, self.expert_order, seg[2], off_m, d_m), expert.group)
            )
        if not _all_ranks_same(checks, dense.group):
            raise RuntimeError(
                "gin_overlap: the recv pool layout (segment sizes / release order / offsets / predecessors) differs "
                "between ranks (dense must match over the sharding group, expert over the moe_sharding group); "
                "check the recv pool sizes and the bucket split on every rank"
            )
        self.segment_bytes = P
        self.lower_bounds = lb
        self.segment_slots = seg
        # start of each segment: intra-node / inter-node in the dense recv window, expert in the expert recv window
        self.base = [0, P[0], 0]
        self.offsets = (off_i, off_n, off_m)
        self.distance = (d_i, d_n, d_m)
        self.dense_recv_bytes = P[0] + P[1]
        self.expert_recv_bytes = P[2]

    def _host_windows(self):
        """Host window layout of both groups, each from 0, every area 256 B aligned.
        dense group: signal (one counter per dense bucket at 8*idx, counting the inter-node puts) | credit_intra (16
        counters, one per node-local sender) | credit_inter (one counter per node) | intra_arrive (16 counters per
        dense bucket at intra_arrive + (idx*16 + l)*8, written by node-local sender l) | gather arrive (8 B per
        dense bucket) | gather intra (16 counters per dense bucket).
        expert group: signal (one counter per expert bucket at 8*idx) | credit_expert (one counter per group rank) |
        gather arrive | gather intra, sized by the expert buckets."""
        nd, ne = len(self.dense_numels), len(self.expert_numels)
        c0 = _align(8 * nd, 256)
        c1 = _align(c0 + 8 * 16, 256)
        dense_signal_end = c1 + 8 * max(self.dense.num_nodes, 1)
        ia = _align(dense_signal_end, 256)
        ga = _align(ia + 8 * 16 * nd, 256)
        gi = _align(ga + 8 * max(nd, 1), 256)
        self.intra_arrive = ia
        self.gather_dense = (ga, gi)
        self.dense_host_bytes = _align(gi + 8 * 16 * max(nd, 1), 4096)
        expert_size = self.expert.size if ne else 0
        c2 = _align(8 * ne, 256)
        moe_signal_end = c2 + 8 * max(expert_size, 1)
        mga = _align(moe_signal_end, 256)
        mgi = _align(mga + 8 * max(ne, 1), 256)
        self.gather_expert = (mga, mgi)
        self.expert_host_bytes = _align(mgi + 8 * 16 * max(ne, 1), 4096)
        self.credit = (c0, c1, c2)

    def bucket_table(self):
        """One (kind, send_offset, signal_offset, shard_numel, arrive_offset, recv_offset_0, recv_offset_1,
        depth_0, depth_1) per bucket, dense buckets first. depth -1 = no slot in that segment."""
        rows = []
        for i, n in enumerate(self.dense_numels):
            di = self.distance[0].get(i)
            dn = self.distance[1].get(i)
            rows.append(
                (
                    _LIB.DENSE_BUCKET,
                    self.dense_send[i],
                    8 * i,
                    n // self.dense.size,
                    self.intra_arrive + i * 16 * 8,
                    self.base[0] + self.offsets[0].get(i, 0),
                    self.base[1] + self.offsets[1].get(i, 0),
                    -1 if di is None else di,
                    -1 if dn is None else dn,
                )
            )
        for i, n in enumerate(self.expert_numels):
            dm = self.distance[2].get(i)
            rows.append(
                (
                    _LIB.EXPERT_BUCKET,
                    self.expert_send[i],
                    8 * i,
                    n // self.expert.size,
                    0,
                    self.base[2] + self.offsets[2].get(i, 0),
                    0,
                    -1 if dm is None else dm,
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
        segs = []
        for i in range(3):
            s = [x for x in self.segment_slots[i] if x > 0]
            med = statistics.median(s) if s else 0
            fit = int(self.segment_bytes[i] // med) if med else 0
            segs.append(
                f"{names[i]} {_gib(self.segment_bytes[i])} GiB (lower bound {_gib(self.lower_bounds[i])}, median "
                f"slot {format(med / 2**20, '.1f')} MiB, holds {fit})"
            )
        n = (
            len(self.dense_order),
            len(self.dense_order),
            len(self.expert_order),
        )
        inner = [
            sum(1 for v in self.distance[i].values() if v < n[i])
            for i in range(3)
        ]
        _log(
            f"init: {len(self.dense_numels)} dense buckets (sharding group {self.dense.num_nodes} nodes x "
            f"{self.dense.ranks_per_node}), {len(self.expert_numels)} expert buckets; recv pool: " + "; ".join(segs)
            + f"; predecessors in the same step: intra-node {inner[0]} / inter-node {inner[1]} / expert {inner[2]}; "
            f"windows {windows}"
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
        info = pg.get_or_create_dev_comm(GIN_CONTEXTS)
        ok = int(
            info["gin_context_count"] >= 1
            and info["lsa_size"] == layout.ranks_per_node
            and info["lsa_rank"] == layout.rank_in_node
        )
        t = paddle.to_tensor(
            [ok, -layout.ranks_per_node, layout.ranks_per_node], dtype="int32"
        )
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=layout.group)
        if int(t[0]) != 1 or int(-t[1]) != int(t[2]):
            raise RuntimeError(
                f"gin_overlap: the device communicator of the {layout.what} group is unusable, or its LSA team "
                f"differs from the node layout (GIN contexts {info['gin_context_count']}, lsa_rank "
                f"{info['lsa_rank']}, lsa_size {info['lsa_size']}; from the rank list rank_in_node "
                f"{layout.rank_in_node}, ranks_per_node {layout.ranks_per_node}; GPUs per node {layout.local_size})"
            )
        self.gin_contexts = info["gin_context_count"]

    def handles(self):
        return (self.send_window, self.recv_window, self.host_window)


# ============================================================ grad_storage -> send window
def _alias_1d(b, window, offset):
    """Replaces grad_storage of a 1D bucket with its range of the send window and re-links each parameter's
    main_grad / _slice_grad. The old grad_storage is copied over, then released."""
    n = b.grad_storage._numel()
    begin = offset // 4
    new = window.send._slice(begin, begin + n)
    new.copy_(b.grad_storage, False)
    b.grad_storage = new
    for p in b._params:
        v = b._sharding_param_grad_view[p.name]
        v._grad_buffer = new
        v._link_grad_to_buffer()
    b._record_addr()
    return new.data_ptr()


def _alias_2d(b, window, offset, npad):
    """Moves a Muon 2D bucket into the send window: the range holds npad fp32, the first n copied from the old
    grad_storage and [n, npad) zero; each parameter's main_grad is re-sliced at its element offset; grad_storage
    becomes the whole npad range."""
    old = b.grad_storage
    n = old._numel()
    if n > npad:
        raise RuntimeError(
            f"gin_overlap: 2D bucket of {b._params[0].name}: {n} elements, more than its padded size {npad}"
        )
    begin = offset // 4
    new = window.send._slice(begin, begin + npad)
    src = (
        old
        if npad == n
        else paddle.concat(
            [old, paddle.zeros([npad - n], dtype=paddle.float32)]
        )
    )
    new.copy_(src, False)
    del src
    for p in b._params:
        g = p.main_grad
        if g is None or g._numel() != p._numel():
            raise RuntimeError(
                f"gin_overlap: 2D bucket: main_grad of {p.name} is missing or does not have {p._numel()} elements"
            )
        eoff = (g.data_ptr() - old.data_ptr()) // 4
        if eoff < 0 or eoff + p._numel() > n:
            raise RuntimeError(
                f"gin_overlap: 2D bucket: main_grad of {p.name} starts at element {eoff}, outside the bucket's "
                f"{n} elements"
            )
        t = new._slice(eoff, eoff + p._numel())
        t.get_tensor()._set_dims(p.shape)
        p.main_grad = t
    b.grad_storage = new
    b._record_addr()
    return new.data_ptr()


# ============================================================ Muon 2D gather table
def _gather_table(layout, dense_2d, expert_2d, dense_index, expert_index):
    """One (unified bucket index, owner group rank, arrive_offset, intra_arrive_offset) per Muon 2D bucket.
    arrive_offset = inter-node arrival counter of the bucket on the owner, intra_arrive_offset = start of its 16
    same-node arrival counters. Collective: the table must match over each group (crc)."""
    nd = len(layout.dense_numels)
    rows, sig = [], {"dense": [], "expert": []}
    for kind, buckets, index, gl, (ga, gi) in (
        ("dense", dense_2d, dense_index, layout.dense, layout.gather_dense),
        (
            "expert",
            expert_2d,
            expert_index,
            layout.expert,
            layout.gather_expert,
        ),
    ):
        for b in buckets:
            idx = index[id(b)]
            _, npad, owner = b._gin_owner_bucket
            want = (layout.dense_numels if kind == "dense" else layout.expert_numels)[idx]  # fmt: skip
            if npad != want:
                raise RuntimeError(
                    f"gin_overlap: 2D {kind} bucket {idx} has padded size {npad}, the layout has {want}"
                )
            send = (layout.dense_send if kind == "dense" else layout.expert_send)[idx]  # fmt: skip
            unified = idx if kind == "dense" else nd + idx
            rows.append((unified, owner, ga + 8 * idx, gi + idx * 16 * 8))
            sig[kind].append((idx, send, npad // gl.size, owner))
    checks = [
        (
            (layout.gather_dense, nd, layout.dense.ranks_per_node, sorted(sig["dense"])),
            layout.dense.group,
        )
    ]  # fmt: skip
    if layout.expert_numels:
        checks.append(
            (
                (layout.gather_expert, len(layout.expert_numels), layout.expert.ranks_per_node,
                 sorted(sig["expert"])),
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
def _pad(n, unit):
    return -(-n // unit) * unit


def _select_v2(opt, world, moe_group, why_no_moe):
    where = "DygraphShardingOptimizerV2"
    _check_buckets(opt._comm_buffer_list, where)
    expert, dense = _split_buckets(
        opt._comm_buffer_list,
        _hook_action().REDUCE_SCATTER,
        world,
        moe_group,
        why_no_moe,
        (),
        where,
    )
    return expert, dense, [], []


def _select_muon(opt, world, moe_group, why_no_moe):
    where = "MuonShardingOptimizer"
    if not opt._use_fuse_gradients:
        raise RuntimeError(
            f"gin_overlap ({where}) needs comm_buffer_size_MB > 0: 2D parameters must be in fused buckets"
        )
    if opt.use_group_call_opt:
        raise RuntimeError(
            f"gin_overlap ({where}) cannot be combined with sharding_configs.comm_group_call_opt"
        )
    _check_buckets(
        list(opt._comm_buffer_list) + list(opt.comm_buffer_2d), where
    )
    act = _hook_action()
    sync_ids = opt._sync_only_buffer_ids
    expert_1d, dense_1d = _split_buckets(
        opt._comm_buffer_list, act.REDUCE_SCATTER, world, moe_group, why_no_moe, sync_ids, f"{where} / 1D buckets"
    )  # fmt: skip
    expert_2d, dense_2d = _split_buckets(
        opt.comm_buffer_2d, act.REDUCE, world, moe_group, why_no_moe, sync_ids, f"{where} / 2D buckets"
    )  # fmt: skip
    nw = world.nranks
    ne = moe_group.nranks if (expert_1d or expert_2d) else 1
    for b in dense_2d:
        n = b.grad_storage._numel()
        # (numel, numel padded to a multiple of 4 x group size, owner's group rank)
        b._gin_owner_bucket = (n, _pad(n, 4 * nw), world.ranks.index(b._dst))
    for b in expert_2d:
        n = b.grad_storage._numel()
        b._gin_owner_bucket = (
            n,
            _pad(n, 4 * ne),
            moe_group.ranks.index(b._dst),
        )
    return expert_1d + expert_2d, dense_1d + dense_2d, expert_2d, dense_2d


def _hook_action():
    from paddle.distributed.fleet.utils.tensor_fusion_helper import HOOK_ACTION

    return HOOK_ACTION


def _numel(b):
    v = getattr(b, "_gin_owner_bucket", None)
    return v[1] if v is not None else b.grad_storage._numel()


def build_grad_comm_scheduler(opt):
    """Builds the GinGradCommScheduler of a DygraphShardingOptimizerV2 or MuonShardingOptimizer, after its buckets
    are built, and registers it (active_scheduler()). Collective over the sharding group and, with expert buckets,
    the moe_sharding group. Raises on an unsupported configuration or bucket.
      - Every bucket is taken over: expert buckets (all parameters color "moe_expert") in the moe_sharding group,
        all other buckets in the sharding group. Muon 2D buckets are padded to a multiple of 4 x group size and
        gathered to their owner after the reduce.
      - grad_storage of each bucket becomes its range of the send window; main_grad is re-linked.
      - Each bucket gets ``_gin_bucket_index`` (unified index) and ``_gin_scheduler``; each parameter gets a backward
        hook that reports its bucket once the bucket is full, i.e. after accumulate_steps gradients per parameter
        (pipeline_configs.accumulate_steps with pp > 1, sharding_configs.accumulate_steps otherwise; it must equal
        the gradient accumulation steps of the trainer)."""
    global _ACTIVE
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
        DygraphShardingOptimizerV2,
    )

    if _LIB is None:
        raise RuntimeError(
            f"sharding_configs.gin_overlap is on, but {_BUILD_HINT}"
        )
    if _ACTIVE is not None:
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
    moe_group, why_no_moe = _moe_sharding_group(hcg)
    select = _select_v2 if is_v2 else _select_muon
    expert, dense, expert_2d, dense_2d = select(
        opt, world, moe_group, why_no_moe
    )
    _check_dense_no_expert(dense, where)
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
        [_numel(b) for b in dense],
        [_numel(b) for b in expert],
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
    nd = len(dense)
    for kind, buckets, windows, offsets in (
        ("dense", dense, dense_windows, layout.dense_send),
        ("expert", expert, expert_windows, layout.expert_send),
    ):
        for i, b in enumerate(buckets):
            if getattr(b, "_gin_owner_bucket", None) is not None:
                ptr = _alias_2d(b, windows, offsets[i], b._gin_owner_bucket[1])
            else:
                ptr = _alias_1d(b, windows, offsets[i])
            b._gin_bucket_index = i if kind == "dense" else nd + i
            b._gin_grad_ptr = ptr
    paddle.device.synchronize()

    c0, c1, c2 = layout.credit
    scheduler = GinGradCommScheduler(
        dense_group=world.process_group,
        expert_group=moe_group.process_group if expert else None,
        dense_windows=dense_windows.handles(),
        expert_windows=expert_windows.handles() if expert else (0, 0, 0),
        buckets=layout.bucket_table(),
        dense_order=layout.dense_order,
        expert_order=[nd + i for i in layout.expert_order],
        credit_intra_offset=c0,
        credit_inter_offset=c1,
        credit_expert_offset=c2,
        gather=gather,
        num_window_marks=NUM_WINDOW_MARKS,
        num_tail_marks=nd + len(expert) + TAIL_EXTRA_MARKS,
    )
    # keep the windows alive as long as the scheduler
    scheduler_state = (layout, dense_windows, expert_windows)

    hook_kw = {"use_comm": True} if is_v2 else {}
    acc = _accumulate_steps()
    for b in dense + expert:
        b._gin_scheduler = scheduler
        b._gin_reported = False
        # full only after the last micro-step's gradients
        b._acc_steps = acc
        b._init_step_dict()
        for param in b._params:
            param._register_backward_hook(
                opt._create_backward_hook(b, param, **hook_kw)
            )
    opt._gin_scheduler = scheduler
    opt._gin_scheduler_state = scheduler_state
    opt._gin_buckets = dense + expert
    dist.barrier(group=world)
    windows = "; ".join(
        f"{w.layout.what} group send {_gib(w.send._numel() * 4)} + recv {_gib(w.recv._numel() * 4)} GiB, host "
        f"{w.host_bytes} B, GIN contexts {w.gin_contexts}"
        for w in (dense_windows, expert_windows)
        if w is not None
    )
    layout.log(windows)
    if gather:
        _log(
            f"init: {len(gather)} Muon 2D buckets gathered to their owners (dense {len(dense_2d)} / expert "
            f"{len(expert_2d)})"
        )
    _ACTIVE = scheduler
    return scheduler


# ============================================================ calls from FusedCommBuffer and the optimizers
def bucket_ready(bucket, reduce_op):
    """Reports a full bucket (FusedCommBuffer._comm_grads). The scale is the factor of reduce_op: 1/N for AVG, 1 for
    SUM. A second call within the step does nothing."""
    if bucket._gin_reported:
        return
    g = bucket.grad_storage
    if g is None or g.data_ptr() != bucket._gin_grad_ptr:
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
    global _COMM_GRADS_CHECKED
    if not _COMM_GRADS_CHECKED:
        _check_comm_grads("first reduce_gradients")
        _COMM_GRADS_CHECKED = True
    opt._gin_scheduler.finish_step(
        float(_sharding_configs().gin_overlap_timeout)
    )


def fused_all_gather(buckets):
    """Parameter all_gather of the REDUCE_SCATTER buckets, fused per (communication group, dtype): this rank's
    shards of a batch of buckets are concatenated, sent with one process_group.all_gather and split back into each
    bucket's param_storage. A batch holds at most sharding_configs.param_allgather_fuse_mb MB (at least one bucket).
    param_storage of a bucket is [rank 0 part, rank 1 part, ...], so bucket x with n elements per part takes
    columns [off, off + n) of each row j of recv[N, S] and goes to param_storage[j * n, (j + 1) * n)."""
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
    for gbuckets in groups.values():
        group = gbuckets[0]._comm_group
        nranks, rank = group.nranks, max(group.rank, 0)
        chunk, nbytes = [], 0
        for b in [*gbuckets, None]:
            if b is not None:
                sz = b.param_storage._numel() * b.param_storage.element_size()
                if not chunk or nbytes + sz <= limit:
                    chunk.append(b)
                    nbytes += sz
                    continue
            _all_gather_chunk(chunk, group, nranks, rank)
            if b is None:
                break
            chunk, nbytes = [b], sz


def _all_gather_chunk(chunk, group, nranks, rank):
    segs = [x.param_storage._numel() // nranks for x in chunk]
    for x, n in zip(chunk, segs):
        if x.param_storage._numel() != nranks * n:
            raise RuntimeError(
                f"gin_overlap: param_storage of the bucket of {x._params[0].name} has {x.param_storage._numel()} "
                f"elements, not a multiple of the group size {nranks}"
            )
    send = paddle.concat(
        [x.param_storage._slice(n * rank, n * (rank + 1)) for x, n in zip(chunk, segs)]
    )  # fmt: skip
    total = send._numel()
    recv = paddle.empty([nranks * total], dtype=send.dtype)
    group.process_group.all_gather(send, recv).wait()
    off = 0
    for x, n in zip(chunk, segs):
        if n:
            for j in range(nranks):
                x.param_storage._slice(j * n, (j + 1) * n).copy_(
                    recv._slice(j * total + off, j * total + off + n), False
                )
        off += n
    del send, recv
