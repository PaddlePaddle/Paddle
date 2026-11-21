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

"""Communication primitive of gin_overlap: the reduce_scatter of each sharding-stage1 gradient bucket, done in stages
with NCCL GIN one-sided puts.

This file covers the layout of the windows, the recv pool and the host windows, setting up and checking the two
communication groups, the per-bucket parameter tables handed to the C++ release layer, the 2D-bucket gather table,
and the init logs. It is the complete description of the Python side; other files refer to it.
Main interfaces: device_engine() (the C++ engine object), GinOverlapPrimitive (callers subclass it), check_cuda, log.
When each stage is released is not decided here; the caller decides.

Terms
  - bucket: a contiguous block of device memory holding the gradients of several parameters; communication works on
    whole buckets.
  - shard: the part of a bucket this rank owns after the bucket is split across the communication group.
  - GIN: NCCL GPU-Initiated Networking. Only its one-sided put is used: it writes a range of local data into a peer's
    window without involving the peer GPU.
  - LSA: in the NCCL device API, the ranks of one node that can reach each other's memory through plain pointers;
    lsaRank / lsaSize = this rank's index in the node / the number of ranks in it.
  - devcomm: ncclDevComm, the communicator handle that kernels use for GIN and LSA calls.
  - CE: copy engine. The intra-node copy runs on it and takes no SMs.
  - window: symmetric memory registered with NCCL that peers can read and write one-sidedly. Each communication group
    has send / recv / signal device windows and one host window (CPU memory that holds the counters). A bucket's
    grad_storage is its range of the group's send window.
  - overlap window: a span of the backward pass in which DeepEP leaves the NIC or the SMs idle (window A / window B /
    pre window).
  - mark: a cudaEvent recorded on the compute stream; work submitted to the GPU waits on it first.
  - release: deciding that one stage of one bucket may run now, and submitting it to the GPU (the C++ release layer,
    release.cu).
  - credit: after a receiver is done with a slot of its recv pool, it adds 1 to a counter of the sender ("returns a
    credit"); the sender writes to that slot again only once the counter reaches its target.
  - owner: the rank a Muon 2D bucket finally belongs to.
  - group rank: a rank's index inside a communication group, not its world rank.
  - tail thread: the background thread of the C++ release layer. It releases everything that is left after the last
    overlap window of a step.
  - shard reduce: summing the partial sums of all nodes (dense bucket) or the parts of all ranks of the expert group
    (expert bucket) into the final result of this rank's shard. In names it is inter_node_reduce (dense) /
    moe_reduce (expert); shard_reduce covers both.

Communication groups
  - Dense buckets use the sharding group (GROUP_DENSE), expert buckets the moe_sharding group (GROUP_MOE); each has
    its own comm / devcomm / windows. The comms are taken from Paddle's existing ProcessGroups.
  - In both groups the rank list must be ascending and every node must hold the same number of members:
    rank_in_node = group rank % ranks_per_node, node_index = group rank // ranks_per_node.

Stages
  - dense bucket: intra-node copy (CE) -> intra-node reduce -> inter-node put to the same rank_in_node -> shard reduce.
  - expert bucket: expert put to every other rank of the expert group -> shard reduce.
  - Muon 2D bucket: after the shard reduce, this rank's shard is gathered to the owner (GIN put across nodes, CE
    within a node).

Recv pool (the recv window, _layout_recv_pool)
  - Three segments: the dense intra-node segment and the dense inter-node segment (both in the dense group's recv
    window), and the expert segment (in the expert group's recv window).
  - Each segment is a ring: buckets take one slot each in the fixed release order and wrap to the start of the
    segment when the rest does not fit, so several buckets can be in flight at once.
  - predecessor distance d: how much smaller the release sequence number of the previous holder of this bucket's slot
    is than this bucket's (1..n; n = the previous holder is this bucket's own copy from the step before). The release
    layer computes credit targets from it.
  - The layout must be identical on all ranks (the two dense segments across the whole world, the expert segment
    within the group); init checks this with a crc.

Host window layout (_layout_host_windows, one u64 per counter)
  - dense group:
    - signal: 16 B per dense bucket; the first u64 is the GIN signal of the inter-node put.
    - credit[0] intra-node: 16 counters; credits from node-local rank l go to counter l.
    - credit[1] inter-node: num_nodes counters; credits from the rank with the same rank_in_node on node n go to
      counter n.
    - intra_arrive, the intra-node arrival counters: 16 per dense bucket at intra_arrive + (bucket id*16 + l)*8;
      sender l of the same node writes how many intra-node copies of this bucket it has submitted.
    - gather_dense: the three (arrive, intra, credit) areas of the dense 2D-bucket gather, see _layout_host_windows.
  - expert group: 16 B of signal per expert bucket; credit[2] expert, one counter per expert group rank; the three
    gather_moe areas.

Constraints
  - Counters accumulate across steps and only grow. The CPU never writes or clears them (_hw is read-only).
  - Each sender writes only its own intra-node arrival counter; a counter has exactly one writer.
  - Every stage is submitted in the fixed release order, which must be the same on all ranks.
  - Every bucket goes through all of its stages every step; otherwise later credits never arrive.
"""

import ctypes
import os
import time
from collections import Counter

import paddle
import paddle.distributed as dist

# The C++ implementation (paddle.base.core.gin_overlap, sources in paddle/fluid/distributed/collective/gin_overlap/)
# exists only when the build conditions hold; otherwise paddle.base.core has no gin_overlap attribute. The
# conditions are spelled out in the error message below.
_BUILD_HINT = (
    "this paddlepaddle was built without gin_overlap: it needs WITH_GPU + WITH_NCCL, NCCL >= 2.30.7 "
    "(with nccl_device.h) and a CUDA arch of 9.0 or 10.0"
)

# Fixed parameters (none of them affects numerical results)
PUT_BLOCKS = 64  # block count of the put kernels, passed to the release layer by release_setup
# GIN contexts requested (the devcomm reports the actual number)
GIN_CONTEXTS = 8
TIMEOUT_S = 300.0  # timeout (s) when waiting for peers; raises when exceeded
NUM_WINDOW_MARKS = 1024  # size of the window mark ring
# Size of the tail mark ring = number of buckets + TAIL_EXTRA_MARKS. Each step uses 1 (hand-off) + buckets ready
#   after the hand-off + combines passed after the hand-off slots; running out within a step raises, there is no
#   wrap-around
TAIL_EXTRA_MARKS = 1024
LOG_EVERY = 100  # steps between two summary log lines of accumulated traffic / scheduling
# Communication group ids; must match GROUP_DENSE / GROUP_MOE of the C++ side (checked in device_engine())
GROUP_DENSE, GROUP_MOE = 0, 1
GROUP_NAME = {GROUP_DENSE: "sharding", GROUP_MOE: "moe_sharding"}

_DEVICE_ENGINE = None


def device_engine():
    """Returns the process-wide paddle.base.core.gin_overlap.GradCommEngine object, created on the first call.
    Raises when paddle.base.core.gin_overlap is missing, when the group ids differ from the C++ side, or when the
    NCCL version used at build time differs from the one loaded at run time."""
    global _DEVICE_ENGINE
    if _DEVICE_ENGINE is None:
        from paddle.base import core

        lib = getattr(core, "gin_overlap", None)
        if lib is None:
            raise RuntimeError(
                f"no paddle.base.core.gin_overlap: {_BUILD_HINT}"
            )
        if (lib.GROUP_DENSE, lib.GROUP_MOE) != (GROUP_DENSE, GROUP_MOE):
            raise RuntimeError(
                f"gin_overlap: group ids of the C++ side ({lib.GROUP_DENSE}, {lib.GROUP_MOE}) "
                f"differ from the Python side ({GROUP_DENSE}, {GROUP_MOE}); both sides must change together"
            )
        eng = lib.GradCommEngine()
        rt, bt = eng.nccl_runtime_version(), eng.nccl_build_version()
        if rt != bt:
            raise RuntimeError(
                f"gin_overlap: the C++ side was built against NCCL {bt}, but the process loaded NCCL {rt}; "
                "the device API's struct layouts are compiled into the cubins, so both must be the same libnccl"
            )
        _DEVICE_ENGINE = eng
    return _DEVICE_ENGINE


def check_cuda(rc, what):
    if rc != 0:
        raise RuntimeError(
            f"gin_overlap: {what} failed: {device_engine().error_string(rc)}"
        )


def log(msg):
    if dist.get_rank() == 0:
        print(f"[gin_overlap] {msg}", flush=True)


def _align(n, a=256):
    return (n + a - 1) // a * a


def _gib(x):
    return format(x / 2**30, ".3f")


def local_size():
    """GPUs per node: PADDLE_LOCAL_SIZE; the number of visible devices when it is unset or not a positive integer."""
    v = os.getenv("PADDLE_LOCAL_SIZE", "").strip()
    if v:
        try:
            n = int(v)
        except ValueError:
            n = 0
        if n > 0:
            return n
    return max(1, paddle.device.cuda.device_count())


class GinOverlapPrimitive:
    def __init__(
        self,
        world,
        moe_group,
        moe_numels,
        dense_numels,
        release_order,
        dense_recv_pool_gb,
        moe_recv_pool_gb,
    ):
        """Creates the windows, devcomms and recv pool layout of both groups and hands them to the C++ side
        (configure).
        Collective: every rank of world (the sharding group) must call it at the same time; when there are expert
        buckets, every rank of the expert group (moe_sharding) must call it too.
          - world / moe_group: the sharding group / expert group objects (moe_group may be None without expert
            buckets).
          - moe_numels / dense_numels: element count of each bucket; must be divisible by the size of its group.
          - release_order: (fixed release order of the dense buckets, fixed release order of the expert buckets),
            both permutations of the bucket ids and identical on all ranks.
          - dense_recv_pool_gb / moe_recv_pool_gb: size (GiB) of the dense recv pool (intra-node segment +
            inter-node segment) and of the expert segment.
        The layout is described in the module docstring; the caller must call release_setup afterwards."""
        device_engine()  # create the engine object and check versions as early as possible
        self.dense_recv_pool_gb = dense_recv_pool_gb
        self.moe_recv_pool_gb = moe_recv_pool_gb
        self.world = world
        self.moe_comm_group = moe_group
        self.group_size, self.group_rank = world.nranks, world.rank
        t_init = [time.perf_counter()]

        self.moe_bucket_numels = list(moe_numels)
        self.expert_group_size = self.expert_group_rank = 0
        if self.moe_bucket_numels:
            assert moe_group is not None, (
                "there are expert buckets but no expert communication group"
            )
            self.expert_group_size, self.expert_group_rank = (
                moe_group.nranks,
                moe_group.rank,
            )
            for n in self.moe_bucket_numels:
                assert n % self.expert_group_size == 0, (
                    f"expert bucket size {n} is not divisible by the group size {self.expert_group_size}"
                )
        self.dense_bucket_numels = list(dense_numels)
        for n in self.dense_bucket_numels:
            assert n % self.group_size == 0, (
                f"dense bucket size {n} is not divisible by the group size {self.group_size}"
            )

        # Node layout of the group (requirements in the module docstring); checked against the devcomm's lsaRank /
        # lsaSize later in _check_devcomm
        self.ranks_per_node, self._local_size = self._topology(
            world, "sharding"
        )
        self.num_nodes = self.group_size // self.ranks_per_node
        self.rank_in_node = self.group_rank % self.ranks_per_node
        self.node_index = self.group_rank // self.ranks_per_node
        self.moe_ranks_per_node, self.moe_num_nodes = 1, 1
        self.moe_rank_in_node = self.moe_node_index = 0
        if self.moe_bucket_numels:
            self.moe_ranks_per_node, _ = self._topology(
                moe_group, "moe_sharding"
            )
            self.moe_num_nodes = (
                self.expert_group_size // self.moe_ranks_per_node
            )
            self.moe_rank_in_node = (
                self.expert_group_rank % self.moe_ranks_per_node
            )
            self.moe_node_index = (
                self.expert_group_rank // self.moe_ranks_per_node
            )

        # Each group has its own send window; buckets are laid out by byte offset (each group starts at 0)
        self.moe_send_offsets, moe_send_bytes = self._send_layout(
            self.moe_bucket_numels
        )
        self.dense_send_offsets, dense_send_bytes = self._send_layout(
            self.dense_bucket_numels
        )
        num_moe, num_dense = (
            len(self.moe_bucket_numels),
            len(self.dense_bucket_numels),
        )
        # 16 B per bucket: [0] inter-node / expert GIN signal, [1] unused
        self.moe_signal_offsets = [16 * i for i in range(num_moe)]
        self.dense_signal_offsets = [16 * i for i in range(num_dense)]
        recv_bytes = self._layout_recv_pool(
            release_order, moe_group
        )  # (dense recv bytes, expert recv bytes)

        dev = paddle.distributed.ParallelEnv().device_id
        self._set_comms(dev, world, moe_group)
        t_init.append(time.perf_counter())
        signal_bytes = self._layout_host_windows()  # signal / host window layout of both groups (returns the signal window bytes of each group)
        p = self.recv_pool
        # Every rank registers windows of its own size; peers must not access beyond them
        self._win = {}
        self._hw = {}
        self._setup_group(
            GROUP_DENSE, "sharding", dense_send_bytes, recv_bytes[0], signal_bytes[0], p["host_bytes"][0],
            self.group_size, self.group_rank, self.ranks_per_node, self.rank_in_node, self.num_nodes, self.node_index,
            world,
        )  # fmt: skip
        if num_moe:
            self._setup_group(
                GROUP_MOE, "moe_sharding", moe_send_bytes, recv_bytes[1], signal_bytes[1], p["host_bytes"][1],
                self.expert_group_size, self.expert_group_rank, self.moe_ranks_per_node, self.moe_rank_in_node,
                self.moe_num_nodes, self.moe_node_index, moe_group,
            )  # fmt: skip
        t_init.append(time.perf_counter())
        for n in self.dense_bucket_numels:
            assert (n // self.group_size) % 4 == 0, (
                f"the per-block numel of a dense bucket must be a multiple of 4: {n // self.group_size}"
            )
        self._configure(num_dense, num_moe)
        t_init.append(time.perf_counter())
        self.moe_offsets = list(
            zip(self.moe_send_offsets, self.moe_signal_offsets)
        )
        self.dense_offsets = list(
            zip(self.dense_send_offsets, self.dense_signal_offsets)
        )
        self._init_times = t_init

    @staticmethod
    def _send_layout(numels):
        """The group's send window: one range per bucket, 256 B aligned. Returns (list of byte offsets by bucket id,
        total bytes)."""
        offsets, total = [], 0
        for b in [n * 4 for n in numels]:
            offsets.append(total)
            total += _align(b)
        return offsets, total

    def _set_comms(self, dev, world, moe_group):
        """Has the C++ side take the existing Paddle ncclComm of both groups by ring id (the `.id` of the group
        object). The C++ side checks the group size and group rank and raises when a comm is missing or does not
        match. A group that has no NCCL comm yet gets one here."""
        rc, err = device_engine().set_comms(
            int(world.id),
            int(moe_group.id)
            if (moe_group is not None and self.moe_bucket_numels)
            else -1,
            dev,
            self.group_size,
            self.group_rank,
            self.expert_group_size,
            self.expert_group_rank,
        )
        if rc:
            raise RuntimeError(
                f"gin_overlap: failed to get Paddle's existing ncclComm: {err or device_engine().error_string(rc)}"
            )

    def _setup_group(self, gid, what, send_bytes, recv_bytes, signal_bytes, host_bytes, size, rank, rpn, ml, nn, mn,
                     group):  # fmt: skip
        """Creates all windows of one group (collective, every rank of the group must call it): alloc_windows ->
        alloc_host_counters (must precede create_dev_comm) -> create_dev_comm -> _check_devcomm ->
        set_group_topology."""
        lib = device_engine()
        self._win[gid] = [
            max(send_bytes, 4096),
            max(recv_bytes, 4096),
            _align(signal_bytes, 4096),
        ]
        check_cuda(
            lib.alloc_windows(gid, *self._win[gid]),
            f"ncclMemAlloc/window({what})",
        )
        rc, ptr, nbytes = lib.alloc_host_counters(gid, host_bytes)
        check_cuda(rc, f"alloc_host_counters({what})")
        if (
            nbytes < host_bytes
        ):  # the C++ side rounds up to the granularity, so this should not happen
            raise RuntimeError(
                f"gin_overlap: host window of the {what} group is {nbytes} B, less than the {host_bytes} B needed"
            )
        self.recv_pool["host_window_bytes"][gid] = nbytes
        # u64 view of the host window (a ctypes array; every index access reads memory). Read only, never write
        self._hw[gid] = (ctypes.c_ulonglong * (nbytes // 8)).from_address(ptr)
        rc, nctx, lsa_rank, lsa_size = lib.create_dev_comm(gid, GIN_CONTEXTS)
        check_cuda(rc, f"ncclDevCommCreate({what})")
        self.recv_pool["nctx"][gid] = nctx
        self._check_devcomm(
            what, group, size, rank, rpn, ml, nn, mn, nctx, lsa_rank, lsa_size
        )
        check_cuda(
            lib.set_group_topology(gid, nn, mn), f"set_group_topology({what})"
        )

    # ------------------------------------------------------------------ topology
    def _topology(self, group, what):
        """Returns (group members per node ranks_per_node, GPUs per node local_size). The group's rank list must be
        ascending and every node must hold the same number of members (it may be less than the GPUs per node);
        otherwise raises."""
        ranks = list(group.ranks)
        size = group.nranks
        ls = local_size()
        if ranks != sorted(ranks):
            raise RuntimeError(
                f"gin_overlap: rank list of the {what} group is not ascending ({ranks[:16]}...), so the node layout cannot be derived"
            )
        per_node = Counter(r // ls for r in ranks)
        sizes = set(per_node.values())
        if len(sizes) != 1:
            raise RuntimeError(
                f"gin_overlap: the {what} group has a different number of members on different nodes (GPUs per node "
                f"PADDLE_LOCAL_SIZE={ls}, members per node {dict(per_node)}), so the modulo / division layout does not hold"
            )
        rpn = sizes.pop()
        if rpn < 1 or size % rpn:
            raise RuntimeError(
                f"gin_overlap: {what} group size {size} is not a multiple of the members per node {rpn}"
            )
        return rpn, ls

    def _check_devcomm(
        self, what, group, size, rank, rpn, ml, nn, mn, nctx, lsa_rank, lsa_size
    ):
        """Checks that the devcomm's lsaRank / lsaSize match what _topology derived (collective; if any rank of the
        group fails, the whole group raises)."""
        ok = int(
            nctx >= 1
            and lsa_size == rpn
            and lsa_rank == ml
            and rpn <= 16
            and nn * rpn == size
        )
        t = paddle.to_tensor([ok, -rpn, rpn], dtype="int32")
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
        if int(t[0]) != 1 or int(-t[1]) != int(t[2]):
            raise RuntimeError(
                f"gin_overlap: the GIN devcomm of the {what} group is unusable, or the group's node layout differs from "
                f"the devcomm's LSA team (ctx={nctx} lsaRank={lsa_rank} lsaSize={lsa_size}; from the rank list "
                f"rank_in_node={ml} ranks_per_node={rpn} node_index={mn} num_nodes={nn}; GPUs per node "
                f"PADDLE_LOCAL_SIZE={self._local_size}; group rank {rank}/{size})"
            )

    # ------------------------------------------------------------------ recv pool (ring layout, several buckets in flight)
    def _layout_recv_pool(self, order, moe_group):
        """Layout of the three recv pool segments (collective): computes each segment's size, each bucket's offset in
        the segment and its predecessor distance, and checks that all ranks agree.
          - Lower bound of a segment = slot size of its largest bucket (256 B aligned, then rounded up to 4 KB); the
            two dense segments take the maximum over the whole world, the expert segment over the expert group.
          - dense_recv_pool_gb is split between the intra-node and inter-node segments in proportion to their lower
            bounds (each rounded down to 4 KB, must be >= its lower bound); moe_recv_pool_gb is the expert segment
            (rounded down to 4 KB).
          - The intra-node and inter-node segments follow each other in the dense group's recv window, the expert
            segment is in the expert group's recv window; all start at 0.
        The result goes into self.recv_pool; returns (dense recv window bytes, expert recv window bytes)."""
        import statistics
        import zlib

        nd, nm = len(self.dense_bucket_numels), len(self.moe_bucket_numels)
        d_order, m_order = (
            [int(i) for i in order[0]],
            [int(i) for i in order[1]],
        )
        if sorted(d_order) != list(range(nd)) or sorted(m_order) != list(
            range(nm)
        ):
            raise RuntimeError(
                f"gin_overlap: the release order is not a permutation of the bucket ids ({nd} dense / {nm} expert)"
            )
        L, Nn = self.ranks_per_node, self.num_nodes
        N = max(self.expert_group_size, 1)
        dm = [n // self.group_size for n in self.dense_bucket_numels]
        mm = [n // N for n in self.moe_bucket_numels]
        seg = [  # slot size of each bucket in each segment
            [
                _align((L - 1) * Nn * m * 4) if L > 1 else 0 for m in dm
            ],  # intra-node segment: only the rows pushed by the other L-1 ranks
            [
                _align(Nn * m * 4) if Nn > 1 else 0 for m in dm
            ],  # inter-node segment: Nn rows, this rank's row stays empty
            [
                _align(N * m * 4) if N > 1 else 0 for m in mm
            ],  # expert segment: N rows, this rank's row stays empty
        ]
        big = [
            max(range(len(s)), key=lambda i: s[i]) if s else -1 for s in seg
        ]  # largest bucket of each segment on this rank
        lb_loc = [_align(max(s, default=0), 4096) for s in seg]
        t = paddle.to_tensor(
            lb_loc[:2], dtype="int64"
        )  # the two dense segments: MAX over the whole world
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.world)
        tm = paddle.to_tensor(
            lb_loc[2:], dtype="int64"
        )  # expert segment: MAX within the expert group
        if moe_group is not None:
            dist.all_reduce(tm, op=dist.ReduceOp.MAX, group=moe_group)
        lb = [int(t[0]), int(t[1]), int(tm[0])]
        names = (
            "dense intra-node segment",
            "dense inter-node segment",
            "expert segment",
        )
        descs = [
            f"{names[i]} lower bound {_gib(lb[i])} GiB (largest bucket on this rank {'dense' if i < 2 else 'expert'}#{big[i]}, slot {_gib(lb_loc[i])} GiB)"
            for i in range(3)
        ]
        P = self._recv_pool_sizes(lb, descs)
        off_i, d_i = self._recv_pool_ring(d_order, seg[0], P[0])
        off_n, d_n = self._recv_pool_ring(d_order, seg[1], P[1])
        off_m, d_m = self._recv_pool_ring(m_order, seg[2], P[2])

        def same(obj, group):
            h = zlib.crc32(repr(obj).encode()) & 0x7FFFFFFF
            v = paddle.to_tensor([h, -h], dtype="int64")
            dist.all_reduce(v, op=dist.ReduceOp.MAX, group=group)
            return int(v[0]) == -int(v[1])

        # Senders compute the write offsets on the peers from their own table, so the table must be identical on all
        # ranks
        ok = same(
            (
                P[0],
                P[1],
                L,
                Nn,
                d_order,
                seg[0],
                seg[1],
                off_i,
                d_i,
                off_n,
                d_n,
            ),
            self.world,
        )
        if moe_group is not None:
            ok = same((P[2], N, m_order, seg[2], off_m, d_m), moe_group) and ok
        v = paddle.to_tensor([int(ok)], dtype="int32")
        dist.all_reduce(v, op=dist.ReduceOp.MIN, group=self.world)
        if int(v[0]) != 1:
            raise RuntimeError(
                "gin_overlap: the recv pool ring layout (segment sizes / fixed order / offsets / predecessors) differs "
                "between ranks (dense must match across the whole world, expert within the group); check "
                "dense_recv_pool_gb / moe_recv_pool_gb and the bucket split on every rank"
            )

        def fit(
            i,
        ):  # estimated number of buckets this segment holds at once, from the median slot size (logs only)
            s = [x for x in seg[i] if x > 0]
            if not s or not P[i]:
                return 0.0, 0
            med = statistics.median(s)
            return med, int(P[i] // med)

        self.recv_pool = {
            "total": sum(P), "lb": lb, "P": P, "desc": "; ".join(descs),
            "big": [(big[i], lb_loc[i]) for i in range(3)], "fit": [fit(i) for i in range(3)],
            # start of each segment: intra-node / inter-node segments in the dense group's recv window, the expert
            # segment in the expert group's (each starting at 0)
            "base": [0, P[0], 0],
            "offsets": (off_i, off_n, off_m),  # per segment {bucket id: byte offset in the segment} (no slot = absent)
            "distance": (d_i, d_n, d_m),  # per segment {bucket id: predecessor distance d} (1..n)
            "dense_order": d_order, "moe_order": m_order,  # fixed release orders (release "positions" index these)
            "buckets": None,  # bucket objects (expert buckets, dense buckets), filled by the caller
            "host_bytes": [0, 0],  # filled by _layout_host_windows: bytes each group's host window needs
            "host_window_bytes": {},  # filled by _setup_group: actual sizes (rounded to the HOST_NUMA granularity)
            "nctx": {},  # filled by _setup_group: GIN context count of each group
        }  # fmt: skip
        return P[0] + P[1], P[2]

    def _recv_pool_sizes(self, lb, descs):
        """Actual sizes of the three segments (bytes, multiples of 4 KB). lb = the three lower bounds, descs = their
        descriptions for error messages."""
        gd, gm = self.dense_recv_pool_gb, self.moe_recv_pool_gb
        P = list(lb)
        Sd = lb[0] + lb[1]
        if Sd > 0:
            Td = int(gd * 2**30)
            if Td < Sd:
                raise RuntimeError(
                    f"dense_recv_pool_gb={gd:g} is too small: the lower bounds of the dense intra-node and inter-node "
                    f"segments add up to {_gib(Sd)} GiB ({Sd} B); {descs[0]}; {descs[1]}"
                )
            # split in proportion to the two lower bounds
            P[0] = (lb[0] * Td // Sd) // 4096 * 4096
            P[1] = (lb[1] * Td // Sd) // 4096 * 4096
            for i in (0, 1):
                if (
                    P[i] < lb[i]
                ):  # cannot happen: lb is a multiple of 4 KB and Td >= Sd
                    raise RuntimeError(
                        f"dense_recv_pool_gb={gd:g}: after the proportional split, {descs[i]} is not met (got {_gib(P[i])} GiB)"
                    )
        if lb[2] > 0:
            Tm = int(gm * 2**30) // 4096 * 4096
            if Tm < lb[2]:
                raise RuntimeError(
                    f"moe_recv_pool_gb={gm:g} is too small: {descs[2]}"
                )
            P[2] = Tm
        return P

    @staticmethod
    def _recv_pool_ring(order, seg, P):
        """Ring layout of one segment: buckets take contiguous ranges in `order`; when one does not fit
        (cur + size > P) it wraps to 0 and the end of the segment stays unused. The layout is the same every step.
        Predecessor = among the earlier holders of the range this bucket's slot covers, the one with the largest
        sequence number (sequence number = step * n + position in order).
        Predecessor distance d = this bucket's sequence number - its predecessor's, taken from step 2 (1..n; n = the
        predecessor is this bucket's own copy from the step before).
        Returns ({bucket: offset}, {bucket: d}); buckets with slot size 0 are absent (they do not use this segment and
        wait for no credit)."""
        n, off, cur = len(order), {}, 0
        for b in order:
            sz = seg[b]
            if sz <= 0:
                continue
            if sz > P:  # cannot happen: P >= lower bound >= largest slot
                raise RuntimeError(
                    f"gin_overlap: slot of bucket {b} is {sz} B, larger than the segment size {P} B"
                )
            if cur + sz > P:
                cur = 0
            off[b] = cur
            cur += sz
        # occ: non-overlapping (start, end, holder sequence number)
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
                    if (
                        pq < 0
                    ):  # in step 2 a bucket covers at least its own copy from step 1
                        raise RuntimeError(
                            f"gin_overlap: no predecessor found for bucket {b} (internal error)"
                        )
                    d[b] = q - pq
        return off, d

    def _layout_host_windows(self):
        """Signal window / host window layout of both groups, each starting at 0. Returns (signal window bytes of the
        dense group, signal window bytes of the expert group) and fills credit / intra_arrive / gather_* / host_bytes
        of recv_pool.

        dense group (sharding group), each area 256 B aligned, in order:
          - signal: 16 B per dense bucket (signal_offsets); the first u64 is the inter-node GIN signal.
          - credit[0] intra-node: 16 counters; credits from node-local rank l go to counter l.
          - credit[1] inter-node: num_nodes counters.
          - intra_arrive, the intra-node arrival counters (meaningful only in the host window): a fixed 16 per dense
            bucket (ranks_per_node must be <= 16), at intra_arrive + (idx*16 + l)*8; only sender l of the same node
            writes counter l, and it writes how many intra-node copies of this bucket it has submitted.
          - gather_dense = (arrive, intra, credit), used by the gather of dense 2D buckets to the owner, sized by the
            number of dense buckets:
            - arrive + 8*idx: inter-node arrival counter on the owner; each inter-node sender's put adds 1.
            - intra + (idx*16 + l)*8: intra-node arrival counter on the owner; intra-node sender l writes its
              sequence number after the CE copy.
            - credit + 8*idx: reserved, never read or written.
        expert group (moe_sharding group): 16 B of signal per expert bucket; credit[2] expert (one counter per expert
          group rank); the three gather_moe areas, same format as gather_dense, sized by the expert bucket ids.
        Counters are written to the same offset of the host window; the signal window only reserves the offsets. The
        layout depends only on the bucket counts and group sizes, so it is the same on all ranks."""
        nd, nm = len(self.dense_bucket_numels), len(self.moe_bucket_numels)
        p = self.recv_pool
        # dense group
        c0 = _align(_align(16 * nd + 64, 256), 256)
        c1 = _align(c0 + 8 * 16, 256)
        dense_signal_end = c1 + 8 * max(self.num_nodes, 1)
        ia = _align(dense_signal_end, 256)
        p["intra_arrive"] = ia
        p["intra_arrive_bytes"] = 8 * 16 * nd
        ga = _align(ia + p["intra_arrive_bytes"], 256)
        gi = _align(ga + 8 * max(nd, 1), 256)
        gc = _align(gi + 8 * 16 * max(nd, 1), 256)
        p["gather_dense"] = (ga, gi, gc)
        p["host_bytes"][GROUP_DENSE] = _align(gc + 8 * max(nd, 1), 4096)
        # expert group
        c2 = _align(_align(16 * nm + 64, 256), 256)
        moe_signal_end = c2 + 8 * max(self.expert_group_size, 1)
        mga = _align(moe_signal_end, 256)
        mgi = _align(mga + 8 * max(nm, 1), 256)
        mgc = _align(mgi + 8 * 16 * max(nm, 1), 256)
        p["gather_moe"] = (mga, mgi, mgc)
        p["host_bytes"][GROUP_MOE] = _align(mgc + 8 * max(nm, 1), 4096)
        p["credit"] = (c0, c1, c2)
        return dense_signal_end, moe_signal_end

    def _configure(self, num_dense, num_moe):
        """Calls configure on the C++ side with the bucket counts, the size of the window mark ring and the offsets of
        the three credit areas, then logs the recv pool. This is the only place bucket counts go to the C++ side."""
        c0, c1, c2 = self.recv_pool["credit"]
        check_cuda(
            device_engine().configure(
                num_dense, num_moe, NUM_WINDOW_MARKS, c0, c1, c2
            ),
            "configure",
        )
        self._log_recv_pool()

    # ------------------------------------------------------------------ parameter tables of the release layer (C++, release.cu)
    def _recv_offset(self, seg, idx):
        """Byte offset in the recv window of bucket idx in segment seg (0 intra-node / 1 inter-node: the dense group's
        recv window; 2 expert: the expert group's) = segment start + offset in the segment."""
        p = self.recv_pool
        return p["base"][seg] + p["offsets"][seg].get(idx, 0)

    def release_setup(self):
        """Hands the fixed release orders and the per-bucket parameter tables to the C++ release layer and starts the
        tail thread. Call it once after __init__.
        - dense, 7 numbers per bucket: [send offset, signal offset, numel of this rank's shard, intra-node segment
          offset, inter-node segment offset, intra-node predecessor distance, inter-node predecessor distance].
        - expert, 5 numbers per bucket: [send offset, signal offset, numel of this rank's shard, expert segment
          offset, predecessor distance].
        - A predecessor distance of -1 means the bucket has no slot in that segment."""
        p, lib = self.recv_pool, device_engine()
        nd, nm = len(self.dense_bucket_numels), len(self.moe_bucket_numels)
        dtab = []
        for i in range(nd):
            so, sg = self.dense_offsets[i]
            di, dn = (
                p["distance"][0].get(i),
                p["distance"][1].get(i),
            )  # predecessor distance, None -> -1
            dtab += [so, sg, self.dense_bucket_numels[i] // self.group_size, self._recv_offset(0, i),
                     self._recv_offset(1, i), -1 if di is None else di, -1 if dn is None else dn]  # fmt: skip
        mtab = []
        for i in range(nm):
            so, sg = self.moe_offsets[i]
            dm = p["distance"][2].get(i)
            mtab += [so, sg, self.moe_bucket_numels[i] // self.expert_group_size, self._recv_offset(2, i),
                     -1 if dm is None else dm]  # fmt: skip
        check_cuda(
            lib.release_setup(
                p["dense_order"], p["moe_order"], dtab, mtab,
                PUT_BLOCKS, p["intra_arrive"], nd + nm + TAIL_EXTRA_MARKS,
            ),
            "release_setup",
        )  # fmt: skip

    # ------------------------------------------------------------------ Muon 2D-bucket gather to the owner
    # Only Muon (which has 2D buckets) uses this part.
    def gather_table_setup(self, moe_2d, dense_2d, moe_group):
        """Builds the 2D-bucket table, checks that all ranks agree, then calls gather_setup on the C++ side.
        Collective: every rank of world must call it at the same time.
        moe_2d / dense_2d = 2D bucket objects (must have _gin_owner_bucket and _gin_bucket_id). The dense table must
        match across the whole world and the expert table within the expert group (bucket id / send offset /
        per-shard size / owner); otherwise raises.
        The result goes into self.gather_table, key = (kind, bucket id); each entry has:
          - group: GROUP_DENSE / GROUP_MOE; slot: event index on the C++ side (the bucket id for dense, number of
            dense buckets + bucket id for expert).
          - send_offset: the bucket's offset in the group's send window; off: offset of this rank's shard; m:
            elements per shard; S: group size; me: this rank's group rank; o: owner's group rank; owner: whether
            this rank is the owner.
          - aoff / ioff / gi0: in the group's host window, the inter-node arrival counter / this rank's intra-node
            arrival counter / the start of the bucket's intra-node arrival counters.
          - the owner also has n_inter (number of inter-node senders) and intra (node-local indices of the
            intra-node senders); a sender also has intra_owner (whether the owner is on the same node)."""
        import zlib

        p = self.recv_pool
        nd = len(self.dense_bucket_numels)
        tab = {}
        for kind, buckets in (("dense", dense_2d), ("moe", moe_2d)):
            ga, gi, _ = (
                p["gather_dense"] if kind == "dense" else p["gather_moe"]
            )
            L = (
                self.ranks_per_node
                if kind == "dense"
                else self.moe_ranks_per_node
            )
            ni = self.node_index if kind == "dense" else self.moe_node_index
            ml = self.rank_in_node if kind == "dense" else self.moe_rank_in_node
            for b in buckets:
                k, idx = b._gin_bucket_id
                assert k == kind, (k, kind)
                _, npad, o = b._gin_owner_bucket
                if kind == "dense":
                    grp, S, me, soff, slot = (
                        GROUP_DENSE,
                        self.group_size,
                        self.group_rank,
                        self.dense_offsets[idx][0],
                        idx,
                    )
                    assert npad == self.dense_bucket_numels[idx], (
                        idx,
                        npad,
                        self.dense_bucket_numels[idx],
                    )
                else:
                    grp, S, me = (
                        GROUP_MOE,
                        self.expert_group_size,
                        self.expert_group_rank,
                    )
                    soff, slot = self.moe_offsets[idx][0], nd + idx
                    assert npad == self.moe_bucket_numels[idx], (
                        idx,
                        npad,
                        self.moe_bucket_numels[idx],
                    )
                m = npad // S
                ent = {
                    "group": grp, "slot": slot, "send_offset": soff, "off": soff + me * m * 4, "m": m, "S": S,
                    "me": me, "o": o, "owner": o == me, "aoff": ga + 8 * idx,
                    "ioff": gi + (idx * 16 + ml) * 8, "gi0": gi + idx * 16 * 8,
                    "ranks_per_node": L, "node_index": ni,
                }  # fmt: skip
                if o == me:
                    peers = [j for j in range(S) if j != me]
                    ent["n_inter"] = sum(1 for j in peers if j // L != ni)
                    ent["intra"] = [j % L for j in peers if j // L == ni]
                else:
                    ent["intra_owner"] = o // L == ni
                tab[(kind, idx)] = ent

        def sig(kind):
            return [
                (i, e["send_offset"], e["m"], e["o"])
                for (k, i), e in sorted(tab.items())
                if k == kind
            ]

        def same(obj, group):
            h = zlib.crc32(repr(obj).encode()) & 0x7FFFFFFF
            v = paddle.to_tensor([h, -h], dtype="int64")
            dist.all_reduce(v, op=dist.ReduceOp.MAX, group=group)
            return int(v[0]) == -int(v[1])

        ok = same(
            (p["gather_dense"][:2], nd, self.ranks_per_node, sig("dense")),
            self.world,
        )
        if moe_group is not None and self.moe_bucket_numels:
            ok = same((p["gather_moe"][:2], len(self.moe_bucket_numels), self.moe_ranks_per_node, sig("moe")),
                      moe_group) and ok  # fmt: skip
        v = paddle.to_tensor([int(ok)], dtype="int32")
        dist.all_reduce(v, op=dist.ReduceOp.MIN, group=self.world)
        if int(v[0]) != 1:
            raise RuntimeError(
                "gin_overlap: the 2D-bucket table (bucket id / send window offset / per-shard size / owner) or the "
                "offsets of the gather counters in the host window differ between ranks (dense must match across the "
                "whole world, expert within the group); senders write the owner at their own offsets, so a mismatch "
                "writes to the wrong place"
            )
        check_cuda(
            device_engine().gather_setup(nd + len(self.moe_bucket_numels)),
            "gather_setup",
        )
        self.gather_table = tab

        own = sum(1 for e in tab.values() if e["owner"])
        rx = sum(e["m"] * 4 * (e["S"] - 1) for e in tab.values() if e["owner"])
        log(
            f"init: 2D-bucket gather to the owner: {len(tab)} buckets (dense {len(dense_2d)} / expert {len(moe_2d)}); "
            f"this rank (rank 0) owns {own} of them and receives {format(rx / 2**20, '.1f')} MiB in the gather; "
            f"host window counters dense arrive={p['gather_dense'][0]} intra={p['gather_dense'][1]}, expert "
            f"arrive={p['gather_moe'][0]} "
            f"intra={p['gather_moe'][1]}"
        )

    def tail_gather_setup(self):
        """Hands the 2D-bucket table to the C++ release layer (call it after release_setup and gather_table_setup,
        with the tail thread in IDLE).
        Each entry: [kind (0 dense / 1 expert), bucket id, slot, whether this rank is the owner, owner's group rank,
        off, m, 1 (reserved), aoff, ioff].
        Later gather puts pass only the entry index. self.gather_keys = entry order; self.gather_entry_index =
        (kind, bucket id) -> entry index."""
        keys = sorted(self.gather_table)
        self.gather_keys = keys
        self.gather_entry_index = {key: i for i, key in enumerate(keys)}
        if not keys:  # no 2D buckets: no table
            return
        rows = []
        for kind, idx in keys:
            e = self.gather_table[(kind, idx)]
            rows += [int(kind == "moe"), idx, e["slot"], int(e["owner"]), e["o"], e["off"], e["m"], 1, e["aoff"],
                     e["ioff"]]  # fmt: skip
        check_cuda(device_engine().tail_gather_setup(rows), "tail_gather_setup")

    def tail_gather_set(self, seqs):
        """Hands this step's put sequence number of every entry (0 = the tail thread does not send it in this step)
        to the C++ side. Call it before handing off to the tail thread."""
        if not self.gather_keys:
            return
        check_cuda(
            device_engine().tail_gather_set_sequences(list(seqs)),
            "tail_gather_set_sequences",
        )

    def tail_gather_collect(self):
        """Gather puts the tail thread sent in this step, [((kind, bucket id), sequence number)], in send order. The
        tail thread must have left RUN."""
        if not self.gather_keys:
            return []
        rc, flat = device_engine().tail_gather_get_sent()
        check_cuda(rc, "tail_gather_get_sent")
        return [
            (
                ("moe" if flat[3 * i] else "dense", int(flat[3 * i + 1])),
                int(flat[3 * i + 2]),
            )
            for i in range(len(flat) // 3)
        ]

    # ------------------------------------------------------------------ init logs
    def _log_recv_pool(self):
        """Init log of the recv pool (rank 0): the three segment sizes, lower bounds, the largest bucket on this rank,
        how many buckets fit at once (estimated from the median slot size), and how many predecessors are in the
        same step."""
        p = self.recv_pool
        names = (
            "dense intra-node segment",
            "dense inter-node segment",
            "expert segment",
        )
        segs = []
        for i in range(3):
            (bi, bsz), (med, k) = p["big"][i], p["fit"][i]
            segs.append(
                f"{names[i]} {_gib(p['P'][i])} GiB (lower bound {_gib(p['lb'][i])}, largest bucket on this rank "
                f"{'dense' if i < 2 else 'expert'}#{bi} slot {_gib(bsz)} GiB, median slot {format(med / 2**20, '.1f')} MiB, "
                f"holds {k} at once)"
            )
        nd, nm = len(self.dense_bucket_numels), len(self.moe_bucket_numels)
        n = (nd, nd, nm)
        inner = [
            sum(1 for v in p["distance"][i].values() if v < n[i])
            for i in range(3)
        ]  # d < n: waits for an earlier bucket of the same step to free the slot
        hw = p["host_window_bytes"]
        hw_txt = ", ".join(
            f"{GROUP_NAME[g]} group {format(hw[g] / 2**20, '.1f')} MiB (needs {p['host_bytes'][g]} B)"
            for g in sorted(hw)
        )
        log(
            "init: recv pool (ring layout, several buckets in flight; the CPU reads the host window before releasing): "
            + "; ".join(segs)
            + f"; total {_gib(p['total'])} GiB (intra-node / inter-node segments in the sharding group's recv window, "
            f"expert segment in the moe_sharding group's); predecessors in the same step: intra-node {inner[0]} / "
            f"inter-node {inner[1]} / expert {inner[2]}; host window per rank {hw_txt} (HOST_NUMA; the intra-node "
            f"arrival area is in the sharding group's, at offset {p['intra_arrive']}, "
            f"{p['intra_arrive_bytes']} B = {nd} buckets x 16 counters x 8 B, one counter per sender)"
        )

    def _log_init(self, t, extra=""):
        """Every rank prints one line with the time of each init phase; rank 0 also logs the topology, buckets,
        traffic and environment variables. extra goes in front of the environment variables."""
        ms = [format((b - a) * 1e3, ".0f") for a, b in zip(t, t[1:])]
        print(
            f"[gin_overlap] init: rank {dist.get_rank()} node {self.node_index} local rank {self.rank_in_node}: "
            f"get Paddle's two ncclComms {ms[0]} / window registration + DevCommCreate {ms[1]} / setup {ms[2]} / "
            f"final barrier {ms[3]} ms",
            flush=True,
        )
        if dist.get_rank() != 0:
            return
        L, Nn, N, W = (
            self.ranks_per_node,
            self.num_nodes,
            self.expert_group_size,
            self.group_size,
        )
        v = device_engine().nccl_runtime_version()
        p = self.recv_pool
        moe_blk = [n // N * 4 / 2**20 for n in self.moe_bucket_numels] or [0]
        den_blk = [n // W * 4 / 2**20 for n in self.dense_bucket_numels] or [0]
        net = sum((N - 1) * (n // N) * 4 for n in self.moe_bucket_numels)
        net += sum((Nn - 1) * (n // W) * 4 for n in self.dense_bucket_numels)
        ce = sum((L - 1) * Nn * (n // W) * 4 for n in self.dense_bucket_numels)
        nput = len(self.moe_bucket_numels) * (N - 1) + len(
            self.dense_bucket_numels
        ) * (Nn - 1)
        envs = " ".join(
            f"{k}={os.getenv(k, '<unset>')}"
            for k in (
                "NCCL_IB_TC",
                "NCCL_GIN_IB_TC",
                "NVSHMEM_IB_TRAFFIC_CLASS",
                "NCCL_NVLS_ENABLE",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "CUDA_MODULE_LOADING",
            )
        )
        wins = "; ".join(
            f"{GROUP_NAME[g]} group send {_gib(self._win[g][0])} + recv {_gib(self._win[g][1])} GiB (GIN ctx="
            f"{p['nctx'][g]})" for g in sorted(self._win)
        )  # fmt: skip
        log(
            f"init: NCCL {v // 10000}.{v // 100 % 100}.{v % 100}, sharding group {Nn} nodes x {L} ranks (group size "
            f"{W}, GPUs per node {self._local_size}), moe_sharding group {self.moe_num_nodes} nodes x "
            f"{self.moe_ranks_per_node} ranks (group size {N}); window {wins}"
        )
        log(
            f"init: expert {len(self.moe_bucket_numels)} buckets (group of {N} ranks, peers by group rank, one put per "
            f"peer, per-peer block {format(min(moe_blk), '.3f')}~{format(max(moe_blk), '.3f')} MB); "
            f"dense {len(self.dense_bucket_numels)} buckets ({Nn - 1} peers with the same rank_in_node on other nodes, "
            f"per-peer block {format(min(den_blk), '.3f')}~{format(max(den_blk), '.3f')} MB)"
        )
        log(
            f"init: per step per rank inter-node {format(net / 1e9, '.2f')} GB / {nput} puts, intra-node CE {format(ce / 1e9, '.2f')} GB"
        )
        log(f"init: {extra}{envs}")
