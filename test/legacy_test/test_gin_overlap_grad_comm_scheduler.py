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

"""Single-process CPU tests of the gin_overlap scheduler builder
(python/paddle/distributed/communication/gin_overlap/grad_comm_scheduler.py) and of the gin_overlap branches of
FusedCommBuffer and the sharding optimizers.

The builder is Python on top of paddle.base.core.gin_overlap, which only exists in builds with WITH_GIN_OVERLAP. These
tests never use it: the module's _LIB is replaced by a namespace holding the two bucket kind constants, the scheduler
class by a recorder, and the collectives by single-process stand-ins under which every rank holds this rank's values
(all_reduce MAX / MIN leave the tensor unchanged). Groups, process groups and buckets are small fakes carrying only
the attributes the builder reads. Tensors are real CPU tensors where the builder moves data (send windows, Muon 2D
padding, parameter all_gather).
"""

import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import paddle
from paddle.distributed.communication.gin_overlap import (
    grad_comm_scheduler as gcs,
)
from paddle.distributed.fleet.meta_optimizers import (
    mixed_sharding_optimizer,
    muon_sharding_optimizer,
)
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    dygraph_sharding_optimizer,
)
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizerV2,
)
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer,
)
from paddle.distributed.fleet.meta_optimizers.mixed_sharding_optimizer import (
    MixedShardingOptimizer,
)
from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
    MuonShardingOptimizer,
)
from paddle.distributed.fleet.utils import tensor_fusion_helper
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    HOOK_ACTION,
    FusedCommBuffer,
)

# Stand-in for core.gin_overlap: bucket_table() reads only these constants.
FAKE_LIB = SimpleNamespace(DENSE_BUCKET=0, EXPERT_BUCKET=1)
DENSE, EXPERT = FAKE_LIB.DENSE_BUCKET, FAKE_LIB.EXPERT_BUCKET


def gib(nbytes):
    """The recv pool size in GiB (the unit of the sharding config) that is exactly nbytes."""
    return nbytes / 2**30


def same_on_every_rank_all_reduce(tensor, op=None, group=None):
    """all_reduce when every rank holds this rank's values: MAX and MIN leave the tensor as it is."""


class FakeProcessGroup:
    """process_group of a fake group: the window API of _GroupWindows, and an all_gather whose row r is the sent
    tensor + 1000 * r (so the test can tell which rank a gathered value came from)."""

    def __init__(self, nranks, lsa_size, lsa_rank, window_handle=7):
        self.nranks = nranks
        self.dev_comm = {
            "gin_context_count": 8,
            "lsa_size": lsa_size,
            "lsa_rank": lsa_rank,
        }
        self.window_handle = window_handle
        self.registered = []
        self.gathered_numels = []

    def register_comm_buffer(self, tensor):
        self.registered.append(tensor)
        return self.window_handle

    def alloc_host_window(self, nbytes):
        return self.window_handle, 0, gcs._align(nbytes, 4096)

    def get_or_create_dev_comm(self, gin_contexts):
        return self.dev_comm

    def all_gather(self, send, recv):
        # sync_params gathers in place (send is a slice of recv): read it before any row is written
        send = send.clone()
        total = send._numel()
        self.gathered_numels.append(total)
        for src_rank in range(self.nranks):
            recv._slice(src_rank * total, (src_rank + 1) * total).copy_(
                send + 1000.0 * src_rank, False
            )
        return SimpleNamespace(wait=lambda: None)


class FakeGroup:
    """A communication group: global ranks, this process's group rank, id, and a FakeProcessGroup whose LSA team is
    the node of this rank (ranks_per_node consecutive ranks)."""

    def __init__(self, ranks, rank=0, group_id=0, ranks_per_node=1):
        self.ranks = list(ranks)
        self.nranks = len(self.ranks)
        self.rank = rank
        self.id = group_id
        self.process_group = FakeProcessGroup(
            self.nranks, ranks_per_node, rank % ranks_per_node
        )


def param(name, numel=4, shape=None, is_moe_param=False, color=None):
    p = SimpleNamespace(name=name, shape=shape or [numel], main_grad=None)
    p._numel = lambda: numel
    p.is_moe_param = is_moe_param
    if color is not None:
        p.color = color
    p.backward_hooks = []
    p._register_backward_hook = p.backward_hooks.append
    return p


def expert_param(name, numel=4, shape=None):
    return param(name, numel, shape, is_moe_param=True)


class FakeGradView:
    """_sharding_param_grad_view entry of a 1D bucket parameter."""

    def __init__(self):
        self._grad_buffer = None
        self.relinked = 0

    def _link_grad_to_buffer(self):
        self.relinked += 1


def bucket(
    params,
    group,
    act=HOOK_ACTION.REDUCE_SCATTER,
    grad_storage=None,
    frozen=False,
    **overrides,
):
    """A fake FusedCommBuffer that every check of _reject_reason and _fallback_reason accepts, unless overridden.
    _gin_scheduler / _gin_reported start as FusedCommBuffer.__init__ leaves them, so a bucket that keeps its own
    collective is recognisable by _gin_scheduler staying None."""
    fake = SimpleNamespace(
        _params=params,
        _comm_group=group,
        _act=act,
        _release_grads=False,
        _free_grads_in_comm=False,
        grad_storage=grad_storage
        if grad_storage is not None
        else SimpleNamespace(dtype=paddle.float32),
        use_main_grad=True,
        _scale_after_comm=True,
        _use_reduce_avg=False,
        _sharding_param_grad_view={p.name: FakeGradView() for p in params},
        recorded_addr=0,
        _gin_scheduler=None,
        _gin_reported=False,
    )
    fake.need_reduce_scale_sync = lambda: not frozen
    fake._record_addr = lambda: setattr(
        fake, "recorded_addr", fake.recorded_addr + 1
    )
    for key, value in overrides.items():
        setattr(fake, key, value)
    return fake


class SingleProcessTestCase(unittest.TestCase):
    """Runs the builder as one rank of groups whose ranks all agree, with the fake core.gin_overlap constants."""

    def setUp(self):
        for patcher in (
            mock.patch.object(gcs, "_LIB", FAKE_LIB),
            mock.patch.object(gcs.dist, "all_reduce", same_on_every_rank_all_reduce),
            mock.patch.object(gcs.dist, "barrier", lambda group=None: None),
            mock.patch.object(gcs.dist, "get_rank", lambda: 0),
            mock.patch.object(gcs.dist, "in_auto_parallel_align_mode", lambda: False),
            mock.patch.object(gcs.paddle.device, "synchronize", lambda: None),
            mock.patch("paddle.distributed.collective._nccl_symmetric_empty", paddle.zeros, create=True),
            mock.patch.object(gcs, "scheduler", None),
            mock.patch.dict(os.environ, {"PADDLE_LOCAL_SIZE": "2", "FLAGS_pp_check_naninf": "0"}),
        ):  # fmt: skip
            patcher.start()
            self.addCleanup(patcher.stop)


class TestSendLayout(unittest.TestCase):
    def test_align(self):
        for nbytes, alignment, aligned in (
            (0, 256, 0),
            (1, 256, 256),
            (256, 256, 256),
            (257, 256, 512),
            (5000, 4096, 8192),
        ):
            self.assertEqual(gcs._align(nbytes, alignment), aligned)

    def test_every_bucket_starts_256_byte_aligned(self):
        # 10 / 64 / 3 fp32 = 40 / 256 / 12 bytes
        self.assertEqual(gcs._send_layout([10, 64, 3]), ([0, 256, 512], 768))
        self.assertEqual(gcs._send_layout([]), ([], 0))


class TestRecvPoolRing(unittest.TestCase):
    # (case, release order, slot bytes by bucket, segment bytes, offset by bucket, predecessor distance by bucket)
    CASES = (
        ("no wrap: each bucket holds its range alone, so d = number of buckets",
         [0, 1], [100, 100], 200, {0: 0, 1: 100}, {0: 2, 1: 2}),
        ("wrap: bucket 2 reuses the range of bucket 0 in the same step",
         [0, 1, 2], [100, 100, 100], 250, {0: 0, 1: 100, 2: 0}, {0: 1, 1: 3, 2: 2}),
        ("a slot that ends exactly at the segment end does not wrap",
         [0, 1, 2], [100, 150, 50], 250, {0: 0, 1: 100, 2: 0}, {0: 1, 1: 3, 2: 2}),
        ("partial overlaps: the predecessor is the latest of several holders",
         [0, 1, 2], [100, 100, 150], 250, {0: 0, 1: 100, 2: 0}, {0: 1, 1: 2, 2: 1}),
        ("the release order, not the bucket id, decides the placement",
         [2, 0, 1], [100, 100, 150], 250, {2: 0, 0: 150, 1: 0}, {2: 1, 0: 3, 1: 2}),
        ("zero-byte slots are skipped but still count in the sequence",
         [0, 1, 2], [100, 0, 100], 300, {0: 0, 2: 100}, {0: 3, 2: 3}),
        ("a smaller slot over the start of a larger one leaves the larger holder's tail",
         [0, 1], [200, 100], 250, {0: 0, 1: 0}, {0: 1, 1: 1}),
        ("one bucket filling the whole segment",
         [0], [4096], 4096, {0: 0}, {0: 1}),
    )  # fmt: skip

    def test_offsets_and_predecessor_distances(self):
        for case, order, slots, size, offsets, distances in self.CASES:
            with self.subTest(case):
                self.assertEqual(
                    gcs._recv_pool_ring(order, slots, size),
                    (offsets, distances),
                )

    def test_slot_larger_than_segment_raises(self):
        with self.assertRaisesRegex(RuntimeError, "larger than its segment"):
            gcs._recv_pool_ring([0, 1], [100, 300], 250)


class TestRecvPoolSizes(unittest.TestCase):
    TEXTS = ("intra bound", "inter bound", "expert bound")
    # largest (dense, expert) bucket in bytes; only used when a pool size is auto (<= 0)
    NO_BUCKETS = (0, 0)

    def test_dense_pool_split_in_proportion_to_the_lower_bounds(self):
        # 40960 B split 3:1 = 30720 / 10240, each rounded down to 4 KB
        sizes = gcs._recv_pool_sizes(
            [12288, 4096, 0], self.TEXTS, gib(40960), 1.0, self.NO_BUCKETS
        )
        self.assertEqual(sizes, [28672, 8192, 0])

    def test_expert_pool_rounded_down_to_4kb(self):
        sizes = gcs._recv_pool_sizes(
            [0, 0, 8192], self.TEXTS, 1.0, gib(12388), self.NO_BUCKETS
        )
        self.assertEqual(sizes, [0, 0, 12288])

    def test_auto_pool_is_twice_the_largest_bucket_rounded_up_to_a_gib(self):
        for largest_bucket, pool_gib in ((1, 1), (2**29, 1), (2**29 + 1, 2), (3 * 2**30, 6), (3 * 2**30 + 4096, 7)):  # fmt: skip
            with self.subTest(largest_bucket=largest_bucket):
                self.assertEqual(gcs._auto_pool_bytes(largest_bucket), pool_gib * 2**30)  # fmt: skip

    def test_auto_dense_pool_is_split_like_an_explicit_one(self):
        # 4 nodes x 8 ranks, largest dense bucket 3 GiB + 4 KB -> auto 7 GiB, the same split as 7 GiB set by hand
        largest = 3 * 2**30 + 4096
        bounds = [gcs._align(largest * 7 // 8, 4096), gcs._align(largest // 8, 4096), 0]  # fmt: skip
        auto = gcs._recv_pool_sizes(bounds, self.TEXTS, 0.0, 0.0, (largest, 0))
        explicit = gcs._recv_pool_sizes(bounds, self.TEXTS, 7.0, 0.0, (0, 0))
        self.assertEqual(auto, explicit)
        # each segment is rounded down to 4 KB, so together they may be one 4 KB page short of the 7 GiB
        self.assertIn(7 * 2**30 - sum(auto), (0, 4096))
        self.assertGreaterEqual(auto[0], bounds[0])
        self.assertGreaterEqual(auto[1], bounds[1])

    def test_auto_expert_pool(self):
        sizes = gcs._recv_pool_sizes([0, 0, 2**28], self.TEXTS, 0.0, -1.0, (0, 2**28))  # fmt: skip
        self.assertEqual(sizes, [0, 0, 2**30])

    def test_auto_pools_always_meet_the_lower_bounds(self):
        # the slots of one bucket of B bytes: (L - 1) / L of it intra-node, 1 / L inter-node, all of it for experts
        for nodes in (1, 2, 4, 16):
            for ranks_per_node in (1, 2, 8, 16):
                for shard_numel in (1, 3, 1000, 2**20 + 7, 2**28 + 1):
                    with self.subTest(nodes=nodes, ranks_per_node=ranks_per_node, shard_numel=shard_numel):  # fmt: skip
                        largest = 4 * shard_numel * nodes * ranks_per_node
                        intra = gcs._align((ranks_per_node - 1) * nodes * shard_numel * 4) if ranks_per_node > 1 else 0  # fmt: skip
                        inter = gcs._align(nodes * shard_numel * 4) if nodes > 1 else 0  # fmt: skip
                        bounds = [gcs._align(intra, 4096), gcs._align(inter, 4096), gcs._align(gcs._align(largest), 4096)]  # fmt: skip
                        sizes = gcs._recv_pool_sizes(bounds, self.TEXTS, 0.0, 0.0, (largest, largest))  # fmt: skip
                        for size, bound in zip(sizes, bounds):
                            self.assertGreaterEqual(size, bound)

    def test_too_small_pools_raise(self):
        cases = (
            ([8192, 8192, 0], gib(12288), 1.0, "dense_recv_pool_gb=.* is too small.*intra bound.*inter bound"),
            # unaligned bounds: the 4 KB rounding of the split drops both below their bound
            ([5000, 5000, 0], gib(10000), 1.0, "after the split, intra bound is not met"),
            ([0, 0, 8192], 1.0, gib(4096), "expert_recv_pool_gb=.* is too small: expert bound"),
        )  # fmt: skip
        for bounds, dense_gb, expert_gb, message in cases:
            with (
                self.subTest(message),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                gcs._recv_pool_sizes(
                    bounds, self.TEXTS, dense_gb, expert_gb, self.NO_BUCKETS
                )


class TestReleaseOrder(unittest.TestCase):
    def setUp(self):
        self.params = [param(f"p{i}") for i in range(6)]
        self.opt = SimpleNamespace(_parameter_list=self.params)

    def test_descending_by_first_parameter_position(self):
        params = self.params
        buckets = [
            bucket([params[4]], None),
            bucket([params[0], params[5]], None),
            bucket([params[2], params[3]], None),
        ]
        self.assertEqual(gcs._release_order(self.opt, buckets), [0, 2, 1])

    def test_ties_go_to_the_larger_bucket_index(self):
        params = self.params
        buckets = [
            bucket([params[1]], None),
            bucket([params[1], params[2]], None),
            bucket([params[0]], None),
        ]
        self.assertEqual(gcs._release_order(self.opt, buckets), [1, 0, 2])

    def test_unplaceable_buckets_raise(self):
        with self.assertRaisesRegex(RuntimeError, "has no parameters"):
            gcs._release_order(self.opt, [bucket([], None)])
        with self.assertRaisesRegex(
            RuntimeError, r"1 parameters .* \['stranger'\]"
        ):
            gcs._release_order(
                self.opt, [bucket([self.params[0], param("stranger")], None)]
            )


class TestBucketClassification(unittest.TestCase):
    def test_reject_reason(self):
        group = FakeGroup(range(4))
        cases = (
            ("accepted", {}, None),
            ("other act", {"act": HOOK_ACTION.REDUCE}, "act="),
            ("release_grads", {"_release_grads": True}, "release_grads=True"),
            ("free_grads_in_comm", {"_free_grads_in_comm": True}, "free_grads_in_comm=True"),
            ("no grad_storage", {"grad_storage": None}, "grad_storage is None"),
            ("bf16 grads", {"grad_storage": SimpleNamespace(dtype=paddle.bfloat16)}, "not fp32 main_grad"),
            ("no main_grad", {"use_main_grad": False}, "not fp32 main_grad"),
        )  # fmt: skip
        for case, overrides, reason in cases:
            with self.subTest(case):
                b = bucket([param("a")], group, **overrides)
                if "grad_storage" in overrides:
                    b.grad_storage = overrides["grad_storage"]
                result = gcs._reject_reason(b, HOOK_ACTION.REDUCE_SCATTER)
                if reason is None:
                    self.assertIsNone(result)
                else:
                    self.assertIn(reason, result)

    def test_fallback_reason(self):
        group = FakeGroup(range(4))
        no_hook = param("shared", color={"color": "moe_weight_no_hook"})
        cases = (
            ("accepted", {}, None),
            ("sync-only bucket", {"sync_only": True}, "shared-parameter bucket"),
            ("no group", {"_comm_group": None}, "has 1 rank"),
            ("one-rank group", {"_comm_group": FakeGroup([0])}, "has 1 rank"),
            ("no-hook color", {"_params": [param("a"), no_hook]}, 'no-hook color "moe_weight_no_hook"'),
            ("frozen", {"frozen": True}, "frozen"),
            ("scaled before comm", {"_scale_after_comm": False}, "scaled in place before every _comm_grads"),
        )  # fmt: skip
        for case, overrides, reason in cases:
            with self.subTest(case):
                overrides = dict(overrides)
                sync_only = overrides.pop("sync_only", False)
                params = overrides.pop("_params", [param("a")])
                b = bucket(params, group, **overrides)
                sync_ids = (id(b),) if sync_only else ()
                result = gcs._fallback_reason(b, sync_ids)
                if reason is None:
                    self.assertIsNone(result)
                else:
                    self.assertIn(reason, result)

    def test_scaled_before_comm_is_accepted_with_reduce_avg(self):
        b = bucket(
            [param("a")],
            FakeGroup(range(4)),
            _scale_after_comm=False,
            _use_reduce_avg=True,
        )
        self.assertIsNone(gcs._fallback_reason(b))


class TestSplitBuckets(unittest.TestCase):
    def setUp(self):
        self.world = FakeGroup(range(8), group_id=1)
        self.moe_group = FakeGroup(range(4), group_id=2)

    def split(self, buckets, moe_group="default"):
        moe_group = self.moe_group if moe_group == "default" else moe_group
        return gcs._split_buckets(
            buckets,
            HOOK_ACTION.REDUCE_SCATTER,
            self.world,
            moe_group,
            (),
            "test",
        )

    def test_buckets_are_split_by_their_communication_group(self):
        dense = bucket([param("d0"), param("d1")], self.world)
        # an expert parallel size of 1 leaves a sub-model's experts in the
        # sharding group, and a bucket may mix both kinds
        unsharded_expert = bucket([expert_param("e0")], self.world)
        mixed = bucket([param("d2"), expert_param("e1")], self.moe_group)
        expert = bucket(
            [param("e2", color={"color": "moe_expert"})], self.moe_group
        )
        self.assertEqual(
            self.split([dense, unsharded_expert, mixed, expert]),
            ([mixed, expert], [dense, unsharded_expert]),
        )

    def test_buckets_that_keep_their_own_collective(self):
        own = (
            bucket([param("one_rank")], FakeGroup([0])),
            bucket(
                [param("shared", color={"color": "moe_weight_no_hook"})],
                self.world,
            ),
            bucket([param("other_group")], FakeGroup(range(2), group_id=3)),
            bucket([param("frozen")], self.world, frozen=True),
        )
        taken = bucket([param("fine")], self.world)
        self.assertEqual(self.split([*own, taken]), ([], [taken]))

    def test_the_first_rank_of_the_sharding_group_logs_the_fallback_buckets(
        self,
    ):
        """One warning per sharding group (world.rank == 0), not global rank 0 only: with pipeline parallelism each
        stage has its own buckets and its own sharding group, so every stage must report its own."""
        for rank, logged in ((0, True), (1, False)):
            with self.subTest(sharding_rank=rank):
                world = FakeGroup(range(8), rank=rank, group_id=1)
                buckets = [
                    bucket([param("frozen")], world, frozen=True),
                    bucket([param("fine")], world),
                ]
                with mock.patch.object(gcs, "logger") as logger:
                    expert, dense = gcs._split_buckets(
                        buckets,
                        HOOK_ACTION.REDUCE_SCATTER,
                        world,
                        self.moe_group,
                        (),
                        "test",
                    )
                self.assertEqual((expert, dense), ([], [buckets[1]]))
                self.assertEqual(logger.warning.called, logged)
                if logged:
                    message = logger.warning.call_args[0][0]
                    self.assertIn(
                        "test (global rank 0): 1/2 buckets keep their own reduce_scatter / reduce",
                        message,
                    )
                    self.assertIn("['frozen']", message)

    def test_unsupported_buckets_raise(self):
        cases = (
            ("other act", bucket([param("a")], self.world, act=HOOK_ACTION.REDUCE), "act="),
            ("release_grads", bucket([param("a")], self.world, _release_grads=True), "release_grads=True"),
        )  # fmt: skip
        for case, b, message in cases:
            with (
                self.subTest(case),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                self.split([b])

    def test_reasons_are_aggregated(self):
        buckets = [
            bucket([param("a")], self.world, _release_grads=True),
            bucket([param("b")], self.world, _release_grads=True),
            bucket([param("c")], self.world, act=HOOK_ACTION.REDUCE),
            bucket([param("fine")], self.world),
        ]
        with self.assertRaises(RuntimeError) as raised:
            self.split(buckets)
        message = str(raised.exception)
        self.assertIn(
            "gin_overlap (test): 3/4 buckets cannot be used with gin_overlap",
            message,
        )
        self.assertIn("2: release_grads=True", message)
        self.assertIn("['a', 'b']", message)
        self.assertIn("1: act=", message)


class TestTopology(unittest.TestCase):
    def topology(self, ranks, local_size=None, device_count=0):
        env = (
            {"PADDLE_LOCAL_SIZE": local_size} if local_size is not None else {}
        )
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(
                gcs.paddle.device.cuda, "device_count", lambda: device_count
            ),
        ):
            if local_size is None:
                os.environ.pop("PADDLE_LOCAL_SIZE", None)
            return gcs._topology(FakeGroup(ranks), "sharding")

    def test_members_per_node(self):
        cases = (
            ("one node", list(range(8)), "8", 0, (8, 8)),
            ("two nodes, 4 each", [0, 1, 2, 3, 8, 9, 10, 11], "8", 0, (4, 8)),
            ("one member per node", [0, 8, 16, 24], "8", 0, (1, 8)),
            ("16 per node", list(range(32)), "16", 0, (16, 16)),
            ("unset: device count", list(range(4)), None, 2, (2, 2)),
            ("unset and no device: 1 GPU per node", list(range(4)), None, 0, (1, 1)),
            ("empty: device count", list(range(4)), " ", 4, (4, 4)),
            ("not a number: device count", list(range(4)), "eight", 4, (4, 4)),
            ("not positive: device count", list(range(4)), "0", 4, (4, 4)),
        )  # fmt: skip
        for case, ranks, local_size, device_count, expected in cases:
            with self.subTest(case):
                self.assertEqual(
                    self.topology(ranks, local_size, device_count), expected
                )

    def test_unsupported_rank_lists(self):
        cases = (
            ("descending", [1, 0], "8", "not ascending"),
            ("uneven nodes", [0, 1, 2, 8], "8", "different number of members on different nodes"),
            ("more than 16 per node", list(range(32)), "32", "at most 16"),
        )  # fmt: skip
        for case, ranks, local_size, message in cases:
            with (
                self.subTest(case),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                self.topology(ranks, local_size)

    def test_group_layout(self):
        with mock.patch.dict(os.environ, {"PADDLE_LOCAL_SIZE": "4"}):
            layout = gcs._GroupLayout(FakeGroup(range(8), rank=6), "sharding")
        self.assertEqual(
            (layout.size, layout.rank, layout.ranks_per_node, layout.num_nodes, layout.rank_in_node, layout.node_index),
            (8, 6, 4, 2, 2, 1),
        )  # fmt: skip


class TestCheckCompat(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(
            gcs.dist, "in_auto_parallel_align_mode", lambda: False
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def opt(self, dp_degree=1, **attrs):
        hcg = SimpleNamespace(get_data_parallel_world_size=lambda: dp_degree)
        return SimpleNamespace(_hcg=hcg, **attrs)

    def test_supported_configuration(self):
        with mock.patch.dict(os.environ, {"FLAGS_pp_check_naninf": "0"}):
            gcs._check_compat(self.opt(), "test")

    def test_align_mode_is_rejected(self):
        # align mode divides the loss already; gin_overlap would divide by the group size again
        with (
            mock.patch.dict(os.environ, {"FLAGS_pp_check_naninf": "0"}),
            mock.patch.object(
                gcs.dist, "in_auto_parallel_align_mode", lambda: True
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "gin_overlap \\(test\\) does not support align mode",
            ),
        ):
            gcs._check_compat(self.opt(), "test")

    def test_unsupported_configurations(self):
        cases = (
            ("FLAGS_pp_check_naninf", "1", self.opt(), "FLAGS_pp_check_naninf=1"),
            ("dp_degree", "0", self.opt(dp_degree=2), "dp_degree > 1 \\(got 2\\)"),
            ("comm_overlap", "0", self.opt(comm_overlap=True), "comm_overlap"),
            ("pp_overlap", "0", self.opt(pp_overlap=True), "sharding_comm_overlap"),
            ("release_gradients", "0", self.opt(sd_release_grads=True), "release_gradients"),
        )  # fmt: skip
        for case, check_naninf, opt, message in cases:
            with (
                self.subTest(case),
                mock.patch.dict(
                    os.environ, {"FLAGS_pp_check_naninf": check_naninf}
                ),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                gcs._check_compat(opt, "test")


def two_rank_max_all_reduce(other_rank_objects):
    """all_reduce of _all_ranks_same as seen with a second rank that checks other_rank_objects: MAX merges the
    other rank's [crc, -crc]; the MIN of the result flag is the same on both ranks, so it is left as it is."""
    other_objects = iter(other_rank_objects)

    def all_reduce(tensor, op=None, group=None):
        if op == gcs.dist.ReduceOp.MAX:
            crc = (
                gcs.zlib.crc32(repr(next(other_objects)).encode()) & 0x7FFFFFFF
            )
            other = paddle.to_tensor([crc, -crc], dtype="int64")
            tensor.copy_(paddle.maximum(tensor, other), False)

    return all_reduce


class TestAllRanksSame(unittest.TestCase):
    def check(self, objects, other_rank_objects):
        group = FakeGroup(range(2))
        with mock.patch.object(
            gcs.dist, "all_reduce", two_rank_max_all_reduce(other_rank_objects)
        ):
            return gcs._all_ranks_same([(obj, group) for obj in objects], group)

    def test_equal_and_differing_objects(self):
        table = (1, [2, 3], {4: 5})
        self.assertTrue(self.check([table], [table]))
        self.assertTrue(self.check([table, "expert"], [table, "expert"]))
        self.assertFalse(self.check([table], [(1, [2, 3], {4: 6})]))
        # a difference in any group makes the result False
        self.assertFalse(self.check([table, "expert"], [table, "other"]))
        self.assertFalse(self.check(["dense", table], ["other", table]))


# The layout used by the _Layout tests, with PADDLE_LOCAL_SIZE=2:
#   sharding group 4 ranks = 2 nodes x 2, moe_sharding group 2 ranks on one node.
#   dense buckets 1024 / 512 / 1536 fp32 -> shards 256 / 128 / 384 -> intra-node and inter-node slots 8 B x shard =
#   2048 / 1024 / 3072 B; expert buckets 128 / 64 -> shards 64 / 32 -> slots 512 / 256 B. Every lower bound rounds
#   up to 4096 B. Dense pool 8192 B -> 4096 + 4096; expert pool 4096 B.
#   Release order dense [2, 0, 1]: bucket 2 at 0, bucket 0 wraps to 0, bucket 1 at 2048; bucket 2 then bucket 0 take
#   over the range of the bucket released just before them (d = 1), bucket 1 the range of bucket 2 (d = 2).
DENSE_NUMELS, EXPERT_NUMELS = [1024, 512, 1536], [128, 64]
RELEASE_ORDER = ([2, 0, 1], [1, 0])
POOL_CONFIG = SimpleNamespace(
    gin_overlap_dense_recv_pool_gb=gib(8192),
    gin_overlap_expert_recv_pool_gb=gib(4096),
)


class LayoutTestCase(SingleProcessTestCase):
    def setUp(self):
        super().setUp()
        self.world = FakeGroup(range(4), rank=1, group_id=1, ranks_per_node=2)
        self.moe_group = FakeGroup([0, 1], rank=1, group_id=2, ranks_per_node=2)
        self.dense_layout = gcs._GroupLayout(self.world, "sharding")
        self.expert_layout = gcs._GroupLayout(self.moe_group, "moe_sharding")

    def layout(
        self,
        dense_numels=DENSE_NUMELS,
        expert_numels=EXPERT_NUMELS,
        order=RELEASE_ORDER,
    ):
        return gcs._Layout(
            self.dense_layout,
            self.expert_layout,
            dense_numels,
            expert_numels,
            order,
            POOL_CONFIG,
        )


class TestLayout(LayoutTestCase):
    def test_send_recv_and_host_windows(self):
        layout = self.layout()
        self.assertEqual(
            (layout.dense_send, layout.dense_send_bytes),
            ([0, 4096, 6144], 12288),
        )
        self.assertEqual(
            (layout.expert_send, layout.expert_send_bytes), ([0, 512], 768)
        )
        self.assertEqual(layout.lower_bounds, [4096, 4096, 4096])
        self.assertEqual(layout.segment_bytes, [4096, 4096, 4096])
        self.assertEqual(
            layout.segment_slots,
            [[2048, 1024, 3072], [2048, 1024, 3072], [512, 256]],
        )
        self.assertEqual(layout.base, [0, 4096, 0])
        self.assertEqual(
            layout.offsets,
            ({2: 0, 0: 0, 1: 2048}, {2: 0, 0: 0, 1: 2048}, {1: 0, 0: 256}),
        )
        self.assertEqual(
            layout.distance,
            ({2: 1, 0: 1, 1: 2}, {2: 1, 0: 1, 1: 2}, {1: 2, 0: 2}),
        )
        self.assertEqual(
            (layout.dense_recv_bytes, layout.expert_recv_bytes), (8192, 4096)
        )
        # dense: signal 0 | credit_intra 256 | credit_inter 512 (2 nodes) | intra_arrive 768 (16 x 3 counters) |
        # gather arrive 1280 | gather intra 1536 (16 x 3 counters, ends at 1920); expert: signal 0 |
        # credit_expert 256 (2 ranks) | gather arrive 512 | gather intra 768 (16 x 2, ends at 1024)
        self.assertEqual(layout.credit, (256, 512, 256))
        self.assertEqual(layout.intra_arrive, 768)
        self.assertEqual(
            (layout.gather_dense, layout.gather_expert),
            ((1280, 1536), (512, 768)),
        )
        self.assertEqual(
            (layout.dense_host_bytes, layout.expert_host_bytes), (4096, 4096)
        )

    def test_bucket_table(self):
        # (kind, send, signal, shard numel, intra arrive, recv 0, recv 1, depth 0, depth 1)
        self.assertEqual(
            self.layout().bucket_table(),
            [
                (DENSE, 0, 0, 256, 768, 0, 4096, 1, 1),
                (DENSE, 4096, 8, 128, 896, 2048, 6144, 2, 2),
                (DENSE, 6144, 16, 384, 1024, 0, 4096, 1, 1),
                (EXPERT, 0, 0, 64, 0, 256, 0, 2, -1),
                (EXPERT, 512, 8, 32, 0, 0, 0, 2, -1),
            ],
        )

    def test_one_rank_per_node_and_no_expert_bucket(self):
        # ranks 0 and 2 with 2 GPUs per node: one member on each node, so no intra-node segment (depth -1)
        self.world = FakeGroup([0, 2], rank=0, group_id=1, ranks_per_node=1)
        self.dense_layout = gcs._GroupLayout(self.world, "sharding")
        layout = self.layout(
            dense_numels=[512], expert_numels=[], order=([0], [])
        )
        # the whole dense pool goes to the inter-node segment
        self.assertEqual(layout.segment_bytes, [0, 8192, 0])
        self.assertEqual(
            layout.bucket_table(), [(DENSE, 0, 0, 256, 768, 0, 0, -1, 1)]
        )
        self.assertEqual(layout.credit[2], 0)
        self.assertEqual(layout.expert_host_bytes, 4096)

    def test_invalid_layouts_raise(self):
        cases = (
            ("order not a permutation", {"order": ([0, 0, 1], [1, 0])}, "not a permutation"),
            ("expert order not a permutation", {"order": ([2, 0, 1], [0])}, "not a permutation"),
            ("dense size not a multiple of the group size", {"dense_numels": [1026, 512, 1536]},
             "multiple of 4 x the sharding group size"),
            ("dense shard not a multiple of 4", {"dense_numels": [1032, 512, 1536]},
             "multiple of 4 x the sharding group size"),
            ("expert size not divisible", {"expert_numels": [129, 64]}, "not divisible by the moe_sharding group"),
        )  # fmt: skip
        for case, overrides, message in cases:
            with (
                self.subTest(case),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                self.layout(**overrides)

    def test_layout_differing_between_ranks_raises(self):
        with (
            mock.patch.object(
                gcs, "_all_ranks_same", lambda checks, world: False
            ),
            self.assertRaisesRegex(
                RuntimeError, "recv pool layout .* differs between ranks"
            ),
        ):
            self.layout()

    def test_auto_sized_recv_pools(self):
        # largest dense bucket 1536 x 4 B, largest expert bucket 128 x 4 B: both pools auto 1 GiB; the dense pool split
        # 1:1 like the equal lower bounds
        config = SimpleNamespace(gin_overlap_dense_recv_pool_gb=0.0, gin_overlap_expert_recv_pool_gb=0.0)  # fmt: skip
        layout = gcs._Layout(self.dense_layout, self.expert_layout, DENSE_NUMELS, EXPERT_NUMELS, RELEASE_ORDER, config)  # fmt: skip
        self.assertEqual(layout.segment_bytes, [2**29, 2**29, 2**30])
        self.assertEqual(layout.auto_sized, (True, True, True))
        with mock.patch.object(gcs, "logger") as logger:
            layout.log("windows text")
        message = logger.info.call_args[0][0]
        self.assertIn("dense intra-node segment 0.500 GiB (auto, lower bound", message)  # fmt: skip
        self.assertIn("expert segment 1.000 GiB (auto, lower bound", message)

    def test_log_on_rank_0_only(self):
        layout = self.layout()
        with mock.patch.object(gcs, "logger") as logger:
            layout.log("windows text")
        message = logger.info.call_args[0][0]
        self.assertIn(
            "3 dense buckets (sharding group 2 nodes x 2), 2 expert buckets",
            message,
        )
        self.assertIn("median slot 0.0 MiB, holds 2", message)
        self.assertIn("expert segment 0.000 GiB (set, lower bound", message)
        # dense: every bucket (d = 1, 1, 2 < 3 buckets); expert: none (d = 2 = number of buckets)
        self.assertIn(
            "intra-node 3 / inter-node 3 / expert 0; windows windows text",
            message,
        )
        with (
            mock.patch.object(gcs, "logger") as logger,
            mock.patch.object(gcs.dist, "get_rank", lambda: 1),
        ):
            layout.log("windows text")
        logger.info.assert_not_called()


class TestGroupWindows(LayoutTestCase):
    def test_windows_are_allocated_registered_and_checked(self):
        windows = gcs._GroupWindows(self.dense_layout, 12288, 100, 4000)
        self.assertEqual(
            (windows.send._numel(), windows.recv._numel()), (3072, 1024)
        )
        self.assertEqual(
            self.world.process_group.registered, [windows.send, windows.recv]
        )
        self.assertEqual(
            (windows.handles(), windows.host_bytes, windows.gin_contexts),
            ((7, 7, 7), 4096, 8),
        )

    def test_unusable_windows_raise(self):
        process_group = self.world.process_group
        with self.subTest("no window API"):
            process_group.window_handle = 0
            with self.assertRaisesRegex(RuntimeError, "returned no window"):
                gcs._GroupWindows(self.dense_layout, 4096, 4096, 4096)
            process_group.window_handle = 7
        for key, value in (
            ("lsa_size", 4),
            ("lsa_rank", 0),
            ("gin_context_count", 0),
        ):
            with self.subTest(key):
                saved = process_group.dev_comm[key]
                process_group.dev_comm[key] = value
                with self.assertRaisesRegex(
                    RuntimeError,
                    "device communicator of the sharding group is unusable",
                ):
                    gcs._GroupWindows(self.dense_layout, 4096, 4096, 4096)
                process_group.dev_comm[key] = saved


def send_window(numel):
    return SimpleNamespace(send=paddle.zeros([numel], dtype="float32"))


class TestAliasGradStorage(unittest.TestCase):
    def test_1d_bucket_moves_into_its_send_range(self):
        window = send_window(64)
        params = [param("w", 6), param("b", 2)]
        b = bucket(params, None, grad_storage=paddle.arange(8, dtype="float32"))
        pointer = gcs._alias_1d(b, window, 128)
        self.assertEqual(pointer, window.send.data_ptr() + 128)
        self.assertEqual(b.grad_storage.data_ptr(), pointer)
        self.assertEqual(window.send.numpy()[32:40].tolist(), list(range(8)))
        for p in params:
            grad_view = b._sharding_param_grad_view[p.name]
            self.assertIs(grad_view._grad_buffer, b.grad_storage)
            self.assertEqual(grad_view.relinked, 1)
        self.assertEqual(b.recorded_addr, 1)

    def two_d_bucket(self, base, begin, end, param_ranges):
        """A 2D bucket whose grad_storage is base[begin:end]; parameter i has main_grad base[param_ranges[i]]."""
        params = []
        for i, (param_begin, param_end, shape) in enumerate(param_ranges):
            p = param(f"w{i}", param_end - param_begin, shape)
            p.main_grad = base._slice(param_begin, param_end)
            params.append(p)
        return bucket(
            params,
            None,
            act=HOOK_ACTION.REDUCE,
            grad_storage=base._slice(begin, end),
        )

    def test_2d_bucket_is_padded_and_main_grad_resliced(self):
        for padded_numel in (6, 8):
            with self.subTest(padded_numel=padded_numel):
                window = send_window(32)
                b = self.two_d_bucket(
                    paddle.arange(6, dtype="float32"),
                    0,
                    6,
                    [(0, 4, [2, 2]), (4, 6, [2, 1])],
                )
                pointer = gcs._alias_2d(b, window, 64, padded_numel)
                self.assertEqual(pointer, window.send.data_ptr() + 64)
                self.assertEqual(b.grad_storage._numel(), padded_numel)
                expected = list(range(6)) + [0] * (padded_numel - 6)
                self.assertEqual(
                    window.send.numpy()[16 : 16 + padded_numel].tolist(),
                    expected,
                )
                first, second = b._params
                self.assertEqual(
                    (first.main_grad.data_ptr(), list(first.main_grad.shape)),
                    (pointer, [2, 2]),
                )
                self.assertEqual(
                    (second.main_grad.data_ptr(), list(second.main_grad.shape)),
                    (pointer + 16, [2, 1]),
                )
                self.assertEqual(b.recorded_addr, 1)

    def test_invalid_2d_buckets_raise(self):
        base = paddle.arange(10, dtype="float32")
        cases = (
            ("more elements than the padded size", self.two_d_bucket(base, 0, 6, [(0, 6, [6])]), 4,
             "6 elements, more than its padded size 4"),
            ("main_grad before the bucket", self.two_d_bucket(base, 2, 8, [(0, 2, [2])]), 8,
             "starts at element -2, outside"),
            ("main_grad after the bucket", self.two_d_bucket(base, 0, 6, [(6, 8, [2])]), 8,
             "starts at element 6, outside"),
        )  # fmt: skip
        for case, b, padded_numel, message in cases:
            with (
                self.subTest(case),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                gcs._alias_2d(b, send_window(16), 0, padded_numel)
        for case, main_grad in (
            ("no main_grad", None),
            ("wrong size", paddle.zeros([3])),
        ):
            with self.subTest(case):
                b = self.two_d_bucket(base, 0, 6, [(0, 2, [2])])
                b._params[0].main_grad = main_grad
                with self.assertRaisesRegex(
                    RuntimeError,
                    "main_grad of w0 is missing or does not have 2 elements",
                ):
                    gcs._alias_2d(b, send_window(16), 0, 8)


class TestGatherTable(LayoutTestCase):
    def two_d_buckets(self, dense_padded=1536, expert_padded=64):
        """Dense bucket 2 owned by group rank 3 and expert bucket 1 owned by group rank 1, both Muon 2D buckets."""
        dense_2d = bucket(
            [param("dense_2d")],
            self.world,
            act=HOOK_ACTION.REDUCE,
            _gin_owner_bucket=(1500, dense_padded, 3),
        )
        expert_2d = bucket(
            [expert_param("expert_2d")],
            self.moe_group,
            act=HOOK_ACTION.REDUCE,
            _gin_owner_bucket=(60, expert_padded, 1),
        )
        return [dense_2d], [expert_2d], {id(dense_2d): 2}, {id(expert_2d): 1}

    def test_rows(self):
        # (unified index, owner, gather arrive + 8 x index, gather intra + 128 x index)
        self.assertEqual(
            gcs._gather_table(self.layout(), *self.two_d_buckets()),
            [(2, 3, 1280 + 16, 1536 + 256), (4, 1, 512 + 8, 768 + 128)],
        )

    def test_dense_only(self):
        layout = self.layout(expert_numels=[], order=([2, 0, 1], []))
        dense_2d, _, dense_index, _ = self.two_d_buckets()
        self.assertEqual(
            gcs._gather_table(layout, dense_2d, [], dense_index, {}),
            [(2, 3, 1296, 1792)],
        )

    def test_invalid_tables_raise(self):
        layout = self.layout()
        for case, buckets, message in (
            ("dense padded size", self.two_d_buckets(dense_padded=1504), "2D dense bucket 2 has padded size 1504, the layout has 1536"),
            ("expert padded size", self.two_d_buckets(expert_padded=60), "2D expert bucket 1 has padded size 60, the layout has 64"),
        ):  # fmt: skip
            with (
                self.subTest(case),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                gcs._gather_table(layout, *buckets)
        with (
            mock.patch.object(
                gcs, "_all_ranks_same", lambda checks, world: False
            ),
            self.assertRaisesRegex(
                RuntimeError, "2D-bucket table .* differ between ranks"
            ),
        ):
            gcs._gather_table(layout, *self.two_d_buckets())


def muon_opt(
    world,
    moe_group,
    buckets_1d,
    buckets_2d,
    optimizer=MuonShardingOptimizer,
    **attrs,
):
    """A MuonShardingOptimizer or MixedShardingOptimizer (without its __init__, which needs a live fleet) holding
    the attributes _select_1d_2d reads. A real instance of the class, because build_grad_comm_scheduler only takes
    the optimizers it knows."""
    fields = {
        "_use_fuse_gradients": True,
        "use_group_call_opt": False,
        "_sync_only_buffer_ids": set(),
        "_comm_buffer_list": buckets_1d,
        "comm_buffer_2d": buckets_2d,
        # both classes delegate an unknown attribute to the inner optimizer
        "_inner_opt": SimpleNamespace(),
    }
    fields.update(attrs)
    opt = optimizer.__new__(optimizer)
    for name, value in fields.items():
        setattr(opt, name, value)
    return opt


class TestSelectBuckets(unittest.TestCase):
    def setUp(self):
        self.world = FakeGroup([0, 1, 2, 3], group_id=1)
        self.moe_group = FakeGroup([2, 3], group_id=2)

    def test_v2_takes_reduce_scatter_buckets(self):
        dense, expert = (
            bucket([param("d")], self.world),
            bucket([expert_param("e")], self.moe_group),
        )
        opt = SimpleNamespace(_comm_buffer_list=[dense, expert])
        self.assertEqual(
            gcs._select_v2(opt, self.world, self.moe_group, "V2"),
            ([expert], [dense], [], []),
        )
        reduce_bucket = bucket([param("r")], self.world, act=HOOK_ACTION.REDUCE)
        opt = SimpleNamespace(_comm_buffer_list=[reduce_bucket])
        with self.assertRaisesRegex(RuntimeError, "V2.*1/1 buckets"):
            gcs._select_v2(opt, self.world, self.moe_group, "V2")

    def test_1d_2d_pads_2d_buckets_and_finds_their_owner(self):
        dense_1d = bucket([param("d1")], self.world)
        dense_2d = bucket(
            [param("d2")],
            self.world,
            act=HOOK_ACTION.REDUCE,
            grad_storage=paddle.zeros([1500]),
            _dst=3,
        )
        expert_2d = bucket(
            [expert_param("e2")],
            self.moe_group,
            act=HOOK_ACTION.REDUCE,
            grad_storage=paddle.zeros([63]),
            _dst=2,
        )
        opt = muon_opt(
            self.world, self.moe_group, [dense_1d], [dense_2d, expert_2d]
        )
        selected = gcs._select_1d_2d(opt, self.world, self.moe_group, "Muon")
        self.assertEqual(
            selected,
            ([expert_2d], [dense_1d, dense_2d], [expert_2d], [dense_2d]),
        )
        # padded to a multiple of 4 x group size: 16 for dense, 8 for expert; owner = group rank of _dst
        self.assertEqual(dense_2d._gin_owner_bucket, (1500, 1504, 3))
        self.assertEqual(expert_2d._gin_owner_bucket, (63, 64, 0))
        self.assertFalse(hasattr(dense_1d, "_gin_owner_bucket"))

    def test_1d_2d_unsupported_configurations(self):
        cases = (
            ("no fused buckets", {"_use_fuse_gradients": False}, [], "needs comm_buffer_size_MB > 0"),
            ("group call", {"use_group_call_opt": True}, [], "comm_group_call_opt"),
        )  # fmt: skip
        for case, attrs, buckets_1d, message in cases:
            with self.subTest(case):
                opt = muon_opt(
                    self.world, self.moe_group, buckets_1d, [], **attrs
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    gcs._select_1d_2d(opt, self.world, self.moe_group, "Muon")
        opt = muon_opt(
            self.world, self.moe_group, [], [bucket([param("r")], self.world)]
        )
        with self.assertRaisesRegex(RuntimeError, "(?s)2D buckets.*act="):
            gcs._select_1d_2d(opt, self.world, self.moe_group, "Muon")
        # a sync-only bucket is not taken over, and does not fail the build
        shared = bucket([param("s")], self.world)
        opt = muon_opt(
            self.world,
            self.moe_group,
            [shared, bucket([param("fine")], self.world)],
            [],
            _sync_only_buffer_ids={id(shared)},
        )
        expert, dense, expert_2d, dense_2d = gcs._select_1d_2d(
            opt, self.world, self.moe_group, "Muon"
        )
        self.assertEqual([b._params[0].name for b in dense], ["fine"])


class RecordingScheduler:
    """Stand-in for core.gin_overlap.GinGradCommScheduler: keeps the constructor arguments and the calls."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.ready = []
        self.finish_timeouts = []

    def bucket_ready(self, bucket_index, scale):
        self.ready.append((bucket_index, scale))

    def finish_step(self, timeout):
        self.finish_timeouts.append(timeout)


def sharding_config(**overrides):
    fields = {
        "gin_overlap_timeout": 30.0,
        "accumulate_steps": 1,
        "gin_overlap_dense_recv_pool_gb": POOL_CONFIG.gin_overlap_dense_recv_pool_gb,
        "gin_overlap_expert_recv_pool_gb": POOL_CONFIG.gin_overlap_expert_recv_pool_gb,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class FakeHcg:
    def __init__(self, world, moe_group=None, pp_degree=1):
        self.world, self.pp_degree = world, pp_degree
        if moe_group is not None:
            self.get_moe_sharding_parallel_group = lambda: moe_group

    def get_data_parallel_world_size(self):
        return 1

    def get_pipe_parallel_world_size(self):
        return self.pp_degree

    def get_sharding_parallel_group(self):
        return self.world


def numbered_grad_storage(numel):
    return paddle.arange(numel, dtype="float32")


class BuildTestCase(LayoutTestCase):
    def setUp(self):
        super().setUp()
        self.config = sharding_config()
        for patcher in (
            mock.patch.object(gcs, "GinGradCommScheduler", RecordingScheduler),
            mock.patch.object(gcs, "_sharding_configs", lambda: self.config),
            mock.patch.object(gcs, "logger"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def v2_opt(
        self, dense_numels=DENSE_NUMELS, expert_numels=EXPERT_NUMELS, hcg=None
    ):
        """A DygraphShardingOptimizerV2 (without its __init__) whose buckets give RELEASE_ORDER: dense first
        parameter positions 2 / 0 / 4, expert 1 / 3."""
        dense = [
            bucket(
                [param(f"dense{i}")],
                self.world,
                grad_storage=numbered_grad_storage(n),
            )
            for i, n in enumerate(dense_numels)
        ]
        expert = [
            bucket(
                [expert_param(f"expert{i}")],
                self.moe_group,
                grad_storage=numbered_grad_storage(n),
            )
            for i, n in enumerate(expert_numels)
        ]
        position_order = ["dense1", "expert0", "dense0", "expert1", "dense2"]
        params = {b._params[0].name: b._params[0] for b in dense + expert}
        opt = DygraphShardingOptimizerV2.__new__(DygraphShardingOptimizerV2)
        opt._inner_opt = SimpleNamespace()
        opt._hcg = hcg or FakeHcg(self.world, self.moe_group)
        opt._comm_buffer_list = dense + expert
        opt._parameter_list = [
            params[name] for name in position_order if name in params
        ]
        opt._create_backward_hook = lambda b, p, use_comm: (
            "hook",
            b,
            p.name,
            use_comm,
        )
        return opt

    def opt_1d_2d(self, optimizer=MuonShardingOptimizer):
        """A MuonShardingOptimizer or MixedShardingOptimizer (without its __init__) with one 1D bucket and two 2D
        buckets, and the buckets themselves: fresh ones per call, because a build moves their grad_storage into the
        send window and registers their hooks."""
        dense_1d = bucket(
            [param("dense_1d")],
            self.world,
            grad_storage=numbered_grad_storage(1024),
        )
        dense_2d_param = param("dense_2d", 1500, [30, 50])
        dense_2d = bucket(
            [dense_2d_param],
            self.world,
            act=HOOK_ACTION.REDUCE,
            grad_storage=numbered_grad_storage(1500),
            _dst=3,
        )
        dense_2d_param.main_grad = dense_2d.grad_storage
        expert_2d_param = expert_param("expert_2d", 63, [7, 9])
        expert_2d = bucket(
            [expert_2d_param],
            self.moe_group,
            act=HOOK_ACTION.REDUCE,
            grad_storage=numbered_grad_storage(63),
            _dst=1,
        )
        expert_2d_param.main_grad = expert_2d.grad_storage
        opt = muon_opt(
            self.world,
            self.moe_group,
            [dense_1d],
            [dense_2d, expert_2d],
            optimizer=optimizer,
        )
        opt._hcg = FakeHcg(self.world, self.moe_group)
        opt._parameter_list = [
            dense_2d_param,
            dense_1d._params[0],
            expert_2d_param,
        ]
        opt._create_backward_hook = lambda b, p: ("hook", b, p.name)
        return opt, dense_1d, dense_2d, expert_2d


class TestBuildGradCommScheduler(BuildTestCase):
    def test_v2_with_dense_and_expert_buckets(self):
        opt = self.v2_opt()
        built = gcs.build_grad_comm_scheduler(opt)
        self.assertIsInstance(built, RecordingScheduler)
        self.assertIs(gcs.scheduler, built)
        self.assertIs(opt._gin_scheduler, built)
        kwargs = built.kwargs
        self.assertIs(kwargs["dense_group"], self.world.process_group)
        self.assertIs(kwargs["expert_group"], self.moe_group.process_group)
        self.assertEqual(
            (kwargs["dense_windows"], kwargs["expert_windows"]),
            ((7, 7, 7), (7, 7, 7)),
        )
        self.assertEqual(kwargs["buckets"], self.layout().bucket_table())
        self.assertEqual(
            (kwargs["dense_order"], kwargs["expert_order"]), ([2, 0, 1], [4, 3])
        )
        self.assertEqual(
            (kwargs["credit_intra_offset"], kwargs["credit_inter_offset"], kwargs["credit_expert_offset"]),
            (256, 512, 256),
        )  # fmt: skip
        self.assertEqual(kwargs["gather"], [])
        self.assertEqual(kwargs["num_window_marks"], gcs.NUM_WINDOW_MARKS)
        self.assertEqual(kwargs["num_tail_marks"], 5 + gcs.TAIL_EXTRA_MARKS)
        _, dense_windows, expert_windows = opt._gin_scheduler_state
        buckets = opt._comm_buffer_list
        self.assertEqual(opt._gin_buckets, buckets)
        sends = [(dense_windows, offset) for offset in (0, 4096, 6144)] + [
            (expert_windows, offset) for offset in (0, 512)
        ]
        for index, (b, (windows, offset)) in enumerate(zip(buckets, sends)):
            with self.subTest(bucket=index):
                # grad_storage moved into the send window with its values
                self.assertEqual(
                    b.grad_storage.data_ptr(), windows.send.data_ptr() + offset
                )
                self.assertEqual(b._gin_grad_ptr, b.grad_storage.data_ptr())
                self.assertEqual(
                    b.grad_storage.numpy().tolist(),
                    list(range(b.grad_storage._numel())),
                )
                self.assertEqual(
                    (b._gin_bucket_index, b._gin_scheduler, b._gin_reported),
                    (index, built, False),
                )
                p = b._params[0]
                self.assertEqual(p.backward_hooks, [("hook", b, p.name, True)])
        info = " ".join(call[0][0] for call in gcs.logger.info.call_args_list)
        self.assertIn(
            "sharding group send 0.000 + recv 0.000 GiB, host 4096 B, GIN contexts 8",
            info,
        )
        self.assertIn("moe_sharding group send", info)

    def test_v2_dense_only_without_expert_parallelism(self):
        opt = self.v2_opt(
            expert_numels=[], hcg=FakeHcg(self.world, pp_degree=2)
        )
        self.config.accumulate_steps = 4
        strategy = SimpleNamespace(pipeline_configs={"accumulate_steps": 4})
        with mock.patch.object(
            paddle.distributed.fleet.fleet,
            "_user_defined_strategy",
            strategy,
            create=True,
        ):
            built = gcs.build_grad_comm_scheduler(opt)
        self.assertIsNone(built.kwargs["expert_group"])
        self.assertEqual(built.kwargs["expert_windows"], (0, 0, 0))
        self.assertEqual(built.kwargs["expert_order"], [])
        self.assertIsNone(opt._gin_scheduler_state[2])

    def test_muon_with_2d_buckets(self):
        opt, dense_1d, dense_2d, expert_2d = self.opt_1d_2d()
        dense_2d_param = dense_2d._params[0]
        built = gcs.build_grad_comm_scheduler(opt)
        # dense host window with 2 buckets: gather arrive 1024, gather intra 1280; expert with 1: 512, 768
        self.assertEqual(
            built.kwargs["gather"],
            [(1, 3, 1024 + 8, 1280 + 128), (2, 1, 512, 768)],
        )
        self.assertEqual(built.kwargs["dense_order"], [0, 1])
        self.assertEqual(
            [row[3] for row in built.kwargs["buckets"]], [256, 376, 32]
        )
        # 2D buckets: padded with zeros in the send window, main_grad re-sliced with the parameter shape
        self.assertEqual(
            dense_2d.grad_storage.numpy().tolist(), [*range(1500), 0, 0, 0, 0]
        )
        self.assertEqual(
            expert_2d.grad_storage.numpy().tolist(), [*range(63), 0]
        )
        self.assertEqual(list(dense_2d_param.main_grad.shape), [30, 50])
        self.assertEqual(
            dense_2d_param.main_grad.data_ptr(),
            dense_2d.grad_storage.data_ptr(),
        )
        self.assertEqual(
            dense_2d_param.backward_hooks, [("hook", dense_2d, "dense_2d")]
        )
        info = " ".join(call[0][0] for call in gcs.logger.info.call_args_list)
        self.assertIn(
            "2 2D buckets gathered to their owners (dense 1 / expert 1)",
            info,
        )

    def test_mixed_builds_like_muon(self):
        """MixedShardingOptimizer goes through the same _select_1d_2d and produces the same layout: the scheduler
        reads the kind of bucket, not the class."""
        muon, *_ = self.opt_1d_2d(MuonShardingOptimizer)
        muon_built = gcs.build_grad_comm_scheduler(muon)
        gcs.scheduler = None
        mixed, *_ = self.opt_1d_2d(MixedShardingOptimizer)
        mixed_built = gcs.build_grad_comm_scheduler(mixed)
        for key in ("gather", "dense_order", "expert_order", "buckets"):
            self.assertEqual(mixed_built.kwargs[key], muon_built.kwargs[key])
        self.assertIs(mixed._gin_scheduler, mixed_built)

    def test_an_optimizer_the_scheduler_does_not_know_raises(self):
        with self.assertRaisesRegex(
            RuntimeError, "SimpleNamespace is not supported"
        ):
            gcs.build_grad_comm_scheduler(SimpleNamespace())
        self.assertIsNone(gcs.scheduler)

    def test_a_fallback_bucket_keeps_its_own_collective(self):
        """A bucket with a no-hook colour parameter is not taken over: it stays out of _gin_buckets, keeps
        _gin_scheduler None (so FusedCommBuffer._comm_grads / scale_grads run its own reduce_scatter), gets no
        unified bucket index and no backward hook."""
        opt = self.v2_opt()
        shared = bucket(
            [param("shared", color={"color": "dense_weight_no_hook"})],
            self.world,
            grad_storage=numbered_grad_storage(512),
        )
        opt._comm_buffer_list = [*opt._comm_buffer_list, shared]
        opt._parameter_list = [*opt._parameter_list, shared._params[0]]
        built = gcs.build_grad_comm_scheduler(opt)
        self.assertNotIn(id(shared), {id(b) for b in opt._gin_buckets})
        self.assertEqual(len(opt._gin_buckets), 5)
        self.assertIsNone(shared._gin_scheduler)
        self.assertFalse(hasattr(shared, "_gin_bucket_index"))
        self.assertEqual(shared._params[0].backward_hooks, [])
        for b in opt._gin_buckets:
            self.assertIs(b._gin_scheduler, built)
        # this rank is not the first of its sharding group, so it logs no fallback warning
        # (TestSplitBuckets covers both ranks)
        self.assertFalse(gcs.logger.warning.called)

    def test_real_buckets_keep_their_main_grad_in_the_send_window(self):
        buckets = [real_bucket(self.world) for _ in range(2)]
        for value, b in enumerate(buckets, start=1):
            b.grad_storage.fill_(float(value))
        opt = self.v2_opt(dense_numels=[], expert_numels=[])
        opt._comm_buffer_list = buckets
        opt._parameter_list = [p for b in buckets for p in b._params]
        gcs.build_grad_comm_scheduler(opt)
        send = opt._gin_scheduler_state[1].send
        send_begin, send_end = (
            send.data_ptr(),
            send.data_ptr() + send._numel() * 4,
        )
        for value, b in enumerate(buckets, start=1):
            with self.subTest(bucket=b._gin_bucket_index):
                self.assertTrue(
                    send_begin <= b.grad_storage.data_ptr() < send_end
                )
                self.assertTrue(
                    bool(paddle.all(b.grad_storage == float(value)))
                )
                for p in b._params:
                    # main_grad is re-linked into the new grad_storage, with the parameter's shape
                    offset = p.main_grad.data_ptr() - b.grad_storage.data_ptr()
                    self.assertTrue(0 <= offset < b.grad_storage._numel() * 4)
                    self.assertEqual(p.main_grad.shape, p.shape)
                    self.assertEqual(
                        b._grads_to_addr[p.name], p.main_grad.data_ptr()
                    )

    def test_unsupported_builds_raise(self):
        cases = (
            ("no gin_overlap in this build", {"_LIB": None}, {}, RuntimeError, "built without gin_overlap.*CUDA >= 13.0"),
            ("already built", {"scheduler": object()}, {}, RuntimeError, "already built in this process"),
            ("timeout 0", {}, {"gin_overlap_timeout": 0}, ValueError, "gin_overlap_timeout must be positive, got 0"),
            ("negative timeout", {}, {"gin_overlap_timeout": -1.0}, ValueError, "must be positive"),
        )  # fmt: skip
        for case, module_overrides, config_overrides, error, message in cases:
            with self.subTest(case):
                self.config = sharding_config(**config_overrides)
                patch_module = (
                    mock.patch.multiple(gcs, **module_overrides)
                    if module_overrides
                    else contextlib.nullcontext()
                )
                with patch_module, self.assertRaisesRegex(error, message):
                    gcs.build_grad_comm_scheduler(self.v2_opt())

    def test_pipeline_accumulate_steps_must_match(self):
        self.config.accumulate_steps = 2
        strategy = SimpleNamespace(pipeline_configs={"accumulate_steps": 4})
        opt = self.v2_opt(hcg=FakeHcg(self.world, self.moe_group, pp_degree=2))
        with (
            mock.patch.object(
                paddle.distributed.fleet.fleet,
                "_user_defined_strategy",
                strategy,
                create=True,
            ),
            self.assertRaisesRegex(
                ValueError,
                "pp_degree=2 .*sharding_configs.accumulate_steps=2 .*pipeline_configs.accumulate_steps=4",
            ),
        ):
            gcs.build_grad_comm_scheduler(opt)

    def test_no_dense_bucket_raises(self):
        with self.assertRaisesRegex(RuntimeError, "there is no dense bucket"):
            gcs.build_grad_comm_scheduler(self.v2_opt(dense_numels=[]))

    def test_unsupported_optimizer_configuration_raises(self):
        opt = self.v2_opt()
        opt.comm_overlap = True
        with self.assertRaisesRegex(
            RuntimeError, "DygraphShardingOptimizerV2.*comm_overlap"
        ):
            gcs.build_grad_comm_scheduler(opt)
        self.assertIsNone(gcs.scheduler)


class TestShardingConfigs(unittest.TestCase):
    def test_read_from_the_fleet_strategy(self):
        config = sharding_config()
        strategy = SimpleNamespace(hybrid_configs={"sharding_configs": config})
        with mock.patch.object(
            paddle.distributed.fleet.fleet,
            "_user_defined_strategy",
            strategy,
            create=True,
        ):
            self.assertIs(gcs._sharding_configs(), config)


def owned_bucket(nranks=4, **attrs):
    """A bucket built into a RecordingScheduler as bucket 3."""
    grad_storage = paddle.ones([8], dtype="float32")
    fields = {
        "grad_storage": grad_storage,
        "_gin_grad_ptr": grad_storage.data_ptr(),
        "_gin_bucket_index": 3,
        "_gin_scheduler": RecordingScheduler(),
        "_gin_reported": False,
        "_comm_group": FakeGroup(range(nranks)),
    }
    fields.update(attrs)
    return SimpleNamespace(**fields)


class TestStepCalls(unittest.TestCase):
    def test_bucket_ready_scale_follows_the_reduce_op(self):
        for reduce_op, scale in (
            (gcs.dist.ReduceOp.AVG, 0.25),
            (gcs.dist.ReduceOp.SUM, 1.0),
        ):
            with self.subTest(reduce_op=reduce_op):
                b = owned_bucket()
                gcs.bucket_ready(b, reduce_op)
                gcs.bucket_ready(b, reduce_op)  # reported once per step
                self.assertEqual(b._gin_scheduler.ready, [(3, scale)])
                self.assertTrue(b._gin_reported)

    def test_bucket_ready_raises_when_grad_storage_moved(self):
        for case, grad_storage in (
            ("replaced", paddle.ones([8])),
            ("released", None),
        ):
            with self.subTest(case):
                b = owned_bucket()
                b.grad_storage = grad_storage
                with self.assertRaisesRegex(
                    RuntimeError,
                    "grad_storage of bucket 3 is no longer its send window range",
                ):
                    gcs.bucket_ready(b, gcs.dist.ReduceOp.SUM)

    def test_finish_step_waits_with_the_configured_timeout(self):
        opt = SimpleNamespace(_gin_scheduler=RecordingScheduler())
        with mock.patch.object(
            gcs,
            "_sharding_configs",
            lambda: sharding_config(gin_overlap_timeout=12),
        ):
            gcs.finish_step(opt)
        self.assertEqual(opt._gin_scheduler.finish_timeouts, [12.0])
        self.assertIsInstance(opt._gin_scheduler.finish_timeouts[0], float)


def param_storage_bucket(
    name,
    group,
    numel,
    dtype="float32",
    act=HOOK_ACTION.REDUCE_SCATTER,
    frozen=False,
):
    b = bucket([param(name)], group, act=act, frozen=frozen)
    b.param_storage = paddle.arange(numel).astype(dtype)
    return b


class TestFusedAllGather(unittest.TestCase):
    """The fake all_gather puts this rank's shards + 1000 x r in row r, so after the gather the part of rank r of
    each bucket must be this rank's part + 1000 x r."""

    def setUp(self):
        # 2 ranks, this process is rank 1
        self.group = FakeGroup([0, 1], rank=1, group_id=1)

    def assert_gathered(self, b, original):
        part = original.shape[0] // 2
        own = original[part:]
        for src_rank in range(2):
            expected = (own + 1000.0 * src_rank).tolist()
            self.assertEqual(
                b.param_storage.numpy()[
                    src_rank * part : (src_rank + 1) * part
                ].tolist(),
                expected,
            )

    def test_batches_by_size_group_and_dtype(self):
        # fp32: a big bucket is 100000 elements = 400 KB, so two fit in a 1 MiB batch; the oversized one (2.4 MB)
        # gets a batch of its own
        big = [
            param_storage_bucket(f"big{i}", self.group, 100000)
            for i in range(3)
        ]
        oversized = param_storage_bucket("oversized", self.group, 600000)
        half = param_storage_bucket("half", self.group, 8, dtype="float16")
        other_group = FakeGroup([0, 1], rank=1, group_id=2)
        elsewhere = param_storage_bucket("elsewhere", other_group, 6)
        skipped = [
            param_storage_bucket("frozen", self.group, 4, frozen=True),
            param_storage_bucket(
                "reduce", self.group, 4, act=HOOK_ACTION.REDUCE
            ),
            param_storage_bucket("one_rank", FakeGroup([0], group_id=3), 4),
        ]
        buckets = [*big, oversized, half, elsewhere, *skipped]
        originals = {
            b._params[0].name: b.param_storage.numpy().copy() for b in buckets
        }
        tensor_fusion_helper.fused_all_gather(buckets, fuse_mb=1)
        # numel of each send (this rank's half of each bucket): [big0 + big1], [big2], [oversized] in fp32, [half]
        # in fp16, [elsewhere] in the other group
        self.assertEqual(
            self.group.process_group.gathered_numels, [100000, 50000, 300000, 4]
        )
        self.assertEqual(other_group.process_group.gathered_numels, [3])
        for b in [*big, oversized, half, elsewhere]:
            with self.subTest(b._params[0].name):
                self.assert_gathered(b, originals[b._params[0].name])
        for b in skipped:
            self.assertEqual(
                b.param_storage.numpy().tolist(),
                originals[b._params[0].name].tolist(),
            )

    def test_empty_part_is_skipped(self):
        empty = param_storage_bucket("empty", self.group, 0)
        filled = param_storage_bucket("filled", self.group, 4)
        tensor_fusion_helper._all_gather_chunk(
            [empty, filled], self.group, 2, 1
        )
        self.assertEqual(
            filled.param_storage.numpy().tolist(), [2, 3, 1002, 1003]
        )

    def test_size_not_a_multiple_of_the_group_raises(self):
        odd = param_storage_bucket("odd", self.group, 5)
        with self.assertRaisesRegex(
            RuntimeError,
            "bucket of odd has 5 elements, not a multiple of the group size 2",
        ):
            tensor_fusion_helper.fused_all_gather([odd], fuse_mb=1)


class CallLog(list):
    """Shared list of the calls of several fakes, in order."""

    def recorder(self, name):
        return lambda *args, **kwargs: self.append(name)


def logged_buffer(calls, name):
    return SimpleNamespace(
        _comm_grads=calls.recorder(f"{name}.comm"),
        scale_grads=calls.recorder(f"{name}.scale"),
        sync_params=calls.recorder(f"{name}.sync"),
    )


def real_bucket(group, frozen=False, **kwargs):
    """A real REDUCE_SCATTER FusedCommBuffer of a Linear(4, 4) over a fake group (building it communicates nothing);
    frozen = all parameters stop_gradient after the bucket is built.
    __current_device_type__ is pinned to "gpu" because the alignment table of tensor_fusion_helper has no CPU entry
    (CPU-only builds)."""
    linear = paddle.nn.Linear(4, 4)
    for p in linear.parameters():
        p.main_grad = None
    with mock.patch.object(
        tensor_fusion_helper, "__current_device_type__", "gpu"
    ):
        b = FusedCommBuffer(
            id=0,
            params=linear.parameters(),
            comm_group=group,
            act=HOOK_ACTION.REDUCE_SCATTER,
            use_main_grad=True,
            **kwargs,
        )
    for p in linear.parameters():
        p.stop_gradient = frozen
    return b


class TestFusedCommBufferGinBranch(unittest.TestCase):
    def gin_bucket(self, frozen=False, **kwargs):
        b = real_bucket(FakeGroup(range(4)), frozen, **kwargs)
        self.assertIsNone(b._gin_scheduler)
        self.assertFalse(b._gin_reported)
        b.grad_storage.fill_(1.0)
        b._gin_scheduler, b._gin_bucket_index, b._gin_grad_ptr = (
            RecordingScheduler(),
            3,
            b.grad_storage.data_ptr(),
        )
        return b

    def test_comm_grads_reports_to_the_scheduler(self):
        cases = (
            ("AVG", {"use_reduce_avg": True}, 0.25),
            ("SUM", {}, 1.0),
            # the gin branch comes before the in-place pre-scaling of the collective path
            ("SUM, scale before comm", {"scale_after_comm": False}, 1.0),
        )  # fmt: skip
        for case, kwargs, scale in cases:
            with self.subTest(case):
                b = self.gin_bucket(**kwargs)
                b._comm_grads()
                b._comm_grads()
                self.assertEqual(b._gin_scheduler.ready, [(3, scale)])
                self.assertIsNone(b._task)
                self.assertTrue(bool(paddle.all(b.grad_storage == 1.0)))

    def test_comm_grads_of_a_frozen_bucket_does_nothing(self):
        b = self.gin_bucket(frozen=True)
        b._comm_grads()
        self.assertEqual(b._gin_scheduler.ready, [])

    def test_scale_grads_after_the_step(self):
        cases = (
            ("SUM, scale after comm", {}, False, 0.25),
            ("AVG", {"use_reduce_avg": True}, False, 1.0),
            ("SUM, scaled before comm", {"scale_after_comm": False}, False, 1.0),
            ("frozen", {}, True, 1.0),
        )  # fmt: skip
        for case, kwargs, frozen, value in cases:
            with self.subTest(case):
                b = self.gin_bucket(frozen, **kwargs)
                b._gin_reported, b._params_checked_in = True, 2
                b.scale_grads()
                self.assertTrue(bool(paddle.all(b.grad_storage == value)))
                self.assertFalse(b._gin_reported)
                self.assertEqual(b._params_checked_in, 0)


class TestFusedAllGatherMatchesSyncParams(unittest.TestCase):
    def test_same_param_storage_as_sync_params(self):
        """Real buckets of a 4-rank group (this process is rank 2): the fused all_gather leaves every param_storage
        exactly as sync_params of each bucket does."""

        def buckets():
            group = FakeGroup(range(4), rank=2)
            pair = [real_bucket(group) for _ in range(2)]
            for offset, b in enumerate(pair):
                numel = b.param_storage._numel()
                b.param_storage.copy_(paddle.arange(numel, dtype="float32") + 100.0 * offset, False)  # fmt: skip
            return pair

        per_bucket, fused = buckets(), buckets()
        for b in per_bucket:
            b.sync_params()
        tensor_fusion_helper.fused_all_gather(fused, fuse_mb=1)
        for b_sync, b_fused in zip(per_bucket, fused):
            self.assertEqual(b_sync.param_storage.numpy().tolist(), b_fused.param_storage.numpy().tolist())  # fmt: skip


class TestShardingOptimizerGinBranches(unittest.TestCase):
    """reduce_gradients / _sharding_sync_parameters / rebuild of the sharding optimizers, called on fake optimizers;
    finish_step and fused_all_gather are recorded. param_allgather_fuse_mb > 0 fuses the parameter all_gather with or
    without gin_overlap; 0 keeps sync_params per bucket."""

    def record_fused_all_gather(self, buckets, fuse_mb):
        self.calls.append(("fused_all_gather", len(buckets), fuse_mb))

    def setUp(self):
        self.calls = CallLog()
        for patcher in (
            mock.patch.object(gcs, "finish_step", lambda opt: self.calls.append("finish_step")),
            mock.patch.object(dygraph_sharding_optimizer, "fused_all_gather", self.record_fused_all_gather),
            mock.patch.object(muon_sharding_optimizer, "fused_all_gather", self.record_fused_all_gather),
            mock.patch.object(mixed_sharding_optimizer, "fused_all_gather", self.record_fused_all_gather),
        ):  # fmt: skip
            patcher.start()
            self.addCleanup(patcher.stop)

    def v2(self, gin, comm_overlap=False, fuse_mb=0.0):
        return SimpleNamespace(
            _comm_buffer_list=[logged_buffer(self.calls, "b0"), logged_buffer(self.calls, "b1")],
            sd_release_grads=False,
            _enable_timer=False,
            _all_gather_overlap_forward=False,
            comm_overlap=comm_overlap,
            _gin_scheduler=RecordingScheduler() if gin else None,
            _param_allgather_fuse_mb=fuse_mb,
        )  # fmt: skip

    def muon_or_mixed(self, gin, fuse_mb=0.0):
        """MuonShardingOptimizer and MixedShardingOptimizer read the same attributes: 2D buckets, 1D buckets, and
        nothing else on the gin_overlap path, so falling through it raises AttributeError on _use_fuse_gradients."""
        return SimpleNamespace(
            comm_buffer_2d=[logged_buffer(self.calls, "2d")],
            _comm_buffer_list=[logged_buffer(self.calls, "1d")],
            _gin_scheduler=RecordingScheduler() if gin else None,
            _param_allgather_fuse_mb=fuse_mb,
            _hcg=SimpleNamespace(get_sharding_parallel_group=lambda: None),
            _rank2params_2d_by_color={},
            _color_to_group_info={},
            use_group_call_opt=False,
        )

    def test_v2_reduce_gradients(self):
        cases = (
            ("gin", True, False, ["b0.comm", "b1.comm", "finish_step", "b0.scale", "b1.scale"]),
            ("collective", False, False, ["b0.comm", "b0.scale", "b1.comm", "b1.scale"]),
            ("collective with comm_overlap", False, True, ["b0.scale", "b1.scale"]),
        )  # fmt: skip
        for case, gin, comm_overlap, expected in cases:
            with self.subTest(case):
                self.calls.clear()
                DygraphShardingOptimizerV2.reduce_gradients(
                    self.v2(gin, comm_overlap), [], None
                )
                self.assertEqual(self.calls, expected)

    # (case, gin on, param_allgather_fuse_mb, fused): fused whenever the size is positive, with or without gin; the
    # default 0 keeps the per-bucket sync_params of the original code
    SYNC_CASES = (
        ("gin, fuse_mb unset", True, 0.0, False),
        ("gin, fuse_mb 2048", True, 2048.0, True),
        ("no gin, fuse_mb 2048", False, 2048.0, True),
        ("no gin, fuse_mb unset", False, 0.0, False),
    )

    def test_v2_sync_parameters(self):
        for case, gin, fuse_mb, fused in self.SYNC_CASES:
            with self.subTest(case):
                self.calls.clear()
                DygraphShardingOptimizerV2._sharding_sync_parameters(
                    self.v2(gin, fuse_mb=fuse_mb)
                )
                expected = [("fused_all_gather", 2, fuse_mb)] if fused else ["b0.sync", "b1.sync"]  # fmt: skip
                self.assertEqual(self.calls, expected)

    def test_v2_rebuild_is_refused(self):
        with self.assertRaisesRegex(
            RuntimeError, "does not support DygraphShardingOptimizerV2.rebuild"
        ):
            DygraphShardingOptimizerV2.rebuild(self.v2(gin=True))

    # the two optimizers with 1D and 2D buckets: the same gin_overlap branches, read through the same attributes
    OPTIMIZERS_1D_2D = (MuonShardingOptimizer, MixedShardingOptimizer)

    def test_1d_2d_reduce_gradients(self):
        for optimizer in self.OPTIMIZERS_1D_2D:
            with self.subTest(optimizer.__name__):
                self.calls.clear()
                optimizer.reduce_gradients(
                    self.muon_or_mixed(gin=True), [], None
                )
                self.assertEqual(
                    self.calls,
                    [
                        "2d.comm",
                        "1d.comm",
                        "finish_step",
                        "2d.scale",
                        "1d.scale",
                    ],
                )

    def test_1d_2d_sync_parameters(self):
        for optimizer in self.OPTIMIZERS_1D_2D:
            for case, gin, fuse_mb, fused in self.SYNC_CASES:
                with self.subTest(f"{optimizer.__name__}: {case}"):
                    self.calls.clear()
                    optimizer._sharding_sync_parameters(
                        self.muon_or_mixed(gin, fuse_mb=fuse_mb)
                    )
                    expected = [("fused_all_gather", 1, fuse_mb)] if fused else ["1d.sync"]  # fmt: skip
                    self.assertEqual(self.calls, expected)


class OwnerPartitioner:
    """The state the Muon / Mixed 2D partitioners read (see test_muon_2d_partition.py), with
    muon_owner_partition_mb. The partitioners are called unbound, on the optimizer class under test."""

    def __init__(self, comm_buffer_size_MB, owner_partition_mb):
        self.comm_buffer_size_MB = comm_buffer_size_MB
        self._owner_partition_mb = owner_partition_mb
        self._global_rank = 1


class TestOwnerPartitionSize(unittest.TestCase):
    """muon_owner_partition_mb > 0 replaces comm_buffer_size_MB as the MB per owner of the 2D
    partition; unset (None, or the default 0) keeps comm_buffer_size_MB as in the original code. Eight 1 MiB fp32
    parameters (8 MiB) may spread over int(8 / MB per owner) + 1 ranks: 9 with 1 MB (so each of the 8 parameters gets
    its own owner) and 3 with 4 MB. MuonShardingOptimizer and MixedShardingOptimizer have the same two
    partitioners."""

    PARAMS = [
        SimpleNamespace(name=f"w{i}", shape=[512, 512], dtype="float32")
        for i in range(8)
    ]

    def owners(self, optimizer, comm_buffer_size_MB, owner_partition_mb):
        state = OwnerPartitioner(comm_buffer_size_MB, owner_partition_mb)
        legacy = optimizer._partition_2d_parameters(state, self.PARAMS, 16)
        group_key = (None, tuple(range(16)))
        balanced = optimizer._partition_2d_parameters_machine_balanced(
            state,
            {group_key: [(p.name, 512 * 512, p.dtype, 4) for p in self.PARAMS]},
            dict.fromkeys(range(16), "host0"),
        )[group_key]
        return [
            sum(1 for names in mapping.values() if names)
            for mapping in (legacy, balanced)
        ]

    def test_owner_count(self):
        for optimizer in (MuonShardingOptimizer, MixedShardingOptimizer):
            for comm_buffer_size_MB, owner_partition_mb, owner_count in (
                (1, None, 8),
                (1, 0, 8),
                (1, 4, 3),
                (4, None, 3),
                (4, 0, 3),
                (4, 1, 8),
            ):
                with self.subTest(
                    optimizer=optimizer.__name__,
                    comm_buffer_size_MB=comm_buffer_size_MB,
                    owner_partition_mb=owner_partition_mb,
                ):
                    self.assertEqual(
                        self.owners(
                            optimizer, comm_buffer_size_MB, owner_partition_mb
                        ),
                        [owner_count] * 2,
                    )


class TestHybridParallelOptimizerChoice(unittest.TestCase):
    def test_gin_overlap_needs_split_param_or_muon(self):
        hcg = SimpleNamespace(get_sharding_parallel_world_size=lambda: 2)
        sharding_configs = SimpleNamespace(split_param=False, gin_overlap=True)
        strategy = SimpleNamespace(
            hybrid_configs={"sharding_configs": sharding_configs}
        )
        with self.assertRaisesRegex(
            ValueError,
            "gin_overlap supports sharding stage1 .* got DygraphShardingOptimizer$",
        ):
            HybridParallelOptimizer(SimpleNamespace(), hcg, strategy)


if __name__ == "__main__":
    unittest.main()
