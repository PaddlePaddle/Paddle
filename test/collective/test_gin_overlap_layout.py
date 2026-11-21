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
"""CPU tests of the layout helpers of paddle.distributed.communication.gin_overlap.

_send_layout, _recv_pool_sizes and _recv_pool_ring are pure Python; their results become the per-bucket tables of
the C++ release layer (offsets and the predecessor distances that the credit targets are computed from). No GPU,
no process group and no paddle.base.core.gin_overlap are needed: importing the module does not create the engine.
"""

import importlib.util
import unittest

# The module is packaged only into builds with WITH_GIN_OVERLAP (setup.py);
# importing it never needs paddle.base.core.gin_overlap.
_HAVE = (
    importlib.util.find_spec("paddle.distributed.communication.gin_overlap")
    is not None
)
if _HAVE:
    from paddle.distributed.communication.gin_overlap import engine

    GinOverlapPrimitive = engine.GinOverlapPrimitive
KB4 = 4096
_skip = unittest.skipUnless(
    _HAVE, "paddle.distributed.communication.gin_overlap is not packaged"
)


def _ring(order, seg, P):
    return GinOverlapPrimitive._recv_pool_ring(order, seg, P)


def _sizes(lb, dense_gb, moe_gb):
    prim = object.__new__(GinOverlapPrimitive)  # no __init__: no engine
    prim.dense_recv_pool_gb = dense_gb
    prim.moe_recv_pool_gb = moe_gb
    return prim._recv_pool_sizes(lb, ["d0", "d1", "d2"])


@_skip
class TestSendLayout(unittest.TestCase):
    def test_offsets_are_256_byte_aligned_and_packed(self):
        offsets, total = GinOverlapPrimitive._send_layout([1, 64, 65, 0])
        # 4 B -> 256, 256 B -> 256, 260 B -> 512, 0 B -> 0
        self.assertEqual(offsets, [0, 256, 512, 1024])
        self.assertEqual(total, 1024)

    def test_empty(self):
        self.assertEqual(GinOverlapPrimitive._send_layout([]), ([], 0))


@_skip
class TestRecvPoolRing(unittest.TestCase):
    def test_single_bucket_does_not_wrap_and_waits_for_itself(self):
        off, d = _ring([0], [KB4], 4 * KB4)
        self.assertEqual(off, {0: 0})
        # the only predecessor is its own copy from the previous step: d == n
        self.assertEqual(d, {0: 1})

    def test_buckets_wrap_to_zero_at_the_end_of_the_segment(self):
        # order 2, 0, 1; each slot is 3 KB of a 7 KB segment: the third does
        # not fit after the first two and wraps to 0
        seg = [3072, 3072, 3072]
        off, d = _ring([2, 0, 1], seg, 7168)
        self.assertEqual(off, {2: 0, 0: 3072, 1: 0})
        # bucket 1 reuses bucket 2's range of the same step: distance 2
        # bucket 2 (position 0, next step) reuses bucket 1's: distance 1
        self.assertEqual(d, {2: 1, 0: 3, 1: 2})

    def test_no_wrap_when_everything_fits(self):
        off, d = _ring([0, 1, 2], [1024, 2048, 1024], 4 * KB4)
        self.assertEqual(off, {0: 0, 1: 1024, 2: 3072})
        self.assertEqual(d, {0: 3, 1: 3, 2: 3})

    def test_zero_size_slot_is_left_out(self):
        off, d = _ring([0, 1, 2], [1024, 0, 1024], KB4)
        self.assertNotIn(1, off)
        self.assertNotIn(1, d)
        self.assertEqual(off, {0: 0, 2: 1024})

    def test_distance_is_between_1_and_n(self):
        order = [3, 1, 4, 0, 2]
        seg = [1000, 3000, 2000, 1500, 2500]
        for P in (3000, 4000, 5000, 6000, 20000):
            with self.subTest(P=P):
                off, d = _ring(order, seg, P)
                self.assertEqual(set(d), set(off))
                for b, v in d.items():
                    self.assertGreaterEqual(v, 1)
                    self.assertLessEqual(v, len(order))
                for b, o in off.items():  # every slot lies inside the segment
                    self.assertGreaterEqual(o, 0)
                    self.assertLessEqual(o + seg[b], P)

    def test_slot_larger_than_the_segment_raises(self):
        with self.assertRaisesRegex(
            RuntimeError, r"slot of bucket 0 is 8192 B, larger than .* 4096 B"
        ):
            _ring([0], [8192], KB4)


@_skip
class TestRecvPoolSizes(unittest.TestCase):
    def test_dense_split_in_proportion_to_the_lower_bounds(self):
        lb = [KB4, 3 * KB4, 0]
        P = _sizes(lb, dense_gb=(32 * KB4) / 2**30, moe_gb=1.0)
        self.assertEqual(P[:2], [8 * KB4, 24 * KB4])
        self.assertEqual(P[2], 0)  # no expert segment: lower bound 0
        for v in P:
            self.assertEqual(v % KB4, 0)

    def test_expert_segment_is_rounded_down_to_4kb(self):
        lb = [0, 0, KB4]
        P = _sizes(lb, dense_gb=1.0, moe_gb=(3 * KB4 + 100) / 2**30)
        self.assertEqual(P, [0, 0, 3 * KB4])

    def test_dense_pool_too_small_raises(self):
        with self.assertRaisesRegex(
            RuntimeError, r"dense_recv_pool_gb=.* is too small.*\(16384 B\)"
        ):
            _sizes([KB4, 3 * KB4, 0], dense_gb=KB4 / 2**30, moe_gb=1.0)

    def test_moe_pool_too_small_raises(self):
        with self.assertRaisesRegex(
            RuntimeError, r"moe_recv_pool_gb=.* is too small: d2"
        ):
            _sizes([0, 0, 2 * KB4], dense_gb=1.0, moe_gb=KB4 / 2**30)


@_skip
class TestBuildHint(unittest.TestCase):
    def test_build_hint_lists_the_cmake_gate(self):
        for cond in (
            "WITH_DISTRIBUTE",
            "WITH_GPU",
            "WITH_NCCL",
            "CUDA >= 13.0",
            "NCCL >= 2.30.7",
            "nccl_device.h",
            ">= 9.0",
        ):
            self.assertIn(cond, engine._BUILD_HINT)


if __name__ == "__main__":
    unittest.main()
