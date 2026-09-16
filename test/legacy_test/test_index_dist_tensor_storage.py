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

# Regression tests for the storage-sharing guard in
# paddle/fluid/pybind/slice_utils.h.
#
# The strided-index gather/scatter locates its buffer via the byte distance
# `sub_view.data() - tensor.data()`. That pointer subtraction is only defined
# when the sub-view shares the source allocation. Dense strided_slice returns a
# view that shares storage, but DistTensor inputs materialize strided_slice into
# a fresh buffer, making that subtraction a cross-allocation operation. The
# guard computes the offset only when the storage is shared (getitem then
# gathers from the materialized buffer with offset 0; setitem refuses).
#
# IMPORTANT: the non-shared path is only reachable when a *basic* slice is
# present (e.g. `x[1:3, idx]`), because a pure advanced index leaves
# `slice_axes` empty and `DecideSubTensor` returns the source tensor itself
# (sub_tensor == tensor, always shared). Every case below therefore combines a
# basic slice with an advanced index.
#
# - The dense tests exercise the common shared-storage path (guard falls through
#   to the pointer subtraction) and guard against regressions on it. They run
#   everywhere.
# - The DistTensor test exercises the non-shared path. It requires a build with
#   WITH_DISTRIBUTE=ON and is skipped otherwise.
import unittest

import numpy as np

import paddle


class TestDenseAdvancedIndexStorage(unittest.TestCase):
    """Common shared-storage path: guard must not change existing behavior."""

    def setUp(self):
        paddle.disable_static()
        self.x_np = np.arange(4 * 5 * 6).reshape(4, 5, 6).astype("float32")
        # Partial bool mask (rank < x.rank) drives the stride gather path
        # instead of the full-rank masked_select early return.
        self.mask_np = np.arange(4 * 5).reshape(4, 5) % 3 == 0

    def test_partial_bool_getitem(self):
        x = paddle.to_tensor(self.x_np)
        mask = paddle.to_tensor(self.mask_np)
        out = x[mask].numpy()
        expected = self.x_np[self.mask_np]
        np.testing.assert_allclose(out, expected)

    def test_partial_bool_setitem(self):
        x = paddle.to_tensor(self.x_np)
        mask = paddle.to_tensor(self.mask_np)
        x[mask] = 0.0
        expected = self.x_np.copy()
        expected[self.mask_np] = 0.0
        np.testing.assert_allclose(x.numpy(), expected)

    def test_slice_and_int_getitem(self):
        # Basic slice + int advanced index: exercises DealWithIndex's offset
        # computation on the shared-storage branch.
        x = paddle.to_tensor(self.x_np)
        idx = paddle.to_tensor(np.array([0, 2, 3], dtype="int64"))
        out = x[1:3, idx].numpy()
        expected = self.x_np[1:3, np.array([0, 2, 3])]
        np.testing.assert_allclose(out, expected)

    def test_slice_and_int_setitem(self):
        x = paddle.to_tensor(self.x_np)
        idx = paddle.to_tensor(np.array([0, 2, 3], dtype="int64"))
        val = paddle.to_tensor(np.ones((2, 3, 6), dtype="float32") * 7.0)
        x[1:3, idx] = val
        expected = self.x_np.copy()
        expected[1:3, np.array([0, 2, 3])] = 7.0
        np.testing.assert_allclose(x.numpy(), expected)


@unittest.skipUnless(
    paddle.is_compiled_with_distribute(),
    "DistTensor strided_slice materializes a fresh buffer (non-shared "
    "storage). This path only exists in a WITH_DISTRIBUTE=ON build.",
)
class TestDistTensorAdvancedIndexStorage(unittest.TestCase):
    """Non-shared-storage path: DistTensor inputs materialize strided_slice.

    A basic slice (`1:3`) makes strided_slice run and, for DistTensor, produce
    a fresh buffer that does not share storage with the source. getitem must
    gather from that materialized buffer with a zero offset; setitem is
    conservatively unsupported and must raise cleanly rather than write through
    a cross-allocation offset.
    """

    def setUp(self):
        paddle.disable_static()
        import paddle.distributed as dist

        self._dist = dist
        self._mesh = dist.ProcessMesh([0], dim_names=["x"])
        self.x_np = np.arange(4 * 5 * 6).reshape(4, 5, 6).astype("float32")

    def _shard(self):
        x = paddle.to_tensor(self.x_np)
        return self._dist.shard_tensor(x, self._mesh, [self._dist.Replicate()])

    def test_slice_and_int_getitem_matches_dense(self):
        dist_x = self._shard()
        idx = paddle.to_tensor(np.array([0, 2, 3], dtype="int64"))
        # Basic slice forces strided_slice -> materialized non-shared buffer.
        out = dist_x[1:3, idx]
        # Guard must yield the same values as the dense reference; a garbage
        # cross-allocation offset would read out of bounds and mismatch.
        expected = self.x_np[1:3, np.array([0, 2, 3])]
        np.testing.assert_allclose(np.asarray(out.numpy()), expected)

    def test_slice_and_int_setitem_is_clean(self):
        dist_x = self._shard()
        idx = paddle.to_tensor(np.array([0, 2, 3], dtype="int64"))
        val = paddle.to_tensor(np.ones((2, 3, 6), dtype="float32") * 7.0)
        # Either the write succeeds and matches the dense reference (shared
        # storage), or it is refused with a clear Unimplemented error. It must
        # never silently corrupt memory through a cross-allocation offset.
        try:
            dist_x[1:3, idx] = val
        except (RuntimeError, ValueError) as e:
            self.assertIn("not supported", str(e))
            return
        expected = self.x_np.copy()
        expected[1:3, np.array([0, 2, 3])] = 7.0
        np.testing.assert_allclose(np.asarray(dist_x.numpy()), expected)


if __name__ == "__main__":
    unittest.main()
