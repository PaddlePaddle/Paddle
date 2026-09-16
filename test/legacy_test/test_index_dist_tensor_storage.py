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
# The strided-index gather/scatter locates its buffer via
# `tensor.data() + (sub_view.data() - tensor.data())`, which is only sound
# when the sub-view shares the source allocation. Dense strided_slice returns
# a view that shares storage, but DistTensor inputs materialize strided_slice
# into a fresh buffer, making that pointer subtraction cross-allocation UB.
#
# - The dense tests below exercise the common shared-storage path (guard is a
#   no-op) and guard against regressions on it. They run everywhere.
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

    def test_int_advanced_setitem(self):
        x = paddle.to_tensor(self.x_np)
        idx = paddle.to_tensor(np.array([0, 2, 3], dtype="int64"))
        val = paddle.to_tensor(np.ones((3, 5, 6), dtype="float32") * 7.0)
        x[idx] = val
        expected = self.x_np.copy()
        expected[np.array([0, 2, 3])] = 7.0
        np.testing.assert_allclose(x.numpy(), expected)


@unittest.skipUnless(
    paddle.is_compiled_with_distribute(),
    "DistTensor strided_slice materializes a fresh buffer (non-shared "
    "storage). This path only exists in a WITH_DISTRIBUTE=ON build.",
)
class TestDistTensorAdvancedIndexStorage(unittest.TestCase):
    """Non-shared-storage path: DistTensor inputs materialize strided_slice.

    getitem must gather from the materialized buffer with a zero offset;
    setitem is conservatively unsupported and must raise cleanly rather than
    write through a garbage offset.
    """

    def setUp(self):
        paddle.disable_static()
        import paddle.distributed as dist

        self._dist = dist
        self._mesh = dist.ProcessMesh([0], dim_names=["x"])
        self.x_np = np.arange(4 * 5 * 6).reshape(4, 5, 6).astype("float32")
        self.mask_np = np.arange(4 * 5).reshape(4, 5) % 3 == 0

    def test_partial_bool_getitem_matches_dense(self):
        x = paddle.to_tensor(self.x_np)
        dist_x = self._dist.shard_tensor(
            x, self._mesh, [self._dist.Replicate()]
        )
        mask = paddle.to_tensor(self.mask_np)
        out = dist_x[mask]
        # Guard must yield the same values as the dense reference; a garbage
        # offset would read out of bounds and mismatch (or crash).
        expected = self.x_np[self.mask_np]
        np.testing.assert_allclose(np.asarray(out.numpy()), expected)

    def test_advanced_setitem_is_clean(self):
        x = paddle.to_tensor(self.x_np)
        dist_x = self._dist.shard_tensor(
            x, self._mesh, [self._dist.Replicate()]
        )
        idx = paddle.to_tensor(np.array([0, 2, 3], dtype="int64"))
        val = paddle.to_tensor(np.ones((3, 5, 6), dtype="float32") * 7.0)
        # Either the write succeeds and matches the dense reference (shared
        # storage), or it is refused with a clear Unimplemented error. It must
        # never silently corrupt memory.
        try:
            dist_x[idx] = val
        except (RuntimeError, ValueError) as e:
            self.assertIn("not supported", str(e))
            return
        expected = self.x_np.copy()
        expected[np.array([0, 2, 3])] = 7.0
        np.testing.assert_allclose(np.asarray(dist_x.numpy()), expected)


if __name__ == "__main__":
    unittest.main()
