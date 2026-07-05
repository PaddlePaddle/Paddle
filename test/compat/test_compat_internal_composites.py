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

"""``paddle.enable_compat()`` at its default level installs only the
``torch -> paddle`` import proxy and does NOT alias ``paddle.*`` — so the composite
APIs below run against genuinely NATIVE implementations (no ``level`` argument, thus
runnable on develop). Each composite internally calls one of the top-level names,
and the test pins the native results, i.e. verifies that merely enabling torch
compat does not perturb paddle's own composite APIs:

- vsplit / hsplit / dsplit / tensor_split / chunk  -> split
- quantile / nanquantile                           -> paddle.sort
- histogram_bin_edges                              -> paddle.min / paddle.max
- nan_to_num / F.nll_loss (ignore_index, mean)     -> paddle.equal

Inputs are fixed (no RNG) and ops are lightweight so the file stays well under the
newly-added-UT CI budget (ctest --repeat-until-fail 3 --timeout 15).
"""

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


def setUpModule():
    # default level: installs the torch import proxy only, does NOT alias paddle.*
    paddle.enable_compat()


def tearDownModule():
    paddle.disable_compat()


class TestSurfaceStaysNative(unittest.TestCase):
    """enable_compat() at default level does NOT alias paddle.*, so the public
    surface — and the composites below — stay native (native uses ``axis=``, not
    the torch-style ``dim=``)."""

    def test_surface_is_native(self):
        t = paddle.to_tensor([[3.0, 1.0, 2.0]])
        self.assertNotIsInstance(paddle.sort(t), tuple)  # native: plain Tensor
        self.assertEqual(
            paddle.max(t, axis=1).numpy().tolist(), [3.0]
        )  # native axis= works
        with self.assertRaises(TypeError):
            paddle.sort(t, dim=-1)  # no torch-style dim= on native


class TestSplitFamily(unittest.TestCase):
    """vsplit/hsplit/dsplit/tensor_split/chunk split along an axis; results must
    match numpy's array_split."""

    def test_vsplit(self):
        x = np.arange(48, dtype="float32").reshape([4, 4, 3])
        outs = paddle.vsplit(paddle.to_tensor(x), 2)
        for o, r in zip(outs, np.array_split(x, 2, axis=0)):
            np.testing.assert_array_equal(o.numpy(), r)

    def test_hsplit(self):
        x = np.arange(24, dtype="float32").reshape([4, 6])
        outs = paddle.hsplit(paddle.to_tensor(x), 3)
        for o, r in zip(outs, np.array_split(x, 3, axis=1)):
            np.testing.assert_array_equal(o.numpy(), r)

    def test_dsplit(self):
        x = np.arange(48, dtype="float32").reshape([2, 4, 6])
        outs = paddle.dsplit(paddle.to_tensor(x), 2)
        for o, r in zip(outs, np.array_split(x, 2, axis=2)):
            np.testing.assert_array_equal(o.numpy(), r)

    def test_tensor_split(self):
        # uneven split: array_split-style remainder distribution
        x = np.arange(7, dtype="float32")
        outs = paddle.tensor_split(paddle.to_tensor(x), 3)
        self.assertEqual(len(outs), 3)
        for o, r in zip(outs, np.array_split(x, 3)):
            np.testing.assert_array_equal(o.numpy(), r)

    def test_chunk(self):
        # chunk: `chunks` is the chunk COUNT (not the per-chunk size).
        x = np.arange(18, dtype="float32").reshape([6, 3])
        outs = paddle.chunk(paddle.to_tensor(x), 3, axis=0)
        self.assertEqual(len(outs), 3)
        for o, r in zip(outs, np.split(x, 3, axis=0)):
            np.testing.assert_array_equal(o.numpy(), r)


class TestReduceSortCompare(unittest.TestCase):
    def test_quantile(self):
        # quantile internally calls paddle.sort(x, axis)
        x = np.array(
            [
                [0.2, 0.7, 0.1, 0.4],
                [1.0, 0.3, 0.8, 0.5],
                [0.6, 0.9, 0.0, 0.25],
            ],
            dtype="float32",
        )
        got = paddle.quantile(paddle.to_tensor(x), 0.35, axis=1)
        np.testing.assert_allclose(
            got.numpy(), np.quantile(x, 0.35, axis=1), rtol=1e-5
        )

    def test_nanquantile(self):
        # nanquantile also routes through paddle.sort, ignoring NaNs
        x = np.array(
            [[0.2, np.nan, 0.1, 0.4], [1.0, 0.3, 0.8, 0.5]], dtype="float32"
        )
        got = paddle.nanquantile(paddle.to_tensor(x), 0.5, axis=1)
        np.testing.assert_allclose(
            got.numpy(), np.nanquantile(x, 0.5, axis=1), rtol=1e-5
        )

    def test_nan_to_num(self):
        # paddle.equal feeds paddle.where
        x = np.array([1.0, np.nan, np.inf, -np.inf, -2.5], dtype="float32")
        got = paddle.nan_to_num(paddle.to_tensor(x), nan=0.5)
        np.testing.assert_allclose(got.numpy(), np.nan_to_num(x, nan=0.5))

    def test_histogram_bin_edges(self):
        # internally computes paddle.min / paddle.max of the input
        x = np.array([0.0, 1.5, 3.0, 4.5, 6.0], dtype="float32")
        got = paddle.histogram_bin_edges(paddle.to_tensor(x), bins=4)
        np.testing.assert_allclose(
            got.numpy(), np.histogram_bin_edges(x, bins=4), rtol=1e-6
        )

    def test_nll_loss(self):
        # reduction='mean' + ignore_index takes the paddle.equal(count, 0.) path
        prob = np.array(
            [
                [0.70, 0.10, 0.10, 0.10],
                [0.20, 0.50, 0.20, 0.10],
                [0.10, 0.20, 0.60, 0.10],
                [0.25, 0.25, 0.25, 0.25],
                [0.10, 0.20, 0.20, 0.50],
            ],
            dtype="float32",
        )
        logp = np.log(prob)
        label = np.array([0, 1, 2, 1, 3], dtype="int64")
        got = F.nll_loss(
            paddle.to_tensor(logp),
            paddle.to_tensor(label),
            ignore_index=1,
            reduction="mean",
        )
        keep = label != 1
        ref = -logp[np.arange(5)[keep], label[keep]].sum() / keep.sum()
        np.testing.assert_allclose(got.item(), ref, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
