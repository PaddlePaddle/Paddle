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

"""``enable_compat(level=2)`` is active for the whole module (a real user session).
Composite paddle APIs that internally call the aliased top-level names must keep
NATIVE behavior: caller-aware dispatch keeps paddle-internal callers on native, so
level=2 only changes the outward ``paddle.*`` surface. Internal call chains covered:

- vsplit / hsplit / dsplit / chunk  -> paddle.split(num_or_sections=, axis=)
- quantile                          -> paddle.sort(x, axis)
- nan_to_num                        -> paddle.equal -> paddle.where
- histogram_bin_edges               -> paddle.min / paddle.max
- F.nll_loss (ignore_index, mean)   -> paddle.equal

Inputs are fixed (no RNG) and ops are lightweight so the file stays well under the
newly-added-UT CI budget (ctest --repeat-until-fail 3 --timeout 15).
"""

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


def setUpModule():
    paddle.enable_compat(level=2)


def tearDownModule():
    paddle.disable_compat()


class TestCompatIsActuallyOn(unittest.TestCase):
    """Guard against a vacuous pass: this module is an external caller, so the
    torch-aligned surface must be in effect here while the composites stay native."""

    def test_external_surface_is_torch_style(self):
        t = paddle.to_tensor([[3.0, 1.0, 2.0]])
        self.assertIsInstance(
            paddle.split(t, 1, dim=0), tuple
        )  # torch: chunk size
        self.assertTrue(hasattr(paddle.sort(t, dim=-1), "values"))
        with self.assertRaises(TypeError):
            paddle.max(t, axis=1)  # native kwarg rejected externally


class TestSplitFamilyStaysNative(unittest.TestCase):
    """vsplit/hsplit/dsplit/chunk internally call paddle.split with native
    num_or_sections=/axis=; a compat-split leak would reinterpret those args."""

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

    def test_chunk(self):
        # chunk: `chunks` is the chunk COUNT; a compat-split leak would read 3 as
        # per-chunk size and fail the count/shape checks.
        x = np.arange(18, dtype="float32").reshape([6, 3])
        outs = paddle.chunk(paddle.to_tensor(x), 3, axis=0)
        self.assertEqual(len(outs), 3)
        for o, r in zip(outs, np.split(x, 3, axis=0)):
            np.testing.assert_array_equal(o.numpy(), r)


class TestReduceAndCompareStayNative(unittest.TestCase):
    def test_quantile_uses_native_sort(self):
        # quantile internally calls paddle.sort(x, axis); a compat-sort leak would
        # hand it a (values, indices) namedtuple instead of a tensor.
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

    def test_nan_to_num_uses_native_equal(self):
        # paddle.equal feeds paddle.where; compat.equal returns a python bool.
        x = np.array([1.0, np.nan, np.inf, -np.inf, -2.5], dtype="float32")
        got = paddle.nan_to_num(paddle.to_tensor(x), nan=0.5)
        np.testing.assert_allclose(got.numpy(), np.nan_to_num(x, nan=0.5))

    def test_histogram_bin_edges_uses_native_min_max(self):
        x = np.array([0.0, 1.5, 3.0, 4.5, 6.0], dtype="float32")
        got = paddle.histogram_bin_edges(paddle.to_tensor(x), bins=4)
        np.testing.assert_allclose(
            got.numpy(), np.histogram_bin_edges(x, bins=4), rtol=1e-6
        )

    def test_nll_loss_uses_native_equal(self):
        # reduction='mean' + ignore_index takes the paddle.equal(count, 0.) path.
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
