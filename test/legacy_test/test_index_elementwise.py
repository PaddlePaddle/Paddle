# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle


def np_index_elementwise(x, index):
    return x[index]


class TestIndexElementwiseBool(unittest.TestCase):
    def init(self):
        self.dim = 3
        self.x_shape = (4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"

    def setUp(self):
        self.init()

        if self.dtype == "bool":
            self.x_np = np.random.randint(
                2, size=self.x_shape, dtype=self.dtype
            )
        elif self.dtype in ["float32", "float64"]:
            self.x_np = np.random.random(self.x_shape).astype(self.dtype)
        elif self.dtype in ["int32", "int8", "int64", "int16", "uint8"]:
            self.x_np = np.random.randint(
                100, size=self.x_shape, dtype=self.dtype
            )
        elif self.dtype == "float16":
            self.x_np = np.random.random(self.x_shape).astype("float16")
        elif self.dtype == "complex64":
            self.x_np = (
                np.random.random(self.x_shape)
                + 1j * np.random.random(self.x_shape)
            ).astype("complex64")
        elif self.dtype == "complex128":
            self.x_np = (
                np.random.random(self.x_shape)
                + 1j * np.random.random(self.x_shape)
            ).astype("complex128")

        self.index_np = np.random.randint(
            2, size=self.index_shape, dtype="bool"
        )

        self.out_np = np_index_elementwise(self.x_np, self.index_np)

    def test_dygraph(self):
        paddle.disable_static()

        x = paddle.to_tensor(self.x_np, dtype=self.dtype)
        index = paddle.to_tensor(self.index_np).astype('bool')
        result = x[index].numpy()

        atol = 1e-05 if self.dtype in ["float32", "float64"] else 0
        rtol = 1e-05 if self.dtype in ["float32", "float64"] else 0

        np.testing.assert_allclose(result, self.out_np, atol=atol, rtol=rtol)

        paddle.enable_static()


class TestIndexElementwiseBool3D(TestIndexElementwiseBool):
    def init(self):
        self.dim = 3
        self.x_shape = (4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseBool4D_k2(TestIndexElementwiseBool):
    def init(self):
        self.dim = 4
        self.x_shape = (3, 4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseBool4D_k3(TestIndexElementwiseBool):
    def init(self):
        self.dim = 4
        self.x_shape = (3, 4, 5, 6)
        self.k = 3
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseBool5D_k2(TestIndexElementwiseBool):
    def init(self):
        self.dim = 5
        self.x_shape = (2, 3, 4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseBool5D_k3(TestIndexElementwiseBool):
    def init(self):
        self.dim = 5
        self.x_shape = (2, 3, 4, 5, 6)
        self.k = 3
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseBool5D_k4(TestIndexElementwiseBool):
    def init(self):
        self.dim = 5
        self.x_shape = (2, 3, 4, 5, 6)
        self.k = 4
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseBool4D_k3_AllDtypes(TestIndexElementwiseBool):
    def init(self):
        self.dim = 4
        self.x_shape = (3, 4, 5, 6)
        self.k = 3
        self.dtype = None
        self.index_shape = self.x_shape[: self.k]

    def setUp(self):
        self.init()
        self.dtypes = [
            "bool",
            "float32",
            "float64",
            "int32",
            "int8",
            "int64",
            "int16",
            "uint8",
            # "float16",
            # "bfloat16",
            "complex64",
            "complex128",
        ]

        for dtype in self.dtypes:
            self.dtype = dtype
            if self.dtype == "bool":
                self.x_np = np.random.randint(
                    2, size=self.x_shape, dtype=self.dtype
                )
            elif self.dtype in ["float32", "float64"]:
                self.x_np = np.random.random(self.x_shape).astype(self.dtype)
            elif self.dtype in ["int32", "int8", "int64", "int16", "uint8"]:
                self.x_np = np.random.randint(
                    100, size=self.x_shape, dtype=self.dtype
                )
            elif self.dtype == "float16":
                self.x_np = np.random.random(self.x_shape).astype("float16")
            elif self.dtype == "complex64":
                self.x_np = (
                    np.random.random(self.x_shape)
                    + 1j * np.random.random(self.x_shape)
                ).astype("complex64")
            elif self.dtype == "complex128":
                self.x_np = (
                    np.random.random(self.x_shape)
                    + 1j * np.random.random(self.x_shape)
                ).astype("complex128")

            self.index_np = np.random.randint(
                2, size=self.index_shape, dtype="bool"
            )
            self.out_np = np_index_elementwise(self.x_np, self.index_np)

            self.test_dygraph()


class TestIndexElementwiseGet0SizeInput(unittest.TestCase):
    """Test IndexElementwiseGetKernel with 0-size input tensor (forward).

    Regression tests for the bug where indexing a 0-size tensor with a
    list-of-list index (integer advanced indexing) triggered CUDA error(700)
    due to dereferencing a null data pointer.

    When x.numel() == 0, x.data<T>() returns nullptr. The kernel must
    early-return and zero-fill the output instead of launching the GPU kernel.
    """

    def _check_0size_getitem(self, dtype, idx, expected_shape):
        """Helper: index a [0,5,4,3] tensor and verify shape + zero values."""
        paddle.disable_static()
        x = paddle.zeros([0, 5, 4, 3], dtype=dtype)
        out = x[idx]
        self.assertEqual(
            list(out.shape),
            expected_shape,
            f"dtype={dtype}: expected shape {expected_shape}, got {list(out.shape)}",
        )
        # All output elements must be zero (no garbage from uninitialized memory)
        np.testing.assert_array_equal(
            out.numpy(),
            np.zeros(expected_shape, dtype=out.numpy().dtype),
            err_msg=f"dtype={dtype}: output should be all zeros",
        )
        paddle.enable_static()

    def test_complex128_positive_indices(self):
        """Reproduces original CUDA error(700): complex128, positive indices."""
        # [[2,3,4],[1,2,5]] is a 2D index of shape [2,3] applied to dim 0
        # Output shape = [2,3] + x.shape[1:] = [2,3,5,4,3]
        self._check_0size_getitem(
            'complex128', [[2, 3, 4], [1, 2, 5]], [2, 3, 5, 4, 3]
        )

    def test_complex128_negative_indices(self):
        """Test complex128 with negative indices in the index list."""
        self._check_0size_getitem(
            'complex128', [[2, -3, -4], [-1, 2, 5]], [2, 3, 5, 4, 3]
        )

    def test_bool_positive_indices(self):
        """Test bool dtype with positive indices."""
        self._check_0size_getitem(
            'bool', [[2, 3, 4], [1, 2, 5]], [2, 3, 5, 4, 3]
        )

    def test_bool_negative_indices(self):
        """Test bool dtype with negative indices."""
        self._check_0size_getitem(
            'bool', [[2, -3, -4], [-1, 2, 5]], [2, 3, 5, 4, 3]
        )

    def test_float32(self):
        """Test float32 dtype."""
        self._check_0size_getitem(
            'float32', [[2, 3, 4], [1, 2, 5]], [2, 3, 5, 4, 3]
        )

    def test_float64(self):
        """Test float64 dtype."""
        self._check_0size_getitem(
            'float64', [[2, 3, 4], [1, 2, 5]], [2, 3, 5, 4, 3]
        )

    def test_int64(self):
        """Test int64 dtype."""
        self._check_0size_getitem(
            'int64', [[2, 3, 4], [1, 2, 5]], [2, 3, 5, 4, 3]
        )

    def test_float16(self):
        """Test float16 dtype."""
        self._check_0size_getitem(
            'float16', [[2, 3, 4], [1, 2, 5]], [2, 3, 5, 4, 3]
        )

    def test_1d_index_on_0size_input(self):
        """Test 1D integer index on 0-size input (single dim advanced index)."""
        paddle.disable_static()
        # x.shape=[0,5,4], index [2,3] for dim 0 → result [2,5,4]
        x = paddle.zeros([0, 5, 4], dtype='float32')
        out = x[[2, 3]]
        self.assertEqual(list(out.shape), [2, 5, 4])
        np.testing.assert_array_equal(
            out.numpy(), np.zeros([2, 5, 4], dtype='float32')
        )
        paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
