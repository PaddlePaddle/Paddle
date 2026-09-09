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


class TestIndexElementwiseNegativeStride(unittest.TestCase):
    """Advanced indexing on a reversed view.

    ``x[::-1]`` is a view with a negative stride whose base offset points at the
    highest address of that axis, so the gather kernel has to reach *backwards*
    from ``slice_offset``. Its offset calculator must use a signed offset type;
    an unsigned one wraps those offsets into huge positive values and the kernel
    reads outside ``x``. numpy is the reference -- torch refuses negative steps.

    Ranks above two matter on their own: once an axis is left over after the
    indexed one, the iteration dimensions get sorted by stride, and ordering a
    negative stride as the smallest one permutes the output layout.
    """

    SHAPES = ((8, 6), (8, 6, 6), (8, 6, 33))

    def _cases(self, ndim):
        cases = [
            ('x[::-1, idx]', lambda t, i: t[::-1, i]),
            ('x[::-2, idx]', lambda t, i: t[::-2, i]),
            ('x[1::-1, idx]', lambda t, i: t[1::-1, i]),
            ('x[idx, ::-1]', lambda t, i: t[i, ::-1]),
            ('x[::-1][idx]', lambda t, i: t[::-1][i]),
            ('x[:, ::-1][idx]', lambda t, i: t[:, ::-1][i]),
            ('x[::-1, ::-1][idx]', lambda t, i: t[::-1, ::-1][i]),
            ('x[..., ::-1][idx]', lambda t, i: t[..., ::-1][i]),
            ('x[::-1, idx2]', lambda t, i: t[::-1, i]),
            # x[::-1][:, idx] is left out on purpose: indexing an axis that
            # comes after a sliced one on a non-contiguous base is broken for
            # positive steps too, so it is not a negative-stride issue.
        ]
        if ndim > 2:
            cases += [
                ('x[::-1, :, idx]', lambda t, i: t[::-1, :, i]),
                ('x[idx, ::-1, :]', lambda t, i: t[i, ::-1, :]),
                ('x[::-1, idx, ::-1]', lambda t, i: t[::-1, i, ::-1]),
            ]
        return cases

    def test_dygraph(self):
        paddle.disable_static()
        try:
            idx_np = np.array([2, 5, 2, 0, 5], dtype=np.int64)
            idx2_np = np.array([[1, 3], [3, 1], [0, 0]], dtype=np.int64)
            for shape in self.SHAPES:
                x_np = np.arange(np.prod(shape), dtype=np.float32).reshape(
                    shape
                )
                for dtype in ('float32', 'float64', 'int64'):
                    for name, fn in self._cases(len(shape)):
                        index_np = idx2_np if 'idx2' in name else idx_np
                        with self.subTest(dtype=dtype, shape=shape, expr=name):
                            expected = fn(x_np.astype(dtype), index_np)
                            out = fn(
                                paddle.to_tensor(x_np, dtype=dtype),
                                paddle.to_tensor(index_np),
                            )
                            self.assertEqual(
                                list(out.shape), list(expected.shape)
                            )
                            np.testing.assert_array_equal(out.numpy(), expected)
        finally:
            paddle.enable_static()

    def test_bool_mask_dygraph(self):
        # A boolean mask takes its own route through the parser
        # (getValueForBoolTensor) instead of the integer-index one, so the
        # reversed base has to be handled there as well.
        paddle.disable_static()
        try:
            for shape in ((8, 6), (8, 6, 6)):
                x_np = np.arange(np.prod(shape), dtype=np.float32).reshape(
                    shape
                )
                mask_np = (np.arange(np.prod(shape)) % 3 == 0).reshape(shape)
                col_np = np.array(
                    [True, False, True, True, False, False], dtype=bool
                )
                cases = [
                    ('x[::-1][mask]', lambda t, m, c: t[::-1][m]),
                    ('x[:, ::-1][mask]', lambda t, m, c: t[:, ::-1][m]),
                    ('x[::-1, col]', lambda t, m, c: t[::-1, c]),
                ]
                # A full-rank mask is served by masked_select, which XPU does
                # not register for float64, so the dtypes stay in the set every
                # backend has -- the reversed axis is handled dtype-agnostically
                # anyway.
                for dtype in ('float32', 'int64'):
                    for name, fn in cases:
                        with self.subTest(dtype=dtype, shape=shape, expr=name):
                            expected = fn(x_np.astype(dtype), mask_np, col_np)
                            out = fn(
                                paddle.to_tensor(x_np, dtype=dtype),
                                paddle.to_tensor(mask_np),
                                paddle.to_tensor(col_np),
                            )
                            self.assertEqual(
                                list(out.shape), list(expected.shape)
                            )
                            np.testing.assert_array_equal(out.numpy(), expected)
        finally:
            paddle.enable_static()


class TestIndexElementwisePutNegativeStride(unittest.TestCase):
    """Assignment through a reversed view.

    ``x[::-1, idx] = v`` writes into the base buffer in place, so a backend
    cannot sidestep the negative stride by materialising the view first --
    that would drop the write. The scatter kernel itself has to honour it.
    numpy is the reference; the indices are kept distinct so that the expected
    result does not depend on which duplicate write lands last.
    """

    SHAPES = ((8, 6), (8, 6, 6))

    def _cases(self, ndim):
        rev = slice(None, None, -1)
        cases = [
            ('x[::-1, idx]', lambda i: (rev, i)),
            ('x[::-2, idx]', lambda i: (slice(None, None, -2), i)),
            ('x[1::-1, idx]', lambda i: (slice(1, None, -1), i)),
            ('x[idx, ::-1]', lambda i: (i, rev)),
        ]
        if ndim > 2:
            cases += [
                ('x[::-1, :, idx]', lambda i: (rev, slice(None), i)),
                ('x[idx, ::-1, :]', lambda i: (i, rev, slice(None))),
                ('x[::-1, idx, ::-1]', lambda i: (rev, i, rev)),
            ]
        return cases

    def test_dygraph(self):
        paddle.disable_static()
        try:
            idx_np = np.array([2, 5, 0, 3], dtype=np.int64)
            for shape in self.SHAPES:
                x_np = np.arange(np.prod(shape), dtype=np.float32).reshape(
                    shape
                )
                for dtype in ('float32', 'float64', 'int64'):
                    for name, key_fn in self._cases(len(shape)):
                        np_key = key_fn(idx_np)
                        target_shape = x_np[np_key].shape
                        value_np = (
                            -np.arange(
                                1, np.prod(target_shape) + 1, dtype=np.float32
                            ).reshape(target_shape)
                        ).astype(dtype)
                        for kind in ('tensor', 'scalar'):
                            with self.subTest(
                                dtype=dtype, shape=shape, expr=name, value=kind
                            ):
                                expected = x_np.astype(dtype)
                                got = paddle.to_tensor(x_np, dtype=dtype)
                                pd_key = key_fn(paddle.to_tensor(idx_np))
                                if kind == 'tensor':
                                    expected[np_key] = value_np
                                    got[pd_key] = paddle.to_tensor(
                                        value_np, dtype=dtype
                                    )
                                else:
                                    expected[np_key] = -7
                                    got[pd_key] = -7
                                np.testing.assert_array_equal(
                                    got.numpy(), expected
                                )
        finally:
            paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
