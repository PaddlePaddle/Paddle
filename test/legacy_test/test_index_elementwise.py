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


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
