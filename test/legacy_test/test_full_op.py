#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base


# Test python API
class TestFullAPI(unittest.TestCase):

    def test_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            positive_2_int32 = paddle.tensor.fill_constant([1], "int32", 2)

            positive_2_int64 = paddle.tensor.fill_constant([1], "int64", 2)
            shape_tensor_int32 = paddle.static.data(
                name="shape_tensor_int32", shape=[2], dtype="int32"
            )

            shape_tensor_int64 = paddle.static.data(
                name="shape_tensor_int64", shape=[2], dtype="int64"
            )

            out_1 = paddle.full(shape=[1, 2], dtype="float32", fill_value=1.1)

            out_2 = paddle.full(
                shape=[1, positive_2_int32], dtype="float32", fill_value=1.1
            )

            out_3 = paddle.full(
                shape=[1, positive_2_int64], dtype="float32", fill_value=1.1
            )

            out_4 = paddle.full(
                shape=shape_tensor_int32, dtype="float32", fill_value=1.2
            )

            out_5 = paddle.full(
                shape=shape_tensor_int64, dtype="float32", fill_value=1.1
            )

            out_6 = paddle.full(
                shape=shape_tensor_int64, dtype=np.float32, fill_value=1.1
            )

            val = paddle.tensor.fill_constant(
                shape=[1], dtype=np.float32, value=1.1
            )
            out_7 = paddle.full(
                shape=shape_tensor_int64, dtype=np.float32, fill_value=val
            )
            out_8 = paddle.full(shape=10, dtype=np.float32, fill_value=val)

            out_9 = paddle.full(
                shape=10, dtype="complex64", fill_value=1.1 + 1.1j
            )

            out_10 = paddle.full(
                shape=10, dtype="complex128", fill_value=1.1 + 1.1j
            )

            out_11 = paddle.full(
                shape=10, dtype="complex64", fill_value=1.1 + np.inf * 1j
            )

            out_12 = paddle.full(
                shape=10, dtype="complex128", fill_value=1.1 + np.inf * 1j
            )

            out_13 = paddle.full(
                shape=10, dtype="complex64", fill_value=1.1 - np.inf * 1j
            )

            out_14 = paddle.full(
                shape=10, dtype="complex128", fill_value=1.1 - np.inf * 1j
            )

            out_15 = paddle.full(
                shape=10, dtype="complex64", fill_value=1.1 + np.nan * 1j
            )

            out_16 = paddle.full(
                shape=10, dtype="complex128", fill_value=1.1 + np.nan * 1j
            )

            out_17 = paddle.full(shape=10, fill_value=1.1 + 1.1j)

            out_18 = paddle.full(shape=10, fill_value=True)

            exe = base.Executor(place=base.CPUPlace())
            (
                res_1,
                res_2,
                res_3,
                res_4,
                res_5,
                res_6,
                res_7,
                res_8,
                res_9,
                res_10,
                res_11,
                res_12,
                res_13,
                res_14,
                res_15,
                res_16,
                res_17,
                res_18,
            ) = exe.run(
                paddle.static.default_main_program(),
                feed={
                    "shape_tensor_int32": np.array([1, 2]).astype("int32"),
                    "shape_tensor_int64": np.array([1, 2]).astype("int64"),
                },
                fetch_list=[
                    out_1,
                    out_2,
                    out_3,
                    out_4,
                    out_5,
                    out_6,
                    out_7,
                    out_8,
                    out_9,
                    out_10,
                    out_11,
                    out_12,
                    out_13,
                    out_14,
                    out_15,
                    out_16,
                    out_17,
                    out_18,
                ],
            )

        np.testing.assert_array_equal(
            res_1, np.full([1, 2], 1.1, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_2, np.full([1, 2], 1.1, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_3, np.full([1, 2], 1.1, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_4, np.full([1, 2], 1.2, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_5, np.full([1, 2], 1.1, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_6, np.full([1, 2], 1.1, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_7, np.full([1, 2], 1.1, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_8, np.full([10], 1.1, dtype="float32")
        )
        np.testing.assert_allclose(
            res_9, np.full([10], 1.1 + 1.1j, dtype="complex64")
        )
        np.testing.assert_allclose(
            res_10, np.full([10], 1.1 + 1.1j, dtype="complex128")
        )
        np.testing.assert_allclose(
            res_9, np.full([10], 1.1 + 1.1j, dtype="complex64")
        )
        np.testing.assert_allclose(
            res_10, np.full([10], 1.1 + 1.1j, dtype="complex128")
        )
        np.testing.assert_allclose(
            res_11, np.full([10], 1.1 + np.inf * 1j, dtype="complex64")
        )
        np.testing.assert_allclose(
            res_12, np.full([10], 1.1 + np.inf * 1j, dtype="complex128")
        )
        np.testing.assert_allclose(
            res_13, np.full([10], 1.1 - np.inf * 1j, dtype="complex64")
        )
        np.testing.assert_allclose(
            res_14, np.full([10], 1.1 - np.inf * 1j, dtype="complex128")
        )
        np.testing.assert_allclose(
            res_15, np.full([10], 1.1 + np.nan * 1j, dtype="complex64")
        )
        np.testing.assert_allclose(
            res_16, np.full([10], 1.1 + np.nan * 1j, dtype="complex128")
        )
        np.testing.assert_allclose(res_17, np.full([10], 1.1 + 1.1j))
        np.testing.assert_array_equal(res_18, np.full([10], True))
        paddle.disable_static()

    def test_api_eager(self):
        with base.dygraph.base.guard():
            positive_2_int32 = paddle.tensor.fill_constant([1], "int32", 2)
            positive_2_int64 = paddle.tensor.fill_constant([1], "int64", 2)
            positive_4_int64 = paddle.tensor.fill_constant(
                [1], "int64", 4, True
            )

            out_1 = paddle.full(shape=[1, 2], dtype="float32", fill_value=1.1)

            out_2 = paddle.full(
                shape=[1, positive_2_int32.item()],
                dtype="float32",
                fill_value=1.1,
            )

            out_3 = paddle.full(
                shape=[1, positive_2_int64.item()],
                dtype="float32",
                fill_value=1.1,
            )

            out_4 = paddle.full(shape=[1, 2], dtype="float32", fill_value=1.2)

            out_5 = paddle.full(shape=[1, 2], dtype="float32", fill_value=1.1)

            out_6 = paddle.full(shape=[1, 2], dtype=np.float32, fill_value=1.1)

            val = paddle.tensor.fill_constant(
                shape=[1], dtype=np.float32, value=1.1
            )
            out_7 = paddle.full(shape=[1, 2], dtype=np.float32, fill_value=val)

            out_8 = paddle.full(
                shape=positive_2_int32, dtype="float32", fill_value=1.1
            )

            out_9 = paddle.full(
                shape=[
                    positive_2_int32,
                    positive_2_int64,
                    positive_4_int64,
                ],
                dtype="float32",
                fill_value=1.1,
            )

            # test for numpy.float64 as fill_value
            out_10 = paddle.full_like(
                out_7, dtype=np.float32, fill_value=np.abs(1.1)
            )

            out_11 = paddle.full(shape=10, dtype="float32", fill_value=1.1)

            out_12 = paddle.full(
                shape=[1, 2, 3], dtype="complex64", fill_value=1.1 + 1.1j
            )

            out_13 = paddle.full(
                shape=[1, 2, 3], dtype="complex128", fill_value=1.1 + 1.1j
            )

            out_14 = paddle.full(
                shape=[1, 2, 3], dtype="complex64", fill_value=1.1 + np.inf * 1j
            )

            out_15 = paddle.full(
                shape=[1, 2, 3],
                dtype="complex128",
                fill_value=1.1 + np.inf * 1j,
            )

            out_16 = paddle.full(
                shape=[1, 2, 3], dtype="complex64", fill_value=1.1 - np.inf * 1j
            )

            out_17 = paddle.full(
                shape=[1, 2, 3],
                dtype="complex128",
                fill_value=1.1 - np.inf * 1j,
            )

            out_18 = paddle.full(
                shape=[1, 2, 3], dtype="complex64", fill_value=1.1 + np.nan * 1j
            )

            out_19 = paddle.full(
                shape=[1, 2, 3],
                dtype="complex128",
                fill_value=1.1 + np.nan * 1j,
            )

            # test without dtype input for complex
            out_20 = paddle.full(shape=[1, 2, 3], fill_value=1.1 + 1.1j)

            # test without dtype input for bool
            out_21 = paddle.full(shape=[1, 2, 3], fill_value=True)

            np.testing.assert_array_equal(
                out_1, np.full([1, 2], 1.1, dtype="float32")
            )
            np.testing.assert_array_equal(
                out_2, np.full([1, 2], 1.1, dtype="float32")
            )
            np.testing.assert_array_equal(
                out_3, np.full([1, 2], 1.1, dtype="float32")
            )
            np.testing.assert_array_equal(
                out_4, np.full([1, 2], 1.2, dtype="float32")
            )
            np.testing.assert_array_equal(
                out_5, np.full([1, 2], 1.1, dtype="float32")
            )
            np.testing.assert_array_equal(
                out_6, np.full([1, 2], 1.1, dtype="float32")
            )
            np.testing.assert_array_equal(
                out_7, np.full([1, 2], 1.1, dtype="float32")
            )
            np.testing.assert_array_equal(
                out_8, np.full([2], 1.1, dtype="float32")
            )
            np.testing.assert_array_equal(
                out_9, np.full([2, 2, 4], 1.1, dtype="float32")
            )
            np.testing.assert_array_equal(
                out_10, np.full([1, 2], 1.1, dtype="float32")
            )
            np.testing.assert_array_equal(
                out_11, np.full([10], 1.1, dtype="float32")
            )
            np.testing.assert_allclose(
                out_12, np.full([1, 2, 3], 1.1 + 1.1j, dtype="complex64")
            )
            np.testing.assert_allclose(
                out_13, np.full([1, 2, 3], 1.1 + 1.1j, dtype="complex128")
            )
            np.testing.assert_allclose(
                out_14, np.full([1, 2, 3], 1.1 + np.inf * 1j, dtype="complex64")
            )
            np.testing.assert_allclose(
                out_15,
                np.full([1, 2, 3], 1.1 + np.inf * 1j, dtype="complex128"),
            )
            np.testing.assert_allclose(
                out_16, np.full([1, 2, 3], 1.1 - np.inf * 1j, dtype="complex64")
            )
            np.testing.assert_allclose(
                out_17,
                np.full([1, 2, 3], 1.1 - np.inf * 1j, dtype="complex128"),
            )
            np.testing.assert_allclose(
                out_18, np.full([1, 2, 3], 1.1 + np.nan * 1j, dtype="complex64")
            )
            np.testing.assert_allclose(
                out_19,
                np.full([1, 2, 3], 1.1 + np.nan * 1j, dtype="complex128"),
            )
            np.testing.assert_allclose(out_20, np.full([1, 2, 3], 1.1 + 1.1j))
            np.testing.assert_array_equal(out_21, np.full([1, 2, 3], True))


class TestFullOpError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # for ci coverage
            # The argument dtype of full must be one of bool, float16,
            # float32, float64, uint8, int16, int32 or int64
            self.assertRaises(
                TypeError, paddle.full, shape=[1], fill_value=5, dtype='uint4'
            )

            # The shape dtype of full op must be int32 or int64.
            def test_shape_tensor_dtype():
                shape = paddle.static.data(
                    name="shape_tensor", shape=[2], dtype="float32"
                )
                paddle.full(shape=shape, dtype="float32", fill_value=1)

            self.assertRaises(TypeError, test_shape_tensor_dtype)

            def test_shape_tensor_list_dtype():
                shape = paddle.static.data(
                    name="shape_tensor_list", shape=[1], dtype="bool"
                )
                paddle.full(shape=[shape, 2], dtype="float32", fill_value=1)

            self.assertRaises(TypeError, test_shape_tensor_list_dtype)
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
