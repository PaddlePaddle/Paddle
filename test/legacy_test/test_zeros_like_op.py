# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_device_place

import paddle
from paddle import _C_ops, base, zeros_like
from paddle.base import Program, program_guard
from paddle.base.framework import convert_np_dtype_to_dtype_


class TestZerosLikeAPI(unittest.TestCase):
    def test_api(self):
        shape = [3, 4]
        startup_program = Program()
        train_program = Program()
        with program_guard(train_program, startup_program):
            x = paddle.static.data('X', shape)
            out1 = zeros_like(x)
            out2 = zeros_like(x, np.bool_)
            out3 = zeros_like(x, 'float64')
            out4 = zeros_like(x, 'int32')
            out5 = zeros_like(x, 'int64')
        place = get_device_place()
        exe = base.Executor(place)
        outs = exe.run(
            train_program,
            feed={'X': np.ones(shape).astype('float32')},
            fetch_list=[out1, out2, out3, out4, out5],
        )
        for i, dtype in enumerate(
            [np.float32, np.bool_, np.float64, np.int32, np.int64]
        ):
            self.assertEqual(outs[i].dtype, dtype)
            self.assertEqual((outs[i] == np.zeros(shape, dtype)).all(), True)


class TestZerosLikeImperative(unittest.TestCase):
    def test_out(self):
        shape = [3, 4]
        place = get_device_place()
        paddle.disable_static(place)
        x = paddle.to_tensor(np.ones(shape))
        for dtype in [np.bool_, np.float32, np.float64, np.int32, np.int64]:
            out = zeros_like(x, dtype)
            self.assertEqual(
                (out.numpy() == np.zeros(shape, dtype)).all(), True
            )
        out = paddle.zeros_like(x)
        self.assertEqual((out.numpy() == np.zeros(shape, dtype)).all(), True)
        out = paddle.tensor.creation.zeros_like(x)
        self.assertEqual((out.numpy() == np.zeros(shape, dtype)).all(), True)
        paddle.enable_static()


class TestZerosAPI(unittest.TestCase):
    def test_api(self):
        shape = [3, 4]
        place = get_device_place()
        paddle.disable_static(place)

        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            out = _C_ops.zeros(shape, convert_np_dtype_to_dtype_(dtype), place)
            self.assertEqual(
                (out.numpy() == np.zeros(shape, dtype)).all(), True
            )

        paddle.enable_static()


class TestZerosLikeAlias(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_check_output(self):
        """
        Test the alias of zeros_like function.
        ``zeros_like(input=x)`` is equivalent to ``zeros_like(x=x)``
        """
        shape_cases = [
            [2],
            [2, 4],
            [2, 4, 8],
        ]
        dtype_cases = [
            None,
            "float32",
            "float64",
            "int32",
            "int64",
            "bool",
        ]

        for shape in shape_cases:
            for dtype in dtype_cases:
                x = paddle.rand(shape)
                for param_alias in ["x", "input"]:
                    if dtype is None:
                        out = paddle.zeros_like(**{param_alias: x})
                        expected = np.zeros_like(x.numpy())
                    else:
                        out = paddle.zeros_like(**{param_alias: x}, dtype=dtype)
                        expected = np.zeros_like(x.numpy(), dtype=dtype)

                    if dtype == "bool":
                        np.testing.assert_array_equal(out.numpy(), expected)
                    else:
                        np.testing.assert_allclose(out.numpy(), expected)


if __name__ == '__main__':
    unittest.main()
