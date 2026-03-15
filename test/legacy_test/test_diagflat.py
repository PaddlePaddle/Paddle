#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_device_place, is_custom_device

import paddle
from paddle.static import Program


class TestDiagFlatAPI(unittest.TestCase):
    def setUp(self):
        self.input_np = np.random.random(size=(10, 10)).astype(np.float64)
        self.expected0 = np.diagflat(self.input_np)
        self.expected1 = np.diagflat(self.input_np, k=1)
        self.expected2 = np.diagflat(self.input_np, k=-1)

        self.input_np2 = np.random.random(size=(20)).astype(np.float64)
        self.expected3 = np.diagflat(self.input_np2)
        self.expected4 = np.diagflat(self.input_np2, k=1)
        self.expected5 = np.diagflat(self.input_np2, k=-1)

    def run_imperative(self):
        x = paddle.to_tensor(self.input_np)
        y = paddle.diagflat(x)
        np.testing.assert_allclose(y.numpy(), self.expected0, rtol=1e-05)

        y = paddle.diagflat(x=x)
        np.testing.assert_allclose(y.numpy(), self.expected0, rtol=1e-05)

        y = paddle.diagflat(input=x)
        np.testing.assert_allclose(y.numpy(), self.expected0, rtol=1e-05)

        y = paddle.diagflat(x, offset=1)
        np.testing.assert_allclose(y.numpy(), self.expected1, rtol=1e-05)

        y = paddle.diagflat(input=x, offset=1)
        np.testing.assert_allclose(y.numpy(), self.expected1, rtol=1e-05)

        y = paddle.diagflat(x, offset=-1)
        np.testing.assert_allclose(y.numpy(), self.expected2, rtol=1e-05)

        y = paddle.diagflat(input=x, offset=-1)
        np.testing.assert_allclose(y.numpy(), self.expected2, rtol=1e-05)

        x = paddle.to_tensor(self.input_np2)
        y = paddle.diagflat(x)
        np.testing.assert_allclose(y.numpy(), self.expected3, rtol=1e-05)

        y = paddle.diagflat(input=x)
        np.testing.assert_allclose(y.numpy(), self.expected3, rtol=1e-05)

        y = paddle.diagflat(x, offset=1)
        np.testing.assert_allclose(y.numpy(), self.expected4, rtol=1e-05)

        y = paddle.diagflat(x, offset=-1)
        np.testing.assert_allclose(y.numpy(), self.expected5, rtol=1e-05)

    def run_static(self, use_gpu=False):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name='input', shape=[10, 10], dtype='float64'
            )
            x2 = paddle.static.data(name='input2', shape=[20], dtype='float64')
            result0 = paddle.diagflat(x)
            result1 = paddle.diagflat(input=x)
            result3 = paddle.diagflat(x2)
            result4 = paddle.diagflat(input=x2, offset=-1)

            place = get_device_place() if use_gpu else paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup)
            res0, res1, res3, res4 = exe.run(
                main,
                feed={"input": self.input_np, 'input2': self.input_np2},
                fetch_list=[result0, result1, result3, result4],
            )

            np.testing.assert_allclose(res0, self.expected0, rtol=1e-05)
            np.testing.assert_allclose(res1, self.expected0, rtol=1e-05)
            np.testing.assert_allclose(res3, self.expected3, rtol=1e-05)
            np.testing.assert_allclose(res4, self.expected5, rtol=1e-05)

    def test_input_param_name_dtype(self):
        paddle.disable_static(place=paddle.CPUPlace())
        try:
            float_input = paddle.to_tensor([1, 2, 3], dtype='float32')
            float_result = paddle.diagflat(input=float_input)
            float_expected = np.diagflat(np.array([1, 2, 3], dtype=np.float32))
            np.testing.assert_allclose(float_result.numpy(), float_expected)
            self.assertEqual(float_result.dtype, paddle.float32)

            complex_input = paddle.to_tensor(
                np.array([1 + 0j, 2 + 0j], dtype=np.complex64)
            )
            complex_result = paddle.diagflat(input=complex_input)
            complex_expected = np.diagflat(
                np.array([1 + 0j, 2 + 0j], dtype=np.complex64)
            )
            np.testing.assert_allclose(
                complex_result.numpy(), complex_expected
            )
            self.assertEqual(complex_result.dtype, paddle.complex64)
        finally:
            paddle.enable_static()

    def test_input_param_name_scalar(self):
        paddle.disable_static(place=paddle.CPUPlace())
        try:
            scalar = paddle.to_tensor(7)
            result = paddle.diagflat(input=scalar)
            expected = np.diagflat(np.array(7))
            np.testing.assert_array_equal(result.numpy(), expected)
        finally:
            paddle.enable_static()

    def test_input_param_name_negative_offset(self):
        paddle.disable_static(place=paddle.CPUPlace())
        try:
            x = paddle.to_tensor([[1, 2], [3, 4]])
            result = paddle.diagflat(input=x, offset=-3)
            expected = np.diagflat(np.array([[1, 2], [3, 4]]), k=-3)
            np.testing.assert_array_equal(result.numpy(), expected)
        finally:
            paddle.enable_static()

    def test_input_param_name_conflict(self):
        t1 = paddle.to_tensor([1, 2])
        t2 = paddle.to_tensor([3, 4])
        with self.assertRaises(ValueError) as context:
            paddle.diagflat(x=t1, input=t2)
        self.assertIn(
            "Cannot specify both 'x' and its alias 'input'",
            str(context.exception),
        )

    def test_cpu(self):
        paddle.disable_static(place=paddle.CPUPlace())
        self.run_imperative()
        paddle.enable_static()

        with paddle.static.program_guard(Program()):
            self.run_static()

    def test_gpu(self):
        if not (paddle.is_compiled_with_cuda() or is_custom_device()):
            return

        paddle.disable_static(place=get_device_place())
        self.run_imperative()
        paddle.enable_static()

        with paddle.static.program_guard(Program()):
            self.run_static(use_gpu=True)

    def test_fp16_with_gpu(self, use_gpu=False):
        if paddle.base.core.is_compiled_with_cuda() or is_custom_device():
            place = get_device_place()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = np.random.random([10, 10]).astype("float16")
                x = paddle.static.data(
                    name="x", shape=[10, 10], dtype="float16"
                )

                y = paddle.diagflat(x)
                expected = np.diagflat(input)

                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "x": input,
                    },
                    fetch_list=[y],
                )


if __name__ == "__main__":
    unittest.main()
