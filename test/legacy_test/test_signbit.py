# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np

import paddle
from paddle.base import core


def ref_np_signbit(x: np.ndarray):
    return np.signbit(x)


class TestSignbitAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.cuda_support_dtypes = [
            'float32',
            'float64',
            'uint8',
            'int8',
            'int16',
            'int32',
            'int64',
        ]
        self.cpu_support_dtypes = [
            'float32',
            'float64',
            'uint8',
            'int8',
            'int16',
            'int32',
            'int64',
        ]
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_dtype(self):
        def run(place):
            paddle.disable_static(place)
            if core.is_compiled_with_cuda():
                support_dtypes = self.cuda_support_dtypes
            else:
                support_dtypes = self.cpu_support_dtypes

            for dtype in support_dtypes:
                x = paddle.to_tensor(
                    np.random.randint(-10, 10, size=[12, 20, 2]).astype(dtype)
                )
                paddle.signbit(x)

        for place in self.place:
            run(place)

    def test_float(self):
        def run(place):
            paddle.disable_static(place)
            if core.is_compiled_with_cuda():
                support_dtypes = self.cuda_support_dtypes
            else:
                support_dtypes = self.cpu_support_dtypes

            for dtype in support_dtypes:
                np_x = np.random.randint(-10, 10, size=[12, 20, 2]).astype(
                    dtype
                )
                x = paddle.to_tensor(np_x)
                out = paddle.signbit(x)
                np_out = out.numpy()
                out_expected = ref_np_signbit(np_x)
                np.testing.assert_allclose(np_out, out_expected, rtol=1e-05)

        for place in self.place:
            run(place)

    def test_input_type(self):
        with self.assertRaises(TypeError):
            x = np.random.randint(-10, 10, size=[12, 20, 2]).astype('float32')
            x = paddle.signbit(x)

    def test_Tensor_dtype(self):
        def run(place):
            paddle.disable_static(place)
            if core.is_compiled_with_cuda():
                support_dtypes = self.cuda_support_dtypes
            else:
                support_dtypes = self.cpu_support_dtypes

            for dtype in support_dtypes:
                x = paddle.to_tensor(
                    np.random.randint(-10, 10, size=[12, 20, 2]).astype(dtype)
                )
                x.signbit()

        for place in self.place:
            run(place)

    def test_static(self):
        np_input1 = np.random.uniform(-10, 10, (12, 10)).astype("int8")
        np_input2 = np.random.uniform(-10, 10, (12, 10)).astype("uint8")
        np_input3 = np.random.uniform(-10, 10, (12, 10)).astype("int16")
        np_input4 = np.random.uniform(-10, 10, (12, 10)).astype("int32")
        np_input5 = np.random.uniform(-10, 10, (12, 10)).astype("int64")
        np_input6 = np.array([-0.0, 0.0]).astype("float32")
        np_input7 = np.array([-0.0, 0.0]).astype("float64")
        np_out1 = np.signbit(np_input1)
        np_out2 = np.signbit(np_input2)
        np_out3 = np.signbit(np_input3)
        np_out4 = np.signbit(np_input4)
        np_out5 = np.signbit(np_input5)
        np_out6 = np.signbit(np_input6)
        np_out7 = np.signbit(np_input7)

        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                # The input type of sign_op must be Variable or numpy.ndarray.
                input1 = 12
                self.assertRaises(TypeError, paddle.tensor.math.sign, input1)
                # The result of sign_op must correct.
                input1 = paddle.static.data(
                    name='input1', shape=[12, 10], dtype="int8"
                )
                input2 = paddle.static.data(
                    name='input2', shape=[12, 10], dtype="uint8"
                )
                input3 = paddle.static.data(
                    name='input3', shape=[12, 10], dtype="int16"
                )
                input4 = paddle.static.data(
                    name='input4', shape=[12, 10], dtype="int32"
                )
                input5 = paddle.static.data(
                    name='input5', shape=[12, 10], dtype="int64"
                )
                input6 = paddle.static.data(
                    name='input6', shape=[2], dtype="float32"
                )
                input7 = paddle.static.data(
                    name='input7', shape=[2], dtype="float64"
                )
                out1 = paddle.signbit(input1)
                out2 = paddle.signbit(input2)
                out3 = paddle.signbit(input3)
                out4 = paddle.signbit(input4)
                out5 = paddle.signbit(input5)
                out6 = paddle.signbit(input6)
                out7 = paddle.signbit(input7)
                exe = paddle.static.Executor(place)
                res1, res2, res3, res4, res5, res6, res7 = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "input1": np_input1,
                        "input2": np_input2,
                        "input3": np_input3,
                        "input4": np_input4,
                        "input5": np_input5,
                        "input6": np_input6,
                        "input7": np_input7,
                    },
                    fetch_list=[out1, out2, out3, out4, out5, out6, out7],
                )
                self.assertEqual((res1 == np_out1).all(), True)
                self.assertEqual((res2 == np_out2).all(), True)
                self.assertEqual((res3 == np_out3).all(), True)
                self.assertEqual((res4 == np_out4).all(), True)
                self.assertEqual((res5 == np_out5).all(), True)
                self.assertEqual((res6 == np_out6).all(), True)
                self.assertEqual((res7 == np_out7).all(), True)

        for place in self.place:
            run(place)


if __name__ == "__main__":
    unittest.main()
