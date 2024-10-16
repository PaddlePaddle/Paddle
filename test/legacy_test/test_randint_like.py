# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


# Test python API
class TestRandintLikeAPI(unittest.TestCase):
    def setUp(self):
        self.x_bool = np.zeros((10, 12)).astype("bool")
        self.x_int32 = np.zeros((10, 12)).astype("int32")
        self.x_int64 = np.zeros((10, 12)).astype("int64")
        self.x_float16 = np.zeros((10, 12)).astype("float16")
        self.x_float32 = np.zeros((10, 12)).astype("float32")
        self.x_float64 = np.zeros((10, 12)).astype("float64")

        self.dtype = ["bool", "int32", "int64", "float16", "float32", "float64"]
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # results are from [-100, 100).
            x_bool = paddle.static.data(
                name="x_bool", shape=[10, 12], dtype="bool"
            )
            exe = paddle.static.Executor(self.place)
            # x dtype is bool output dtype in ["bool", "int32", "int64", "float16", "float32", "float64"]
            outlist1 = [
                paddle.randint_like(x_bool, low=-10, high=10, dtype=dtype)
                for dtype in self.dtype
            ]
            outs1 = exe.run(feed={'x_bool': self.x_bool}, fetch_list=[outlist1])
            for out, dtype in zip(outs1, self.dtype):
                self.assertTrue(out.dtype, np.dtype(dtype))
                self.assertTrue(((out >= -10) & (out <= 10)).all(), True)
        paddle.disable_static()

    def test_static_api_with_int32(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x_int32 = paddle.static.data(
                name="x_int32", shape=[10, 12], dtype="int32"
            )
            exe = paddle.static.Executor(self.place)
            # x dtype is int32 output dtype in ["bool", "int32", "int64", "float16", "float32", "float64"]
            outlist2 = [
                paddle.randint_like(x_int32, low=-5, high=10, dtype=dtype)
                for dtype in self.dtype
            ]
            outs2 = exe.run(
                paddle.static.default_main_program(),
                feed={'x_int32': np.zeros((10, 12)).astype(np.int32)},
                fetch_list=[outlist2],
            )
            for out2, dtype in zip(outs2, self.dtype):
                self.assertTrue(out2.dtype, np.dtype(dtype))
                self.assertTrue(((out2 >= -5) & (out2 <= 10)).all(), True)
        paddle.disable_static()

    def test_static_api_with_int64(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x_int64 = paddle.static.data(
                name="x_int64", shape=[10, 12], dtype="int64"
            )
            exe = paddle.static.Executor(self.place)
            # x dtype is int64 output dtype in ["bool", "int32", "int64", "float16", "float32", "float64"]
            outlist3 = [
                paddle.randint_like(x_int64, low=-100, high=100, dtype=dtype)
                for dtype in self.dtype
            ]
            outs3 = exe.run(feed={'x_int64': self.x_int64}, fetch_list=outlist3)
            for out, dtype in zip(outs3, self.dtype):
                self.assertTrue(out.dtype, np.dtype(dtype))
                self.assertTrue(((out >= -100) & (out <= 100)).all(), True)
        paddle.disable_static()

    def test_static_api_with_fp16(self):
        paddle.enable_static()
        if paddle.is_compiled_with_cuda():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x_float16 = paddle.static.data(
                    name="x_float16", shape=[10, 12], dtype="float16"
                )
                exe = paddle.static.Executor(self.place)
                # x dtype is float16 output dtype in ["bool", "int32", "int64", "float16", "float32", "float64"]
                outlist4 = [
                    paddle.randint_like(x_float16, low=-3, high=25, dtype=dtype)
                    for dtype in self.dtype
                ]
                outs4 = exe.run(
                    feed={'x_float16': self.x_float16}, fetch_list=outlist4
                )
                for out, dtype in zip(outs4, self.dtype):
                    self.assertTrue(out.dtype, np.dtype(dtype))
                    self.assertTrue(((out >= -3) & (out <= 25)).all(), True)
        paddle.disable_static()

    def test_static_api_with_float32(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x_float32 = paddle.static.data(
                name="x_float32", shape=[10, 12], dtype="float32"
            )
            exe = paddle.static.Executor(self.place)
            # x dtype is float32 output dtype in ["bool", "int32", "int64", "float16", "float32", "float64"]
            outlist5 = [
                paddle.randint_like(x_float32, low=-25, high=25, dtype=dtype)
                for dtype in self.dtype
            ]
            outs5 = exe.run(
                feed={'x_float32': self.x_float32}, fetch_list=outlist5
            )
            for out, dtype in zip(outs5, self.dtype):
                self.assertTrue(out.dtype, np.dtype(dtype))
                self.assertTrue(((out >= -25) & (out <= 25)).all(), True)
        paddle.disable_static()

    def test_static_api_with_float64(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x_float64 = paddle.static.data(
                name="x_float64", shape=[10, 12], dtype="float64"
            )
            exe = paddle.static.Executor(self.place)
            # x dtype is float64 output dtype in ["bool", "int32", "int64", "float16", "float32", "float64"]
            outlist6 = [
                paddle.randint_like(x_float64, low=-16, high=16, dtype=dtype)
                for dtype in self.dtype
            ]
            outs6 = exe.run(
                feed={'x_float64': self.x_float64}, fetch_list=outlist6
            )
            for out, dtype in zip(outs6, self.dtype):
                self.assertTrue(out.dtype, dtype)
                self.assertTrue(((out >= -16) & (out <= 16)).all(), True)
        paddle.disable_static()

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        # x dtype ["bool", "int32", "int64", "float32", "float64"]
        for x in [
            self.x_bool,
            self.x_int32,
            self.x_int64,
            self.x_float32,
            self.x_float64,
        ]:
            x_inputs = paddle.to_tensor(x)
            # self.dtype ["bool", "int32", "int64", "float16", "float32", "float64"]
            for dtype in self.dtype:
                out = paddle.randint_like(
                    x_inputs, low=-100, high=100, dtype=dtype
                )
                self.assertTrue(out.numpy().dtype, np.dtype(dtype))
                self.assertTrue(
                    ((out.numpy() >= -100) & (out.numpy() <= 100)).all(), True
                )
        # x dtype ["float16"]
        if paddle.is_compiled_with_cuda():
            x_inputs = paddle.to_tensor(self.x_float16)
            # self.dtype ["bool", "int32", "int64", "float16", "float32", "float64"]
            for dtype in self.dtype:
                out = paddle.randint_like(
                    x_inputs, low=-100, high=100, dtype=dtype
                )
                self.assertTrue(out.numpy().dtype, np.dtype(dtype))
                self.assertTrue(
                    ((out.numpy() >= -100) & (out.numpy() <= 100)).all(), True
                )
        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x_bool = paddle.static.data(
                name="x_bool", shape=[10, 12], dtype="bool"
            )
            x_int32 = paddle.static.data(
                name="x_int32", shape=[10, 12], dtype="int32"
            )
            x_int64 = paddle.static.data(
                name="x_int64", shape=[10, 12], dtype="int64"
            )
            x_float16 = paddle.static.data(
                name="x_float16", shape=[10, 12], dtype="float16"
            )
            x_float32 = paddle.static.data(
                name="x_float32", shape=[10, 12], dtype="float32"
            )
            x_float64 = paddle.static.data(
                name="x_float64", shape=[10, 12], dtype="float64"
            )

            # x dtype is bool
            # low is 5 and high is 5, low must less then high
            self.assertRaises(
                ValueError, paddle.randint_like, x_bool, low=5, high=5
            )
            # low(default value) is 0 and high is -5, low must less then high
            self.assertRaises(ValueError, paddle.randint_like, x_bool, high=-5)
            # if high is None, low must be greater than 0
            self.assertRaises(ValueError, paddle.randint_like, x_bool, low=-5)

            # x dtype is int32
            # low is 5 and high is 5, low must less then high
            self.assertRaises(
                ValueError, paddle.randint_like, x_int32, low=5, high=5
            )
            # low(default value) is 0 and high is -5, low must less then high
            self.assertRaises(ValueError, paddle.randint_like, x_int32, high=-5)
            # if high is None, low must be greater than 0
            self.assertRaises(ValueError, paddle.randint_like, x_int32, low=-5)

            # x dtype is int64
            # low is 5 and high is 5, low must less then high
            self.assertRaises(
                ValueError, paddle.randint_like, x_int64, low=5, high=5
            )
            # low(default value) is 0 and high is -5, low must less then high
            self.assertRaises(ValueError, paddle.randint_like, x_int64, high=-5)
            # if high is None, low must be greater than 0
            self.assertRaises(ValueError, paddle.randint_like, x_int64, low=-5)

            # x dtype is float16
            # low is 5 and high is 5, low must less then high
            if paddle.is_compiled_with_cuda():
                self.assertRaises(
                    ValueError, paddle.randint_like, x_float16, low=5, high=5
                )
                # low(default value) is 0 and high is -5, low must less then high
                self.assertRaises(
                    ValueError, paddle.randint_like, x_float16, high=-5
                )
                # if high is None, low must be greater than 0
                self.assertRaises(
                    ValueError, paddle.randint_like, x_float16, low=-5
                )

            # x dtype is float32
            # low is 5 and high is 5, low must less then high
            self.assertRaises(
                ValueError, paddle.randint_like, x_float32, low=5, high=5
            )
            # low(default value) is 0 and high is -5, low must less then high
            self.assertRaises(
                ValueError, paddle.randint_like, x_float32, high=-5
            )
            # if high is None, low must be greater than 0
            self.assertRaises(
                ValueError, paddle.randint_like, x_float32, low=-5
            )

            # x dtype is float64
            # low is 5 and high is 5, low must less then high
            self.assertRaises(
                ValueError, paddle.randint_like, x_float64, low=5, high=5
            )
            # low(default value) is 0 and high is -5, low must less then high
            self.assertRaises(
                ValueError, paddle.randint_like, x_float64, high=-5
            )
            # if high is None, low must be greater than 0
            self.assertRaises(
                ValueError, paddle.randint_like, x_float64, low=-5
            )


if __name__ == "__main__":
    unittest.main()
