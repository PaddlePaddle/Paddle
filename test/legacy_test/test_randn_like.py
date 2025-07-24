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
from utils import dygraph_guard, static_guard

import paddle


# Test python API
class TestRandnLikeAPI(unittest.TestCase):
    def setUp(self):
        self.x_float16 = np.zeros((10, 12)).astype("float16")
        self.x_float32 = np.zeros((10, 12)).astype("float32")
        self.x_float64 = np.zeros((10, 12)).astype("float64")

        self.dtype = ["float16", "float32", "float64"]
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with (
            static_guard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            x_float32 = paddle.static.data(
                name="x_float32", shape=[10, 12], dtype="float32"
            )
            exe = paddle.static.Executor(self.place)
            outlist = [paddle.randn_like(x_float32)]
            outs = exe.run(
                feed={'x_float32': self.x_float32}, fetch_list=outlist
            )
            for out, dtype in zip(outs, self.dtype):
                self.assertTrue(out.dtype, np.dtype(dtype))
                self.assertTrue(((out >= -25) & (out <= 25)).all(), True)

    def test_static_api_with_fp16(self):
        with static_guard():
            if paddle.is_compiled_with_cuda():
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x_float16 = paddle.static.data(
                        name="x_float16", shape=[10, 12], dtype="float16"
                    )
                    exe = paddle.static.Executor(self.place)
                    outlist1 = [
                        paddle.randn_like(x_float16, dtype=dtype)
                        for dtype in self.dtype
                    ]
                    outs1 = exe.run(
                        feed={'x_float16': self.x_float16}, fetch_list=outlist1
                    )
                    for out, dtype in zip(outs1, self.dtype):
                        self.assertTrue(out.dtype, np.dtype(dtype))
                        self.assertTrue(
                            ((out >= -25) & (out <= 25)).all(), True
                        )

    def test_static_api_with_fp32(self):
        with (
            static_guard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            x_float32 = paddle.static.data(
                name="x_float32", shape=[10, 12], dtype="float32"
            )
            exe = paddle.static.Executor(self.place)
            outlist2 = [
                paddle.randn_like(x_float32, dtype=dtype)
                for dtype in self.dtype
            ]
            outs2 = exe.run(
                feed={'x_float32': self.x_float32}, fetch_list=outlist2
            )
            for out, dtype in zip(outs2, self.dtype):
                self.assertTrue(out.dtype, np.dtype(dtype))
                self.assertTrue(((out >= -25) & (out <= 25)).all(), True)

    def test_static_api_with_fp64(self):
        with (
            static_guard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            x_float64 = paddle.static.data(
                name="x_float64", shape=[10, 12], dtype="float64"
            )
            exe = paddle.static.Executor(self.place)
            outlist3 = [
                paddle.randn_like(x_float64, dtype=dtype)
                for dtype in self.dtype
            ]
            outs3 = exe.run(
                feed={'x_float64': self.x_float64}, fetch_list=outlist3
            )
            for out, dtype in zip(outs3, self.dtype):
                self.assertTrue(out.dtype, dtype)
                self.assertTrue(((out >= -25) & (out <= 25)).all(), True)

    def test_dygraph_api(self):
        with dygraph_guard():
            for x in [
                self.x_float32,
                self.x_float64,
            ]:
                x_inputs = paddle.to_tensor(x, place=self.place)
                for dtype in self.dtype:
                    out = paddle.randn_like(x_inputs, dtype=dtype)
                    self.assertTrue(out.numpy().dtype, np.dtype(dtype))
                    self.assertTrue(
                        ((out.numpy() >= -25) & (out.numpy() <= 25)).all(), True
                    )

            x_inputs = paddle.to_tensor(self.x_float32)
            out = paddle.randn_like(x_inputs)
            self.assertTrue(out.numpy().dtype, np.dtype("float32"))
            self.assertTrue(
                ((out.numpy() >= -25) & (out.numpy() <= 25)).all(), True
            )

            if paddle.is_compiled_with_cuda():
                x_inputs = paddle.to_tensor(self.x_float16)
                for dtype in self.dtype:
                    out = paddle.randn_like(x_inputs, dtype=dtype)
                    self.assertTrue(out.numpy().dtype, np.dtype(dtype))
                    self.assertTrue(
                        ((out.numpy() >= -25) & (out.numpy() <= 25)).all(), True
                    )


if __name__ == "__main__":
    unittest.main()
