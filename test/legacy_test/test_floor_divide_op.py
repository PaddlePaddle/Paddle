#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base, static


def get_places():
    places = []
    if base.is_compiled_with_cuda():
        places.append(paddle.CUDAPlace(0))
    places.append(paddle.CPUPlace())
    return places


class TestFloorDivideAPI_Compatibility(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        for p in get_places():
            for dtype in (
                'int8',
                'int16',
                'int32',
                'int64',
                'float16',
                'float32',
                'float64',
            ):
                np_x = np.array([2, 3, 8, 7]).astype(dtype)
                np_y = np.array([1, 5, 3, 3]).astype(dtype)
                out_expected = np.floor_divide(np_x, np_y)
                x = paddle.to_tensor(np_x)
                y = paddle.to_tensor(np_y)
                paddle_dygraph_out = []

                out1 = paddle.floor_divide(x, y)
                paddle_dygraph_out.append(out1)

                out2 = paddle.floor_divide(x=x, y=y)
                paddle_dygraph_out.append(out2)

                out3 = paddle.floor_divide(input=x, other=y)
                paddle_dygraph_out.append(out3)

                out5 = paddle.empty(
                    out_expected.shape, dtype=out_expected.dtype
                )
                out4 = paddle.floor_divide(x, y, out=out5)
                paddle_dygraph_out.append(out4)
                paddle_dygraph_out.append(out5)

                for out in paddle_dygraph_out:
                    self.assertEqual((out == out_expected).all(), True)

            for dtype in (
                'int8',
                'int16',
                'int32',
                'int64',
                'float16',
                'float32',
                'float64',
            ):
                np_x = np.array([2, 3, 8, 7]).astype(dtype)
                y_number = 2.0
                out_expected = np.floor_divide(np_x, y_number)
                x = paddle.to_tensor(np_x)
                paddle_dygraph_out = []

                out1 = paddle.floor_divide(x, y_number)
                paddle_dygraph_out.append(out1)

                out2 = paddle.floor_divide(x=x, y=y_number)
                paddle_dygraph_out.append(out2)

                out3 = paddle.floor_divide(input=x, other=y_number)
                paddle_dygraph_out.append(out3)

                out5 = paddle.empty(
                    out_expected.shape, dtype=out_expected.dtype
                )
                out4 = paddle.floor_divide(x, y_number, out=out5)
                paddle_dygraph_out.append(out4)
                paddle_dygraph_out.append(out5)

                for out in paddle_dygraph_out:
                    self.assertEqual((out == out_expected).all(), True)

        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()
        for p in get_places():
            for dtype in (
                'int32',
                'int64',
                'float16',
                'float32',
                'float64',
            ):
                np_x = np.array([2, 3, 8, 7]).astype(dtype)
                np_y = np.array([1, 5, 3, 3]).astype(dtype)
                out_expected = np.floor_divide(np_x, np_y)
                mp, sp = static.Program(), static.Program()
                with static.program_guard(mp, sp):
                    x = static.data("x", shape=[4], dtype=dtype)
                    y = static.data("y", shape=[4], dtype=dtype)
                    out1 = paddle.floor_divide(x, y)
                    out2 = paddle.floor_divide(x=x, y=y)
                    out3 = paddle.floor_divide(input=x, other=y)
                exe = static.Executor(p)
                exe.run(sp)
                fetches = exe.run(
                    mp,
                    feed={"x": np_x, "y": np_y},
                    fetch_list=[out1, out2, out3],
                )
                for out in fetches:
                    self.assertEqual((out == out_expected).all(), True)

            for dtype in (
                'int32',
                'int64',
                'float16',
                'float32',
                'float64',
            ):
                np_x = np.array([2, 3, 8, 7]).astype(dtype)
                y_number = 2.0
                out_expected = np.floor_divide(np_x, y_number)
                mp, sp = static.Program(), static.Program()
                with static.program_guard(mp, sp):
                    x = static.data("x", shape=[4], dtype=dtype)
                    out1 = paddle.floor_divide(x, y_number)
                    out2 = paddle.floor_divide(x=x, y=y_number)
                    out3 = paddle.floor_divide(input=x, other=y_number)
                exe = static.Executor(p)
                exe.run(sp)
                fetches = exe.run(
                    mp,
                    feed={"x": np_x, "y": y_number},
                    fetch_list=[out1, out2, out3],
                )
                for out in fetches:
                    self.assertEqual((out == out_expected).all(), True)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
