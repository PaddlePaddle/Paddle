#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import Program, program_guard

paddle.enable_static()


class TestBroadcastToError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            x1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.CPUPlace()
            )
            shape = [2, 2]
            self.assertRaises(TypeError, paddle.tensor.broadcast_to, x1, shape)
            x2 = paddle.static.data(name='x2', shape=[-1, 4], dtype="uint8")
            self.assertRaises(TypeError, paddle.tensor.broadcast_to, x2, shape)
            x3 = paddle.static.data(name='x3', shape=[-1, 4], dtype="bool")
            x3.stop_gradient = False
            self.assertRaises(ValueError, paddle.tensor.broadcast_to, x3, shape)


# Test python API
class TestBroadcastToAPI(unittest.TestCase):
    def test_api(self):
        input = np.random.random([12, 14]).astype("float32")
        x = paddle.static.data(name='x', shape=[12, 14], dtype="float32")

        positive_2 = paddle.tensor.fill_constant([1], "int32", 12)
        expand_shape = paddle.static.data(
            name="expand_shape",
            shape=[2],
            dtype="int32",
        )

        out_1 = paddle.broadcast_to(x, shape=[12, 14])
        out_2 = paddle.broadcast_to(x, shape=[positive_2, 14])
        out_3 = paddle.broadcast_to(x, shape=expand_shape)

        g0 = base.backward.calc_gradient(out_2, x)

        exe = base.Executor(place=base.CPUPlace())
        res_1, res_2, res_3 = exe.run(
            base.default_main_program(),
            feed={
                "x": input,
                "expand_shape": np.array([12, 14]).astype("int32"),
            },
            fetch_list=[out_1, out_2, out_3],
        )
        np.testing.assert_array_equal(res_1, np.tile(input, (1, 1)))
        np.testing.assert_array_equal(res_2, np.tile(input, (1, 1)))
        np.testing.assert_array_equal(res_3, np.tile(input, (1, 1)))

    def test_api_fp16_gpu(self):
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = np.random.random([12, 14]).astype("float16")
                x = paddle.static.data(
                    name="x", shape=[12, 14], dtype="float16"
                )

                positive_2 = paddle.tensor.fill_constant([1], "int32", 12)
                expand_shape = paddle.static.data(
                    name="expand_shape",
                    shape=[2],
                    dtype="int32",
                )

                out_1 = paddle.broadcast_to(x, shape=[12, 14])
                out_2 = paddle.broadcast_to(x, shape=[positive_2, 14])
                out_3 = paddle.broadcast_to(x, shape=expand_shape)

                exe = paddle.static.Executor(place)
                res_1, res_2, res_3 = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "x": input,
                        "expand_shape": np.array([12, 14]).astype("int32"),
                    },
                    fetch_list=[out_1, out_2, out_3],
                )
                np.testing.assert_array_equal(res_1, np.tile(input, (1, 1)))
                np.testing.assert_array_equal(res_2, np.tile(input, (1, 1)))
                np.testing.assert_array_equal(res_3, np.tile(input, (1, 1)))


if __name__ == "__main__":
    unittest.main()
