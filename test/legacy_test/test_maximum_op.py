# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core


class ApiMaximumTest(unittest.TestCase):
    def setUp(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

        self.input_x = np.random.rand(10, 15).astype("float32")
        self.input_y = np.random.rand(10, 15).astype("float32")
        self.input_z = np.random.rand(15).astype("float32")
        self.input_a = np.array([0, np.nan, np.nan]).astype('int64')
        self.input_b = np.array([2, np.inf, -np.inf]).astype('int64')
        self.input_c = np.array([4, 1, 3]).astype('int64')

        self.input_nan_a = np.array([0, np.nan, np.nan]).astype('float32')
        self.input_nan_b = np.array([0, 1, 2]).astype('float32')

        self.np_expected1 = np.maximum(self.input_x, self.input_y)
        self.np_expected2 = np.maximum(self.input_x, self.input_z)
        self.np_expected3 = np.maximum(self.input_a, self.input_c)
        self.np_expected4 = np.maximum(self.input_b, self.input_c)
        self.np_expected_nan_aa = np.maximum(
            self.input_nan_a, self.input_nan_a
        )  # maximum(Nan, Nan)
        self.np_expected_nan_ab = np.maximum(
            self.input_nan_a, self.input_nan_b
        )  # maximum(Nan, Num)
        self.np_expected_nan_ba = np.maximum(
            self.input_nan_b, self.input_nan_a
        )  # maximum(Num, Nan)

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data_x = paddle.static.data("x", shape=[10, 15], dtype="float32")
            data_y = paddle.static.data("y", shape=[10, 15], dtype="float32")
            result_max = paddle.maximum(data_x, data_y)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={"x": self.input_x, "y": self.input_y},
                fetch_list=[result_max],
            )
        np.testing.assert_allclose(res, self.np_expected1, rtol=1e-05)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data_x = paddle.static.data("x", shape=[10, 15], dtype="float32")
            data_z = paddle.static.data("z", shape=[15], dtype="float32")
            result_max = paddle.maximum(data_x, data_z)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={"x": self.input_x, "z": self.input_z},
                fetch_list=[result_max],
            )
        np.testing.assert_allclose(res, self.np_expected2, rtol=1e-05)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data_a = paddle.static.data("a", shape=[3], dtype="int64")
            data_c = paddle.static.data("c", shape=[3], dtype="int64")
            result_max = paddle.maximum(data_a, data_c)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={"a": self.input_a, "c": self.input_c},
                fetch_list=[result_max],
            )
        np.testing.assert_allclose(res, self.np_expected3, rtol=1e-05)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data_b = paddle.static.data("b", shape=[3], dtype="int64")
            data_c = paddle.static.data("c", shape=[3], dtype="int64")
            result_max = paddle.maximum(data_b, data_c)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={"b": self.input_b, "c": self.input_c},
                fetch_list=[result_max],
            )
        np.testing.assert_allclose(res, self.np_expected4, rtol=1e-05)

    def test_dynamic_api(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.input_x)
        y = paddle.to_tensor(self.input_y)
        z = paddle.to_tensor(self.input_z)

        a = paddle.to_tensor(self.input_a)
        b = paddle.to_tensor(self.input_b)
        c = paddle.to_tensor(self.input_c)

        res = paddle.maximum(x, y)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected1, rtol=1e-05)

        # test broadcast
        res = paddle.maximum(x, z)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected2, rtol=1e-05)

        res = paddle.maximum(a, c)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected3, rtol=1e-05)

        res = paddle.maximum(b, c)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected4, rtol=1e-05)

    @unittest.skipIf(
        core.is_compiled_with_xpu(),
        "XPU need fix the bug",
    )
    def test_equal_tensors(self):
        numpy_tensor = np.ones([10000]).astype("float32")
        paddle_x = paddle.to_tensor(numpy_tensor)
        paddle_x.stop_gradient = False
        numpy_tensor = np.ones([10000]).astype("float32")
        paddle_x2 = paddle.to_tensor(numpy_tensor)
        paddle_x2.stop_gradient = False

        numpy_tensor = np.ones([10000]).astype("float32")
        paddle_outgrad = paddle.to_tensor(numpy_tensor)

        paddle_out = paddle.maximum(paddle_x, paddle_x2)
        paddle_x_grad, paddle_x2_grad = paddle.grad(
            [paddle_out],
            [paddle_x, paddle_x2],
            grad_outputs=[paddle_outgrad],
            allow_unused=True,
        )

        np.testing.assert_allclose(
            paddle_out.numpy(),
            numpy_tensor,
            1e-2,
            1e-2,
        )

        np.testing.assert_allclose(
            paddle_x_grad.numpy(),
            numpy_tensor * 0.5,
            1e-2,
            1e-2,
        )

        np.testing.assert_allclose(
            paddle_x2_grad.numpy(),
            numpy_tensor * 0.5,
            1e-2,
            1e-2,
        )

    @unittest.skipIf(
        core.is_compiled_with_xpu(),
        "XPU need fix the bug",
    )
    def test_dynamic_nan(self):
        with dygraph_guard():
            nan_a = paddle.to_tensor(self.input_nan_a)
            nan_b = paddle.to_tensor(self.input_nan_b)
            res = paddle.maximum(nan_a, nan_a)
            res = res.numpy()
            np.testing.assert_allclose(
                res, self.np_expected_nan_aa, rtol=1e-05, equal_nan=True
            )

            res = paddle.maximum(nan_a, nan_b)
            res = res.numpy()
            np.testing.assert_allclose(
                res, self.np_expected_nan_ab, rtol=1e-05, equal_nan=True
            )

            res = paddle.maximum(nan_b, nan_a)
            res = res.numpy()
            np.testing.assert_allclose(
                res, self.np_expected_nan_ba, rtol=1e-05, equal_nan=True
            )

    @unittest.skipIf(
        core.is_compiled_with_xpu(),
        "XPU need fix the bug",
    )
    def test_static_nan(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                data_a = paddle.static.data("a", shape=[3], dtype="float32")
                data_b = paddle.static.data("b", shape=[3], dtype="float32")
                result_max = paddle.maximum(data_a, data_b)
                exe = paddle.static.Executor(self.place)
                (res,) = exe.run(
                    feed={"a": self.input_nan_a, "b": self.input_nan_a},
                    fetch_list=[result_max],
                )
            np.testing.assert_allclose(
                res, self.np_expected_nan_aa, rtol=1e-05, equal_nan=True
            )

            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                data_a = paddle.static.data("a", shape=[3], dtype="float32")
                data_b = paddle.static.data("b", shape=[3], dtype="float32")
                result_max = paddle.maximum(data_a, data_b)
                exe = paddle.static.Executor(self.place)
                (res,) = exe.run(
                    feed={"a": self.input_nan_a, "b": self.input_nan_b},
                    fetch_list=[result_max],
                )
            np.testing.assert_allclose(
                res, self.np_expected_nan_ab, rtol=1e-05, equal_nan=True
            )

            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                data_a = paddle.static.data("a", shape=[3], dtype="float32")
                data_b = paddle.static.data("b", shape=[3], dtype="float32")
                result_max = paddle.maximum(data_a, data_b)
                exe = paddle.static.Executor(self.place)
                (res,) = exe.run(
                    feed={"a": self.input_nan_b, "b": self.input_nan_a},
                    fetch_list=[result_max],
                )
            np.testing.assert_allclose(
                res, self.np_expected_nan_ba, rtol=1e-05, equal_nan=True
            )

    def test_0size_input(self):
        numpy_tensor = np.ones([0, 1, 2]).astype("float32")
        paddle_x = paddle.to_tensor(numpy_tensor)
        paddle_x.stop_gradient = False
        numpy_tensor = np.ones([1, 3598, 2]).astype("float32")
        paddle_x2 = paddle.to_tensor(numpy_tensor)
        paddle_x2.stop_gradient = False

        numpy_tensor = np.ones([0, 3598, 2]).astype("float32")
        paddle_outgrad = paddle.to_tensor(numpy_tensor)

        paddle_out = paddle.maximum(paddle_x, paddle_x2)
        paddle_x_grad, paddle_x2_grad = paddle.grad(
            [paddle_out],
            [paddle_x, paddle_x2],
            grad_outputs=[paddle_outgrad],
            allow_unused=True,
        )

        np.testing.assert_allclose(
            paddle_out.numpy(),
            numpy_tensor,
            1e-2,
            1e-2,
        )

        numpy_tensor = np.ones([0, 1, 2]).astype("float32")

        np.testing.assert_allclose(
            paddle_x_grad.numpy(),
            numpy_tensor,
            1e-2,
            1e-2,
        )

        numpy_tensor = np.zeros([1, 3598, 2]).astype("float32")

        np.testing.assert_allclose(
            paddle_x2_grad.numpy(),
            numpy_tensor,
            1e-2,
            1e-2,
        )


if __name__ == '__main__':
    unittest.main()
