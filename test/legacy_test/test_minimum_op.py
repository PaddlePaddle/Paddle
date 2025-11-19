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
from op_test import get_device_place, is_custom_device
from utils import dygraph_guard, static_guard

import paddle
from paddle.base import core


class ApiMinimumTest(unittest.TestCase):
    def setUp(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            self.place = get_device_place()
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

        self.np_expected1 = np.minimum(self.input_x, self.input_y)
        self.np_expected2 = np.minimum(self.input_x, self.input_z)
        self.np_expected3 = np.minimum(self.input_a, self.input_c)
        self.np_expected4 = np.minimum(self.input_b, self.input_c)
        self.np_expected_nan_aa = np.minimum(
            self.input_nan_a, self.input_nan_a
        )  # minimum(Nan, Nan)
        self.np_expected_nan_ab = np.minimum(
            self.input_nan_a, self.input_nan_b
        )  # minimum(Nan, Num)
        self.np_expected_nan_ba = np.minimum(
            self.input_nan_b, self.input_nan_a
        )  # minimum(Num, Nan)

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data_x = paddle.static.data("x", shape=[10, 15], dtype="float32")
            data_y = paddle.static.data("y", shape=[10, 15], dtype="float32")
            result_max = paddle.minimum(data_x, data_y)
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
            result_max = paddle.minimum(data_x, data_z)
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
            result_max = paddle.minimum(data_a, data_c)
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
            result_max = paddle.minimum(data_b, data_c)
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

        res = paddle.minimum(x, y)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected1, rtol=1e-05)

        # test broadcast
        res = paddle.minimum(x, z)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected2, rtol=1e-05)

        res = paddle.minimum(a, c)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected3, rtol=1e-05)

        res = paddle.minimum(b, c)
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

        paddle_out = paddle.minimum(paddle_x, paddle_x2)
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

            res = paddle.minimum(nan_a, nan_a)
            res = res.numpy()
            np.testing.assert_allclose(
                res, self.np_expected_nan_aa, rtol=1e-05, equal_nan=True
            )

            res = paddle.minimum(nan_a, nan_b)
            res = res.numpy()
            np.testing.assert_allclose(
                res, self.np_expected_nan_ab, rtol=1e-05, equal_nan=True
            )

            res = paddle.minimum(nan_b, nan_a)
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
                result_max = paddle.minimum(data_a, data_b)
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
                result_max = paddle.minimum(data_a, data_b)
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
                result_max = paddle.minimum(data_a, data_b)
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

        paddle_out = paddle.minimum(paddle_x, paddle_x2)
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


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestElementwiseMinimumOp_Stride(unittest.TestCase):
    def setUp(self):
        self.python_api = paddle.minimum
        self.public_python_api = paddle.minimum
        self.place = get_device_place()

    def init_dtype(self):
        self.dtype = np.float64

    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.minimum(self.x, self.y)
        self.perm = [1, 0]
        self.y_trans = np.transpose(self.y, self.perm)

    def test_dynamic_api(self):
        self.init_dtype()
        self.init_input_output()
        paddle.disable_static()
        self.y_trans = paddle.to_tensor(self.y_trans, place=self.place)
        self.x = paddle.to_tensor(self.x, place=self.place)
        self.y = paddle.to_tensor(self.y, place=self.place)
        if self.strided_input_type == "transpose":
            y_trans_tmp = paddle.transpose(self.y_trans, self.perm)
        elif self.strided_input_type == "as_stride":
            y_trans_tmp = paddle.as_strided(
                self.y_trans, self.shape_param, self.stride_param
            )
        else:
            raise TypeError(f"Unsupported test type {self.strided_input_type}.")
        res = paddle.minimum(self.x, y_trans_tmp)
        res = res.numpy()
        np.testing.assert_allclose(res, self.out, rtol=1e-05)


class TestElementwiseMinimumOp_Stride1(TestElementwiseMinimumOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.out = np.minimum(self.x, self.y)
        self.perm = [0, 1, 3, 2]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseMinimumOp_Stride2(TestElementwiseMinimumOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.out = np.minimum(self.x, self.y)
        self.perm = [0, 2, 1, 3]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseMinimumOp_Stride3(TestElementwiseMinimumOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [20, 2, 13, 1]).astype(self.dtype)
        self.out = np.minimum(self.x, self.y)
        self.perm = [0, 1, 3, 2]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseMinimumOp_Stride4(TestElementwiseMinimumOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [1, 2, 13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [20, 2, 13, 1]).astype(self.dtype)
        self.out = np.minimum(self.x, self.y)
        self.perm = [1, 0, 2, 3]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseMinimumOp_Stride5(TestElementwiseMinimumOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "as_stride"
        self.x = np.random.uniform(0.1, 1, [23, 10, 1, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [23, 2, 13, 20]).astype(self.dtype)
        self.y_trans = self.y
        self.y = self.y[:, 0:1, :, 0:1]
        self.out = np.minimum(self.x, self.y)
        self.shape_param = [23, 1, 13, 1]
        self.stride_param = [520, 260, 20, 1]


class TestElementwiseMinimumOp_Stride_ZeroDim1(TestElementwiseMinimumOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.minimum(self.x, self.y)
        self.perm = [1, 0]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseMinimumOp_Stride_ZeroSize1(
    TestElementwiseMinimumOp_Stride
):
    def init_data(self):
        self.strided_input_type = "transpose"
        self.x = np.random.rand(1, 0, 2).astype('float32')
        self.y = np.random.rand(3, 0, 1).astype('float32')
        self.out = np.minimum(self.x, self.y)
        self.perm = [2, 1, 0]
        self.y_trans = np.transpose(self.y, self.perm)


if __name__ == '__main__':
    unittest.main()
