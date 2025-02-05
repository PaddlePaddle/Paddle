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
from op_test import OpTest, convert_float_to_uint16
from utils import static_guard

import paddle
from paddle import base
from paddle.base import core


class TestRollOp(OpTest):
    def setUp(self):
        self.python_api = paddle.roll
        self.op_type = "roll"
        self.public_python_api = paddle.roll
        self.prim_op_type = "prim"
        self.init_dtype_type()
        self.attrs = {'shifts': self.shifts, 'axis': self.axis}
        bf16_ut = self.dtype == np.uint16
        x = np.random.random(self.x_shape).astype(
            np.float32 if bf16_ut else self.dtype
        )
        out = np.roll(x, self.attrs['shifts'], self.attrs['axis'])
        if bf16_ut:
            x = convert_float_to_uint16(x)
            out = convert_float_to_uint16(out)
        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        self.dtype = np.float64
        self.x_shape = (100, 4, 5)
        self.shifts = [101, -1]
        self.axis = [0, -2]

    def test_check_output(self):
        self.check_output(
            check_prim=True, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad_normal(self):
        self.check_grad(
            ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True
        )


class TestRollOpCase2(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.float32
        self.x_shape = (100, 10, 5)
        self.shifts = [8, -1]
        self.axis = [-1, -2]


class TestRollOpCase3(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.float32
        self.x_shape = (11, 11)
        self.shifts = [1, 1]
        self.axis = [-1, 1]


class TestRollFP16OP(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.float16
        self.x_shape = (100, 4, 5)
        self.shifts = [101, -1]
        self.axis = [0, -2]


class TestRollFP16OpCase2(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.float16
        self.x_shape = (100, 10, 5)
        self.shifts = [8, -1]
        self.axis = [-1, -2]


class TestRollFP16OpCase3(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.float16
        self.x_shape = (11, 11)
        self.shifts = [1, 1]
        self.axis = [-1, 1]


class TestRollBollOp(OpTest):
    def setUp(self):
        self.python_api = paddle.roll
        self.op_type = "roll"
        self.public_python_api = paddle.roll
        self.prim_op_type = "prim"
        self.init_dtype_type()
        self.attrs = {'shifts': self.shifts, 'axis': self.axis}
        x = np.random.random(self.x_shape).astype(self.dtype)
        out = np.roll(x, self.attrs['shifts'], self.attrs['axis'])
        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        self.dtype = np.bool_
        self.x_shape = (100, 4, 5)
        self.shifts = [101, -1]
        self.axis = [0, -2]

    def test_check_output(self):
        self.check_output(
            check_prim=True, check_pir=True, check_symbol_infer=True
        )


class TestRollBoolOpCase2(TestRollBollOp):
    def init_dtype_type(self):
        self.dtype = np.bool_
        self.x_shape = (100, 10, 5)
        self.shifts = [8, -1]
        self.axis = [-1, -2]


class TestRollBoolOpCase3(TestRollBollOp):
    def init_dtype_type(self):
        self.dtype = np.bool_
        self.x_shape = (11, 11)
        self.shifts = [1, 1]
        self.axis = [-1, 1]


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestRollBF16OP(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.uint16
        self.x_shape = (10, 4, 5)
        self.shifts = [101, -1]
        self.axis = [0, -2]
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        self.check_output_with_place(
            self.place, check_prim=True, check_pir=True
        )

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place, ['X'], 'Out', check_prim=True, check_pir=True
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestRollBF16OpCase2(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.uint16
        self.x_shape = (10, 5, 5)
        self.shifts = [8, -1]
        self.axis = [-1, -2]
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        self.check_output_with_place(
            self.place, check_prim=True, check_pir=True
        )

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place,
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestRollBF16OpCase3(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.uint16
        self.x_shape = (11, 11)
        self.shifts = [1, 1]
        self.axis = [-1, 1]
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        self.check_output_with_place(
            self.place, check_prim=True, check_pir=True
        )

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place,
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestRollAPI(unittest.TestCase):
    def input_data(self):
        self.data_x = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )

    def test_roll_op_api_case1(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[-1, 3], dtype='float32')
                data_x = np.array(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
                ).astype('float32')
                z = paddle.roll(x, shifts=1)
                exe = paddle.static.Executor(paddle.CPUPlace())
                (res,) = exe.run(
                    paddle.static.default_main_program(),
                    feed={'x': data_x},
                    fetch_list=[z],
                    return_numpy=False,
                )
                expect_out = np.array(
                    [[9.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
                )
            np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_roll_op_api_case2(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[-1, 3], dtype='float32')
                data_x = np.array(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
                ).astype('float32')
                z = paddle.roll(x, shifts=1, axis=0)
                exe = paddle.static.Executor(paddle.CPUPlace())
                (res,) = exe.run(
                    paddle.static.default_main_program(),
                    feed={'x': data_x},
                    fetch_list=[z],
                    return_numpy=False,
                )
                expect_out = np.array(
                    [[7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
                )
            np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)
            paddle.disable_static()

    def test_dygraph_api(self):
        self.input_data()
        # case 1:
        with base.dygraph.guard():
            x = paddle.to_tensor(self.data_x)
            z = paddle.roll(x, shifts=1)
            np_z = z.numpy()
        expect_out = np.array(
            [[9.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
        )
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case 2:
        with base.dygraph.guard():
            x = paddle.to_tensor(self.data_x)
            z = paddle.roll(x, shifts=1, axis=0)
            np_z = z.numpy()
        expect_out = np.array(
            [[7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

    def test_roll_op_false(self):
        def test_axis_out_range():
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[-1, 3], dtype='float32')
                data_x = np.array(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
                ).astype('float32')
                z = paddle.roll(x, shifts=1, axis=10)
                exe = base.Executor(base.CPUPlace())
                (res,) = exe.run(
                    feed={'x': data_x},
                    fetch_list=[z],
                    return_numpy=False,
                )

        self.assertRaises(ValueError, test_axis_out_range)
        paddle.disable_static()

    def test_shifts_as_tensor_dygraph(self):
        with base.dygraph.guard():
            x = paddle.arange(9).reshape([3, 3])
            shape = paddle.shape(x)
            shifts = shape // 2
            axes = [0, 1]
            out = paddle.roll(x, shifts=shifts, axis=axes).numpy()
            expected_out = np.array([[8, 6, 7], [2, 0, 1], [5, 3, 4]])
            np.testing.assert_allclose(out, expected_out, rtol=1e-05)

    def test_shifts_as_tensor_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.arange(9).reshape([3, 3]).astype('float32')
            shape = paddle.shape(x)
            shifts = shape // 2
            axes = [0, 1]
            out = paddle.roll(x, shifts=shifts, axis=axes)
            expected_out = np.array([[8, 6, 7], [2, 0, 1], [5, 3, 4]])

            exe = paddle.static.Executor(paddle.CPUPlace())
            [out_np] = exe.run(fetch_list=[out])
            np.testing.assert_allclose(out_np, expected_out, rtol=1e-05)

            if paddle.is_compiled_with_cuda():
                exe = base.Executor(base.CPUPlace())
                [out_np] = exe.run(fetch_list=[out])
                np.testing.assert_allclose(out_np, expected_out, rtol=1e-05)
        paddle.disable_static()


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for bool dtype is not fully supported",
)
class TestRollBoolAPI(unittest.TestCase):
    def input_data(self):
        self.data_x_bool = np.array(
            [[True, False, True], [False, True, False], [True, False, True]]
        ).astype('bool')

    def test_roll_op_api_case1(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[3, 3], dtype='bool')
                data_x = np.array(
                    [
                        [True, False, True],
                        [False, True, False],
                        [True, False, True],
                    ]
                ).astype('bool')
                z = paddle.roll(x, shifts=1)
                exe = paddle.static.Executor(paddle.CPUPlace())
                (res,) = exe.run(
                    paddle.static.default_main_program(),
                    feed={'x': data_x},
                    fetch_list=[z],
                    return_numpy=False,
                )
                expect_out = np.array(
                    [
                        [True, True, False],
                        [True, False, True],
                        [False, True, False],
                    ]
                )
            np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_roll_op_api_case2(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[3, 3], dtype='bool')
                data_x = np.array(
                    [
                        [True, False, True],
                        [False, True, False],
                        [True, False, True],
                    ]
                ).astype('bool')
                z = paddle.roll(x, shifts=1, axis=0)
                exe = paddle.static.Executor(paddle.CPUPlace())
                (res,) = exe.run(
                    paddle.static.default_main_program(),
                    feed={'x': data_x},
                    fetch_list=[z],
                    return_numpy=False,
                )
                expect_out = np.array(
                    [
                        [True, False, True],
                        [True, False, True],
                        [False, True, False],
                    ]
                )
            np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_dygraph_api(self):
        self.input_data()
        # case 1:
        with base.dygraph.guard():
            x = paddle.to_tensor(self.data_x_bool)
            z = paddle.roll(x, shifts=1)
            np_z = z.numpy()
        expect_out = np.array(
            [[True, True, False], [True, False, True], [False, True, False]]
        )
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case 2:
        with base.dygraph.guard():
            x = paddle.to_tensor(self.data_x_bool)
            z = paddle.roll(x, shifts=1, axis=0)
            np_z = z.numpy()
        expect_out = np.array(
            [[True, False, True], [True, False, True], [False, True, False]]
        )
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for zero size is not fully supported",
)
class TestRoll0SizelAPI(unittest.TestCase):
    def input_data(self):
        self.data_x_zero_size1 = np.array([]).reshape(0, 3).astype('float32')
        self.data_x_zero_size2 = np.array([]).reshape(4, 0, 3).astype('float32')

    def test_roll_op_api_case1(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='x', shape=[0, 3], dtype='float32')
            data_x = np.array([]).reshape(0, 3).astype('float32')
            z = paddle.roll(x, shifts=1)
            exe = paddle.static.Executor(paddle.CPUPlace())
            (res,) = exe.run(
                paddle.static.default_main_program(),
                feed={'x': data_x},
                fetch_list=[z],
                return_numpy=False,
            )
            expect_out = np.array([]).reshape(0, 3)
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_roll_op_api_case2(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[0, 3], dtype='float32')
                data_x = np.array([]).reshape(0, 3).astype('float32')
                z = paddle.roll(x, shifts=1, axis=0)
                exe = paddle.static.Executor(paddle.CPUPlace())
                (res,) = exe.run(
                    paddle.static.default_main_program(),
                    feed={'x': data_x},
                    fetch_list=[z],
                    return_numpy=False,
                )
                expect_out = np.array([]).reshape(0, 3)
            np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_roll_op_api_case3(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name='x', shape=[4, 0, 3], dtype='float32'
                )
                data_x = np.array([]).reshape(4, 0, 3).astype('float32')
                z = paddle.roll(x, shifts=1)
                exe = paddle.static.Executor(paddle.CPUPlace())
                (res,) = exe.run(
                    paddle.static.default_main_program(),
                    feed={'x': data_x},
                    fetch_list=[z],
                    return_numpy=False,
                )
                expect_out = np.array([]).reshape(4, 0, 3)
            np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_roll_op_api_case4(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name='x', shape=[4, 0, 3], dtype='float32'
                )
                data_x = np.array([]).reshape(4, 0, 3).astype('float32')
                z = paddle.roll(x, shifts=1, axis=0)
                exe = paddle.static.Executor(paddle.CPUPlace())
                (res,) = exe.run(
                    paddle.static.default_main_program(),
                    feed={'x': data_x},
                    fetch_list=[z],
                    return_numpy=False,
                )
                expect_out = np.array([]).reshape(4, 0, 3)
            np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_dygraph_api(self):
        self.input_data()
        # case 1:
        with base.dygraph.guard():
            x = paddle.to_tensor(self.data_x_zero_size1)
            z = paddle.roll(x, shifts=1)
            np_z = z.numpy()
        expect_out = np.array([]).reshape(0, 3)
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case 2:
        with base.dygraph.guard():
            x = paddle.to_tensor(self.data_x_zero_size1)
            z = paddle.roll(x, shifts=1, axis=0)
            np_z = z.numpy()
        expect_out = np.array([]).reshape(0, 3)
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case 3:
        with base.dygraph.guard():
            x = paddle.to_tensor(self.data_x_zero_size2)
            z = paddle.roll(x, shifts=1)
            np_z = z.numpy()
        expect_out = np.array([]).reshape(4, 0, 3)
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case 4:
        with base.dygraph.guard():
            x = paddle.to_tensor(self.data_x_zero_size2)
            z = paddle.roll(x, shifts=1, axis=0)
            np_z = z.numpy()
        expect_out = np.array([]).reshape(4, 0, 3)
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
