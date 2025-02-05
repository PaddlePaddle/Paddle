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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import core

np.random.seed(123)
paddle.enable_static()


class PyArgsort:
    def __init__(self, input_shape, axis, descending, dtype):
        self.x = np.random.random(input_shape).astype(dtype)
        self.label = np.random.random(input_shape).astype(dtype)
        if axis < 0:
            self.axis = axis + len(self.x.shape)
        else:
            self.axis = axis
        self.descending = descending

    def forward(self):
        if self.descending:
            self.indices = np.flip(
                np.argsort(self.x, kind='quicksort', axis=self.axis), self.axis
            )
            self.sorted_x = np.flip(
                np.sort(self.x, kind='quicksort', axis=self.axis), self.axis
            )
        else:
            self.indices = np.argsort(self.x, kind='quicksort', axis=self.axis)
            self.sorted_x = np.sort(self.x, kind='quicksort', axis=self.axis)
        self.loss = self.sorted_x * self.label
        self.loss = np.sum(self.loss)
        out = (
            np.array(self.indices, dtype=self.indices.dtype),
            np.array(self.sorted_x, dtype=self.sorted_x.dtype),
            np.array(self.loss, dtype=self.loss.dtype),
        )
        return out


def create_tensor(np_data, place):
    tensor = core.DenseTensor()
    tensor.set(np_data, place)
    return tensor


class TestArgsortErrorOnCPU(unittest.TestCase):
    def setUp(self):
        self.place = core.CPUPlace()

    def test_error(self):
        def test_base_var_type():
            with paddle.static.program_guard(paddle.static.Program()):
                x = [1]
                output = paddle.argsort(x=x)
            self.assertRaises(TypeError, test_base_var_type)

        def test_paddle_var_type():
            with paddle.static.program_guard(paddle.static.Program()):
                x = [1]
                output = paddle.argsort(x=x)
            self.assertRaises(TypeError, test_paddle_var_type)


class TestArgsortErrorOnGPU(TestArgsortErrorOnCPU):
    def setUp(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()


class TestArgsort(unittest.TestCase):
    def setUp(self):
        self.input_shape = [
            10000,
        ]
        self.axis = 0
        self.data = np.random.rand(*self.input_shape)

    def test_api_static1(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()
        with paddle.static.program_guard(paddle.static.Program()):
            input = paddle.static.data(
                name="input", shape=self.input_shape, dtype="float64"
            )
            output = paddle.argsort(input, axis=self.axis)
            np_result = np.argsort(self.data, axis=self.axis)
            exe = paddle.static.Executor(self.place)
            result = exe.run(
                paddle.static.default_main_program(),
                feed={'input': self.data},
                fetch_list=[output],
            )

            self.assertEqual((result == np_result).all(), True)

    def test_api_static2(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()
        with paddle.static.program_guard(paddle.static.Program()):
            input = paddle.static.data(
                name="input", shape=self.input_shape, dtype="float64"
            )
            output2 = paddle.argsort(input, axis=self.axis, descending=True)
            np_result2 = np.argsort(-self.data, axis=self.axis)
            exe = paddle.static.Executor(self.place)
            result2 = exe.run(
                paddle.static.default_main_program(),
                feed={'input': self.data},
                fetch_list=[output2],
            )

            self.assertEqual((result2 == np_result2).all(), True)


class TestArgsort2(TestArgsort):
    def init(self):
        self.input_shape = [10000, 1]
        self.axis = 0


class TestArgsort3(TestArgsort):
    def init(self):
        self.input_shape = [1, 10000]
        self.axis = 1


class TestArgsort4(TestArgsort):
    def init(self):
        self.input_shape = [2, 3, 4]
        self.axis = 1


class TestStableArgsort(unittest.TestCase):
    def init(self):
        self.input_shape = [
            30,
        ]
        self.axis = 0
        self.data = np.array([100.0, 50.0, 10.0] * 10)

    def setUp(self):
        self.init()

    def cpu_place(self):
        self.place = core.CPUPlace()

    def gpu_place(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

    def test_api_static1_cpu(self):
        self.cpu_place()
        with paddle.static.program_guard(paddle.static.Program()):
            input = paddle.static.data(
                name="input", shape=self.input_shape, dtype="float64"
            )
            output = paddle.argsort(input, axis=self.axis, stable=True)
            np_result = np.argsort(self.data, axis=self.axis, kind='stable')
            exe = paddle.static.Executor(self.place)
            result = exe.run(
                paddle.static.default_main_program(),
                feed={'input': self.data},
                fetch_list=[output],
            )

            self.assertEqual((result == np_result).all(), True)

    def test_api_static1_gpu(self):
        self.gpu_place()
        with paddle.static.program_guard(paddle.static.Program()):
            input = paddle.static.data(
                name="input", shape=self.input_shape, dtype="float64"
            )
            output = paddle.argsort(input, axis=self.axis, stable=True)
            np_result = np.argsort(self.data, axis=self.axis, kind='stable')
            exe = paddle.static.Executor(self.place)
            result = exe.run(
                paddle.static.default_main_program(),
                feed={'input': self.data},
                fetch_list=[output],
            )

            self.assertEqual((result == np_result).all(), True)

    def test_api_static2_cpu(self):
        self.cpu_place()
        with paddle.static.program_guard(paddle.static.Program()):
            input = paddle.static.data(
                name="input", shape=self.input_shape, dtype="float64"
            )
            output2 = paddle.argsort(
                input, axis=self.axis, descending=True, stable=True
            )
            np_result2 = np.argsort(-self.data, axis=self.axis, kind='stable')
            exe = paddle.static.Executor(self.place)
            result2 = exe.run(
                paddle.static.default_main_program(),
                feed={'input': self.data},
                fetch_list=[output2],
            )

            self.assertEqual((result2 == np_result2).all(), True)

    def test_api_static2_gpu(self):
        self.gpu_place()
        with paddle.static.program_guard(paddle.static.Program()):
            input = paddle.static.data(
                name="input", shape=self.input_shape, dtype="float64"
            )
            output2 = paddle.argsort(
                input, axis=self.axis, descending=True, stable=True
            )
            np_result2 = np.argsort(-self.data, axis=self.axis, kind='stable')
            exe = paddle.static.Executor(self.place)
            result2 = exe.run(
                paddle.static.default_main_program(),
                feed={'input': self.data},
                fetch_list=[output2],
            )

            self.assertEqual((result2 == np_result2).all(), True)


class TestStableArgsort2(TestStableArgsort):
    def init(self):
        self.input_shape = [30, 1]
        self.data = np.array([100.0, 50.0, 10.0] * 10).reshape(self.input_shape)
        self.axis = 0


class TestStableArgsort3(TestStableArgsort):
    def init(self):
        self.input_shape = [1, 30]
        self.data = np.array([100.0, 50.0, 10.0] * 10).reshape(self.input_shape)
        self.axis = 1


class TestStableArgsort4(TestStableArgsort):
    def init(self):
        self.input_shape = [40, 3, 4]
        self.axis = 0
        self.data = np.array(
            [
                [
                    [100.0, 50.0, -10.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [100.0, 50.0, -10.0, 1.0],
                ],
                [
                    [70.0, -30.0, 60.0, 100.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [100.0, 50.0, -10.0, 1.0],
                ],
            ]
            * 20
        )


class TestArgsortImperative(unittest.TestCase):
    def init(self):
        self.input_shape = [
            10000,
        ]
        self.axis = 0

    def setUp(self):
        self.init()
        self.input_data = np.random.rand(*self.input_shape)
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

    def test_api(self):
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.argsort(var_x, axis=self.axis)
        expect = np.argsort(self.input_data, axis=self.axis)
        self.assertEqual((expect == out.numpy()).all(), True)

        out2 = paddle.argsort(var_x, axis=self.axis, descending=True)
        expect2 = np.argsort(-self.input_data, axis=self.axis)
        self.assertEqual((expect2 == out2.numpy()).all(), True)

        paddle.enable_static()


class TestArgsortImperative2(TestArgsortImperative):
    def init(self):
        self.input_shape = [10000, 1]
        self.axis = 0


class TestArgsortImperative3(TestArgsortImperative):
    def init(self):
        self.input_shape = [1, 10000]
        self.axis = 1


class TestArgsortImperative4(TestArgsortImperative):
    def init(self):
        self.input_shape = [2, 3, 4]
        self.axis = 1


class TestStableArgsortImperative(unittest.TestCase):
    def init(self):
        self.input_shape = [
            30,
        ]
        self.axis = 0
        self.input_data = np.array([100.0, 50.0, 10.0] * 10)

    def setUp(self):
        self.init()

    def cpu_place(self):
        self.place = core.CPUPlace()

    def gpu_place(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

    def test_api_cpu(self):
        self.cpu_place()
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.argsort(var_x, axis=self.axis, stable=True)
        expect = np.argsort(self.input_data, axis=self.axis, kind='stable')
        self.assertEqual((expect == out.numpy()).all(), True)

        out2 = paddle.argsort(
            var_x, axis=self.axis, descending=True, stable=True
        )
        expect2 = np.argsort(-self.input_data, axis=self.axis, kind='stable')
        self.assertEqual((expect2 == out2.numpy()).all(), True)

        paddle.enable_static()

    def test_api_gpu(self):
        self.gpu_place()
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.argsort(var_x, axis=self.axis, stable=True)
        expect = np.argsort(self.input_data, axis=self.axis, kind='stable')
        self.assertEqual((expect == out.numpy()).all(), True)

        out2 = paddle.argsort(
            var_x, axis=self.axis, descending=True, stable=True
        )
        expect2 = np.argsort(-self.input_data, axis=self.axis, kind='stable')
        self.assertEqual((expect2 == out2.numpy()).all(), True)

        paddle.enable_static()


class TestStableArgsortImperative2(TestStableArgsortImperative):
    def init(self):
        self.input_shape = [30, 1]
        self.input_data = np.array([100.0, 50.0, 10.0] * 10).reshape(
            self.input_shape
        )
        self.axis = 0


class TestStableArgsortImperative3(TestStableArgsortImperative):
    def init(self):
        self.input_shape = [1, 30]
        self.input_data = np.array([100.0, 50.0, 10.0] * 10).reshape(
            self.input_shape
        )
        self.axis = 1


class TestStableArgsortImperative4(TestStableArgsortImperative):
    def init(self):
        self.input_shape = [40, 3, 4]
        self.axis = 0
        self.input_data = np.array(
            [
                [
                    [100.0, 50.0, -10.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [100.0, 50.0, -10.0, 1.0],
                ],
                [
                    [70.0, -30.0, 60.0, 100.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [100.0, 50.0, -10.0, 1.0],
                ],
            ]
            * 20
        )


class TestArgsortWithInputNaN(unittest.TestCase):
    def init(self):
        self.axis = 0

    def setUp(self):
        self.init()
        self.input_data = np.array([1.0, np.nan, 3.0, 2.0])
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

    def test_api(self):
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.argsort(var_x, axis=self.axis)
        self.assertEqual((out.numpy() == np.array([0, 3, 2, 1])).all(), True)

        out = paddle.argsort(var_x, axis=self.axis, descending=True)
        self.assertEqual((out.numpy() == np.array([1, 2, 3, 0])).all(), True)
        paddle.enable_static()


class TestArgsortOpFp16(unittest.TestCase):

    def test_fp16(self):
        if base.core.is_compiled_with_cuda():
            paddle.enable_static()
            x_np = np.random.random((2, 8)).astype('float16')
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(shape=[2, 8], name='x', dtype='float16')
                out = paddle.argsort(x)
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                exe.run(paddle.static.default_startup_program())
                out = exe.run(feed={'x': x_np}, fetch_list=[out])
            paddle.disable_static()


class TestArgsortFP16Op(OpTest):
    def setUp(self):
        self.init()
        self.init_direction()
        self.op_type = "argsort"
        self.prim_op_type = "prim"
        self.python_api = paddle.argsort
        self.public_python_api = paddle.argsort
        self.python_out_sig = ["Out"]
        self.dtype = np.float16
        self.descending = False
        self.attrs = {"axis": self.axis, "descending": self.descending}
        X = np.random.rand(*self.input_shape).astype('float16')
        Out = np.sort(X, kind='quicksort', axis=self.axis)
        indices = np.argsort(X, kind='quicksort', axis=self.axis)
        self.inputs = {'X': X}
        self.outputs = {
            'Out': Out,
            'Indices': indices,
        }

    def init(self):
        self.input_shape = [
            10000,
        ]
        self.axis = 0

    def init_direction(self):
        self.descending = False

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_dygraph=False,
            check_pir=True,
            check_prim_pir=True,
        )


class TestArgsortFP16OpDescendingTrue(TestArgsortFP16Op):
    def init_direction(self):
        self.descending = True


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestArgsortBF16Op(OpTest):
    def setUp(self):
        self.init()
        self.init_direction()
        self.op_type = "argsort"
        self.prim_op_type = "prim"
        self.python_api = paddle.argsort
        self.public_python_api = paddle.argsort
        self.python_out_sig = ["Out"]
        self.dtype = np.uint16
        self.np_dtype = np.float32
        self.descending = False
        self.attrs = {"axis": self.axis, "descending": self.descending}
        X = np.random.rand(*self.input_shape).astype(self.np_dtype)
        Out = np.sort(X, kind='quicksort', axis=self.axis)
        indices = np.argsort(X, kind='quicksort', axis=self.axis)
        self.inputs = {'X': convert_float_to_uint16(X)}
        self.outputs = {
            'Out': convert_float_to_uint16(Out),
            'Indices': convert_float_to_uint16(indices),
        }

    def init(self):
        self.input_shape = [
            10000,
        ]
        self.axis = 0

    def init_direction(self):
        self.descending = False

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_dygraph=False,
            check_pir=True,
            check_prim_pir=True,
        )


class TestArgsortBF16OpDescendingTrue(TestArgsortBF16Op):
    def init_direction(self):
        self.descending = True


if __name__ == "__main__":
    unittest.main()
