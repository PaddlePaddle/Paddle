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

import os
import sys
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.framework import core

paddle.enable_static()


class TestTakeAlongAxisOp(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "take_along_axis"
        self.prim_op_type = "prim"
        self.python_api = paddle.tensor.take_along_axis
        self.public_python_api = paddle.tensor.take_along_axis
        self.check_cinn = True
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.target = np.take_along_axis(self.xnp, self.index, self.axis)
        broadcast_shape_list = list(self.x_shape)
        broadcast_shape_list[self.axis] = 1
        self.broadcast_shape = tuple(broadcast_shape_list)
        self.index_broadcast = np.broadcast_to(self.index, self.broadcast_shape)
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index_broadcast,
        }
        self.attrs = {'Axis': self.axis}
        self.outputs = {'Result': self.target}

    def test_check_output(self):
        self.check_output(check_cinn=self.check_cinn, check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['Input'],
            'Result',
            check_cinn=self.check_cinn,
            check_pir=True,
            check_prim_pir=True,
        )

    def init_data(self):
        self.x_type = "float64"
        self.x_shape = (5, 5, 5)
        self.index_type = "int32"
        self.axis = 2
        dim_size = self.x_shape[self.axis]
        self.index = np.random.randint(
            -dim_size, dim_size, size=(5, 1, 1)
        ).astype(self.index_type)
        self.axis_type = "int64"


class TestTakeAlongAxisDuplicatedIndices(TestTakeAlongAxisOp):
    def init_data(self):
        self.dtype = np.float32
        self.x_type = "float32"
        self.x_shape = (5, 6, 7)
        self.index_type = "int64"
        self.axis = 2
        dim_size = self.x_shape[self.axis]
        self.index = (
            np.asarray([-dim_size, -dim_size, dim_size - 1, dim_size - 1, 0])
            .astype(self.index_type)
            .reshape([5, 1, 1])
        )
        self.axis_type = "int64"

    def test_check_output(self):
        self.check_output(
            check_cinn=self.check_cinn, check_pir=True, check_prim_pir=True
        )

    def test_check_grad(self):
        self.check_grad(
            ['Input'],
            'Result',
            check_cinn=self.check_cinn,
            check_pir=True,
            check_prim_pir=True,
        )


class TestTakeAlongAxisFP16Op(TestTakeAlongAxisOp):
    def init_data(self):
        self.dtype = np.float16
        self.x_type = "float16"
        self.x_shape = (5, 5, 5)
        self.index_type = "int32"
        self.axis = 2
        dim_size = self.x_shape[self.axis]
        self.index = np.random.randint(
            -dim_size, dim_size, size=(5, 1, 1)
        ).astype(self.index_type)
        self.axis_type = "int64"


class TestTakeAlongAxisOp2(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "take_along_axis"
        self.python_api = paddle.tensor.take_along_axis
        self.check_cinn = True
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.target = np.zeros((2, 3, 4)).astype(self.x_type)
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.target[i, j, k] = self.xnp[i, j, self.index[i, j, k]]
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
        }
        self.attrs = {'Axis': self.axis, 'broadcast': False}
        self.outputs = {'Result': self.target}

    def init_data(self):
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.index_type = "int64"
        self.axis = 2
        dim_size = self.x_shape[self.axis]
        self.index = np.random.randint(-dim_size, dim_size, (2, 3, 4)).astype(
            self.index_type
        )
        self.axis_type = "int64"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestTakeAlongAxisBF16Op(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "take_along_axis"
        self.prim_op_type = "prim"
        self.python_api = paddle.tensor.take_along_axis
        self.public_python_api = paddle.tensor.take_along_axis
        self.check_cinn = True
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.target = np.take_along_axis(self.xnp, self.index, self.axis)
        broadcast_shape_list = list(self.x_shape)
        broadcast_shape_list[self.axis] = 1
        self.broadcast_shape = tuple(broadcast_shape_list)
        self.index_broadcast = np.broadcast_to(self.index, self.broadcast_shape)
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index_broadcast,
        }
        self.attrs = {'Axis': self.axis}
        self.outputs = {'Result': self.target}

        self.inputs['Input'] = convert_float_to_uint16(self.inputs['Input'])
        self.outputs['Result'] = convert_float_to_uint16(self.outputs['Result'])
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        self.check_output_with_place(
            self.place, check_cinn=self.check_cinn, check_pir=True
        )

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ['Input'],
            'Result',
            check_cinn=self.check_cinn,
            check_pir=True,
            check_prim_pir=True,
        )

    def init_data(self):
        self.dtype = np.uint16
        self.x_type = "float32"
        self.x_shape = (5, 5, 5)
        self.index_type = "int32"
        self.axis = 2
        dim_size = self.x_shape[self.axis]
        self.index = np.random.randint(
            -dim_size, dim_size, size=(5, 1, 1)
        ).astype(self.index_type)
        self.axis_type = "int64"


class TestCase1(TestTakeAlongAxisOp):
    def init_data(self):
        self.x_type = "float64"
        self.x_shape = (5, 5, 5)
        self.index_type = "int32"
        self.axis = 0
        dim_size = self.x_shape[self.axis]
        self.index = np.random.randint(
            -dim_size, dim_size, size=(1, 1, 5)
        ).astype(self.index_type)
        self.axis_type = "int64"


class TestTakeAlongAxisAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [3, 3]
        self.index_shape = [1, 3]
        self.axis = 0
        dim_size = self.shape[self.axis]
        self.index_np = np.random.randint(
            -dim_size, dim_size, size=([1, 3])
        ).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            index = paddle.static.data('Index', self.index_shape, "int64")
            out = paddle.take_along_axis(x, index, self.axis)
            exe = paddle.static.Executor(self.place[0])
            res = exe.run(
                feed={'X': self.x_np, 'Index': self.index_np}, fetch_list=[out]
            )
        out_ref = np.array(
            np.take_along_axis(self.x_np, self.index_np, self.axis)
        )
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=0.001)

    def test_api_dygraph(self):
        paddle.disable_static(self.place[0])
        x_tensor = paddle.to_tensor(self.x_np)
        self.index = paddle.to_tensor(self.index_np)
        out = paddle.take_along_axis(x_tensor, self.index, self.axis)
        out_ref = np.array(
            np.take_along_axis(self.x_np, self.index_np, self.axis)
        )
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
        paddle.enable_static()

    def test_api_dygraph_dtype(self):
        if sys.platform == 'darwin' or sys.platform == 'win32':
            return
        paddle.disable_static(paddle.CPUPlace())
        with self.assertRaises(AssertionError):
            x_tensor = paddle.to_tensor(self.x_np)
            self.index = paddle.to_tensor(self.index_np).astype("float32")
            out = paddle.take_along_axis(x_tensor, self.index, self.axis)
            out_ref = np.array(
                np.take_along_axis(self.x_np, self.index_np, self.axis)
            )
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
        paddle.enable_static()


class TestTakeAlongAxisAPICase1(TestTakeAlongAxisAPI):
    def setUp(self):
        np.random.seed(0)
        self.shape = [2, 2]
        self.index_shape = [4, 2]
        self.axis = 0
        dim_size = self.shape[self.axis]
        self.index_np = np.random.randint(
            -dim_size, dim_size, size=(4, 2)
        ).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))


class TestTakeAlongAxisAPICase2(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [3, 3]
        self.index_shape = [1, 3]
        self.axis = 0
        dim_size = self.shape[self.axis]
        self.index_np = np.random.randint(
            -dim_size, dim_size, size=(1, 3)
        ).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            index = paddle.static.data('Index', self.index_shape, "int64")
            out = paddle.take_along_axis(x, index, self.axis, False)
            exe = paddle.static.Executor(self.place[0])
            res = exe.run(
                feed={'X': self.x_np, 'Index': self.index_np}, fetch_list=[out]
            )
        out_ref = np.zeros_like(self.index_np, dtype=self.x_np.dtype)
        for i in range(self.index_shape[0]):
            for j in range(self.index_shape[1]):
                out_ref[i, j] = self.x_np[self.index_np[i, j], j]
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=0.001)

    def test_api_dygraph(self):
        paddle.disable_static(self.place[0])
        x_tensor = paddle.to_tensor(self.x_np)
        self.index = paddle.to_tensor(self.index_np)
        out = paddle.take_along_axis(x_tensor, self.index, self.axis, False)
        out_ref = np.zeros_like(self.index_np, dtype=self.x_np.dtype)
        for i in range(self.index_shape[0]):
            for j in range(self.index_shape[1]):
                out_ref[i, j] = self.x_np[self.index_np[i, j], j]
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
        paddle.enable_static()

    def test_error(self):
        paddle.disable_static(self.place[0])
        tensorx = paddle.to_tensor([[1, 2, 3], [4, 5, 6]]).astype("float32")
        indices = paddle.to_tensor([1]).astype("int32")
        # len(arr.shape) != len(indices.shape)
        with self.assertRaises(ValueError):
            res = paddle.take_along_axis(tensorx, indices, 0, False)
        # the element of indices out of range
        # (only catch cpu assertion though gpu can raise exception)
        with self.assertRaises(IndexError):
            indices = paddle.to_tensor([[100]]).astype("int32")
            res = paddle.take_along_axis(
                tensorx.to("cpu"), indices.to("cpu"), 0, False
            )
        with self.assertRaises(IndexError):
            indices = paddle.to_tensor([[-100]]).astype("int32")
            res = paddle.take_along_axis(
                tensorx.to("cpu"), indices.to("cpu"), 0, False
            )
        # the shape of indices doesn't match
        with self.assertRaises(RuntimeError):
            indices = paddle.to_tensor(
                [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
            ).astype("int32")
            res = paddle.take_along_axis(tensorx, indices, 0, False)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
