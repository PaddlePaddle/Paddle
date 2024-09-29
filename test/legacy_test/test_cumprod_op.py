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

import os
import random
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core

np.random.seed(0)


def cumprod_wrapper(x, dim=-1, exclusive=False, reverse=False):
    return paddle._C_ops.cumprod(x, dim, exclusive, reverse)


# define cumprod grad function.
def cumprod_grad(x, y, dy, dx, shape, dim, exclusive=False, reverse=False):
    if dim < 0:
        dim += len(shape)
    mid_dim = shape[dim]
    outer_dim = 1
    inner_dim = 1
    for i in range(0, dim):
        outer_dim *= shape[i]
    for i in range(dim + 1, len(shape)):
        inner_dim *= shape[i]
    if not reverse:
        for i in range(outer_dim):
            for k in range(inner_dim):
                for j in range(mid_dim):
                    index = i * mid_dim * inner_dim + j * inner_dim + k
                    for n in range(mid_dim):
                        pos = i * mid_dim * inner_dim + n * inner_dim + k
                        elem = 0
                        if exclusive:
                            if pos > index:
                                elem = dy[pos] * y[index]
                                for m in range(
                                    index + inner_dim, pos, inner_dim
                                ):
                                    elem *= x[m]
                            else:
                                elem = 0
                        else:
                            if j == 0:
                                elem = dy[pos]
                            else:
                                elem = dy[pos] * y[index - inner_dim]
                            if pos > index:
                                for m in range(
                                    index + inner_dim,
                                    pos + inner_dim,
                                    inner_dim,
                                ):
                                    elem *= x[m]
                            elif pos < index:
                                elem = 0
                        dx[index] += elem
    else:
        for i in range(outer_dim):
            for k in range(inner_dim):
                for j in range(mid_dim - 1, -1, -1):
                    index = i * mid_dim * inner_dim + j * inner_dim + k
                    for n in range(mid_dim - 1, -1, -1):
                        pos = i * mid_dim * inner_dim + n * inner_dim + k
                        elem = 0
                        if exclusive:
                            if pos < index:
                                elem = dy[pos] * y[index]
                                for m in range(
                                    index - inner_dim, pos, -inner_dim
                                ):
                                    elem *= x[m]
                        else:
                            if j == mid_dim - 1:
                                elem = dy[pos]
                            else:
                                elem = dy[pos] * y[index + inner_dim]
                            if pos < index:
                                for m in range(
                                    index - inner_dim,
                                    pos - inner_dim,
                                    -inner_dim,
                                ):
                                    elem *= x[m]
                            elif pos > index:
                                elem = 0
                        dx[index] += elem


# test function.
class TestCumprod(OpTest):
    def init_params(self):
        self.shape = (2, 3, 4, 5)
        self.zero_nums = [0, 10, 20, 30, int(np.prod(self.shape))]

    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def setUp(self):
        paddle.enable_static()
        self.init_params()
        self.init_dtype()
        self.op_type = "cumprod"
        self.python_api = cumprod_wrapper
        self.inputs = {'X': None}
        self.outputs = {'Out': None}
        self.attrs = {'dim': None}

    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype)
            + 0.5
            # np.ones(self.shape).astype(self.val_dtype)
        )
        if zero_num > 0:
            zero_num = min(zero_num, self.x.size)
            shape = self.x.shape
            self.x = self.x.flatten()
            indices = random.sample(range(self.x.size), zero_num)
            for i in indices:
                self.x[i] = 0
            self.x = np.reshape(self.x, self.shape)
        self.out = np.cumprod(self.x, axis=dim)
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
        self.attrs = {'dim': dim}

    def init_grad_input_output(self, dim):
        reshape_x = self.x.reshape(self.x.size)
        self.grad_out = np.ones(self.x.size, self.val_dtype)
        self.grad_x = np.zeros(self.x.size, self.val_dtype)
        out_data = self.out.reshape(self.x.size)
        if self.dtype == np.complex128 or self.dtype == np.complex64:
            reshape_x = np.conj(reshape_x)
            out_data = np.conj(out_data)
        cumprod_grad(
            reshape_x, out_data, self.grad_out, self.grad_x, self.shape, dim
        )
        if self.dtype == np.uint16:
            self.grad_x = convert_float_to_uint16(
                self.grad_x.reshape(self.shape)
            )
            self.grad_out = convert_float_to_uint16(
                self.grad_out.reshape(self.shape)
            )
        else:
            self.grad_x = self.grad_x.reshape(self.shape)
            self.grad_out = self.grad_out.reshape(self.shape)

    # test forward.
    def test_check_output(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in self.zero_nums:
                self.prepare_inputs_outputs_attrs(dim, zero_num)
                self.check_output(check_pir=True)

    # test backward.
    def test_check_grad(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in self.zero_nums:
                self.prepare_inputs_outputs_attrs(dim, zero_num)
                self.init_grad_input_output(dim)
                if self.dtype == np.float64:
                    self.check_grad(['X'], 'Out', check_pir=True)
                else:
                    self.check_grad(
                        ['X'],
                        'Out',
                        user_defined_grads=[self.grad_x],
                        user_defined_grad_outputs=[self.grad_out],
                        check_pir=True,
                    )


# test float32 case.
class TestCumprodFP32Op(TestCumprod):
    def init_dtype(self):
        self.dtype = np.float32
        self.val_dtype = np.float32


class TestCumprodFP16Op(TestCumprod):
    def init_dtype(self):
        self.dtype = np.float16
        self.val_dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestCumprodBF16Op(TestCumprod):
    def init_dtype(self):
        self.dtype = np.uint16
        self.val_dtype = np.float32

    # test forward.
    def test_check_output(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in self.zero_nums:
                self.prepare_inputs_outputs_attrs(dim, zero_num)
                self.check_output_with_place(core.CUDAPlace(0))

    # test backward.
    def test_check_grad(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in self.zero_nums:
                self.prepare_inputs_outputs_attrs(dim, zero_num)
                self.init_grad_input_output(dim)
                self.check_grad_with_place(
                    core.CUDAPlace(0),
                    ['X'],
                    'Out',
                    user_defined_grads=[self.grad_x],
                    user_defined_grad_outputs=[self.grad_out],
                )


# test complex64 case.
class TestCumprodComplex64Op(TestCumprod):
    def init_dtype(self):
        self.dtype = np.complex64
        self.val_dtype = np.complex64


# test complex128 case.
class TestCumprodComplex128Op(TestCumprod):
    def init_dtype(self):
        self.dtype = np.complex128
        self.val_dtype = np.complex128


# test api.
class TestCumprodAPI(unittest.TestCase):
    def init_dtype(self):
        self.dtype = 'float64'
        self.shape = [2, 3, 10, 10]

    def setUp(self):
        paddle.enable_static()
        self.init_dtype()
        self.x = (np.random.rand(2, 3, 10, 10) + 0.5).astype(self.dtype)
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    # test static graph api.

    def test_static_api(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.shape, dtype=self.dtype)
                out = paddle.cumprod(x, -2)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={'X': self.x}, fetch_list=[out])
            out_ref = np.cumprod(self.x, -2)

            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

        for place in self.place:
            run(place)

    # test dynamic graph api.
    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            out = paddle.cumprod(x, 1)
            out_ref = np.cumprod(self.x, 1)
            np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
            paddle.enable_static()

        for place in self.place:
            run(place)


# test function.
class TestCumprodReverse(TestCumprod):
    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype) + 0.5
        )
        if zero_num > 0:
            zero_num = min(zero_num, self.x.size)
            shape = self.x.shape
            self.x = self.x.flatten()
            indices = random.sample(range(self.x.size), zero_num)
            for i in indices:
                self.x[i] = 0
            self.x = np.reshape(self.x, self.shape)
        self.out = np.flip(
            np.flip(self.x, axis=dim).cumprod(axis=dim), axis=dim
        )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
        self.attrs = {'dim': dim, 'reverse': True}

    def init_grad_input_output(self, dim):
        reshape_x = self.x.reshape(self.x.size)
        self.grad_out = np.ones(self.x.size, self.val_dtype)
        self.grad_x = np.zeros(self.x.size, self.val_dtype)
        out_data = self.out.reshape(self.x.size)
        if self.dtype == np.complex128 or self.dtype == np.complex64:
            reshape_x = np.conj(reshape_x)
            out_data = np.conj(out_data)
        cumprod_grad(
            reshape_x,
            out_data,
            self.grad_out,
            self.grad_x,
            self.shape,
            dim,
            exclusive=False,
            reverse=True,
        )
        if self.dtype == np.uint16:
            self.grad_x = convert_float_to_uint16(
                self.grad_x.reshape(self.shape)
            )
            self.grad_out = convert_float_to_uint16(
                self.grad_out.reshape(self.shape)
            )
        else:
            self.grad_x = self.grad_x.reshape(self.shape)
            self.grad_out = self.grad_out.reshape(self.shape)


# test function.
class TestCumprodReverseCase1(TestCumprod):
    def init_params(self):
        self.shape = (120,)
        self.zero_nums = [0, 1, 10]

    # test backward.
    def test_check_grad(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in self.zero_nums:
                self.prepare_inputs_outputs_attrs(dim, zero_num)
                self.init_grad_input_output(dim)
                if self.dtype == np.float64:
                    self.check_grad(
                        ['X'], 'Out', check_pir=True, max_relative_error=2e-7
                    )
                else:
                    self.check_grad(
                        ['X'],
                        'Out',
                        user_defined_grads=[self.grad_x],
                        user_defined_grad_outputs=[self.grad_out],
                        check_pir=True,
                    )


# test function.
class TestCumprodReverseCase2(TestCumprod):
    def init_params(self):
        self.shape = (12, 10)
        self.zero_nums = [0, 1, 10]


# test function.
class TestCumprodReverseCase3(TestCumprod):
    def init_params(self):
        self.shape = (3, 4, 10)
        self.zero_nums = [0, 1, 10]


# test function.
class TestCumprodReverseCase4(TestCumprod):
    def init_params(self):
        self.shape = (2, 3, 4, 5, 2)
        self.zero_nums = [0, 1, 10]

    # test backward.
    def test_check_grad(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in self.zero_nums:
                self.prepare_inputs_outputs_attrs(dim, zero_num)
                self.init_grad_input_output(dim)
                if self.dtype == np.float64:
                    self.check_grad(
                        ['X'], 'Out', check_pir=True, max_relative_error=3e-7
                    )
                else:
                    self.check_grad(
                        ['X'],
                        'Out',
                        user_defined_grads=[self.grad_x],
                        user_defined_grad_outputs=[self.grad_out],
                        check_pir=True,
                    )


# test function.
class TestCumprodExclusive(TestCumprod):
    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype) + 0.5
        )
        if zero_num > 0:
            zero_num = min(zero_num, self.x.size)
            shape = self.x.shape
            self.x = self.x.flatten()
            indices = random.sample(range(self.x.size), zero_num)
            for i in indices:
                self.x[i] = 0
            self.x = np.reshape(self.x, self.shape)
        ones_shape = list(self.shape)
        ones_shape[dim] = 1
        if dim == -4 or dim == 0:
            x_temp = self.x[:-1, :, :, :]
        elif dim == -3 or dim == 1:
            x_temp = self.x[:, :-1, :, :]
        elif dim == -2 or dim == 2:
            x_temp = self.x[:, :, :-1, :]
        elif dim == -1 or dim == 3:
            x_temp = self.x[:, :, :, :-1]
        self.out = np.concatenate(
            (
                np.ones(ones_shape, dtype=self.dtype),
                x_temp.cumprod(axis=dim),
            ),
            axis=dim,
        )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
        self.attrs = {'dim': dim, 'exclusive': True}

    def init_grad_input_output(self, dim):
        reshape_x = self.x.reshape(self.x.size)
        self.grad_out = np.ones(self.x.size, self.val_dtype)
        self.grad_x = np.zeros(self.x.size, self.val_dtype)
        out_data = self.out.reshape(self.x.size)
        if self.dtype == np.complex128 or self.dtype == np.complex64:
            reshape_x = np.conj(reshape_x)
            out_data = np.conj(out_data)
        cumprod_grad(
            reshape_x,
            out_data,
            self.grad_out,
            self.grad_x,
            self.shape,
            dim,
            exclusive=True,
            reverse=False,
        )
        if self.dtype == np.uint16:
            self.grad_x = convert_float_to_uint16(
                self.grad_x.reshape(self.shape)
            )
            self.grad_out = convert_float_to_uint16(
                self.grad_out.reshape(self.shape)
            )
        else:
            self.grad_x = self.grad_x.reshape(self.shape)
            self.grad_out = self.grad_out.reshape(self.shape)


# test function.
class TestCumprodExclusiveCase1(TestCumprodExclusive):
    def init_params(self):
        self.shape = (120,)
        self.zero_nums = [0, 1, 10]

    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype) + 0.5
        )
        if zero_num > 0:
            zero_num = min(zero_num, self.x.size)
            shape = self.x.shape
            self.x = self.x.flatten()
            indices = random.sample(range(self.x.size), zero_num)
            for i in indices:
                self.x[i] = 0
            self.x = np.reshape(self.x, self.shape)
        ones_shape = (1,)
        x_temp = self.x[:-1]
        self.out = np.concatenate(
            (
                np.ones(ones_shape, dtype=self.dtype),
                x_temp.cumprod(axis=dim),
            ),
            axis=dim,
        )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
        self.attrs = {'dim': dim, 'exclusive': True}

    # test backward.
    def test_check_grad(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in self.zero_nums:
                self.prepare_inputs_outputs_attrs(dim, zero_num)
                self.init_grad_input_output(dim)
                if self.dtype == np.float64:
                    self.check_grad(
                        ['X'], 'Out', check_pir=True, max_relative_error=2e-7
                    )
                else:
                    self.check_grad(
                        ['X'],
                        'Out',
                        user_defined_grads=[self.grad_x],
                        user_defined_grad_outputs=[self.grad_out],
                        check_pir=True,
                    )


# test function.
class TestCumprodExclusiveCase2(TestCumprodExclusive):
    def init_params(self):
        self.shape = (12, 10)
        self.zero_nums = [0, 1, 10]

    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype) + 0.5
        )
        if zero_num > 0:
            zero_num = min(zero_num, self.x.size)
            shape = self.x.shape
            self.x = self.x.flatten()
            indices = random.sample(range(self.x.size), zero_num)
            for i in indices:
                self.x[i] = 0
            self.x = np.reshape(self.x, self.shape)
        ones_shape = list(self.shape)
        ones_shape[dim] = 1
        if dim == -2 or dim == 0:
            x_temp = self.x[:-1, :]
        elif dim == -1 or dim == 1:
            x_temp = self.x[:, :-1]
        self.out = np.concatenate(
            (
                np.ones(ones_shape, dtype=self.dtype),
                x_temp.cumprod(axis=dim),
            ),
            axis=dim,
        )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
        self.attrs = {'dim': dim, 'exclusive': True}


# test function.
class TestCumprodExclusiveCase3(TestCumprodExclusive):
    def init_params(self):
        self.shape = (3, 4, 10)
        self.zero_nums = [0, 1, 10]

    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype) + 0.5
        )
        if zero_num > 0:
            zero_num = min(zero_num, self.x.size)
            shape = self.x.shape
            self.x = self.x.flatten()
            indices = random.sample(range(self.x.size), zero_num)
            for i in indices:
                self.x[i] = 0
            self.x = np.reshape(self.x, self.shape)
        ones_shape = list(self.shape)
        ones_shape[dim] = 1
        if dim == -3 or dim == 0:
            x_temp = self.x[:-1, :, :]
        elif dim == -2 or dim == 1:
            x_temp = self.x[:, :-1, :]
        elif dim == -1 or dim == 2:
            x_temp = self.x[:, :, :-1]
        self.out = np.concatenate(
            (
                np.ones(ones_shape, dtype=self.dtype),
                x_temp.cumprod(axis=dim),
            ),
            axis=dim,
        )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
        self.attrs = {'dim': dim, 'exclusive': True}


# test function.
class TestCumprodExclusiveCase4(TestCumprodExclusive):
    def init_params(self):
        self.shape = (2, 3, 4, 5, 2)
        self.zero_nums = [0, 1, 10]

    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype) + 0.5
        )
        if zero_num > 0:
            zero_num = min(zero_num, self.x.size)
            shape = self.x.shape
            self.x = self.x.flatten()
            indices = random.sample(range(self.x.size), zero_num)
            for i in indices:
                self.x[i] = 0
            self.x = np.reshape(self.x, self.shape)
        ones_shape = list(self.shape)
        ones_shape[dim] = 1
        if dim == -5 or dim == 0:
            x_temp = self.x[:-1, :, :, :, :]
        elif dim == -4 or dim == 1:
            x_temp = self.x[:, :-1, :, :, :]
        elif dim == -3 or dim == 2:
            x_temp = self.x[:, :, :-1, :, :]
        elif dim == -2 or dim == 3:
            x_temp = self.x[:, :, :, :-1, :]
        elif dim == -1 or dim == 4:
            x_temp = self.x[:, :, :, :, :-1]
        self.out = np.concatenate(
            (
                np.ones(ones_shape, dtype=self.dtype),
                x_temp.cumprod(axis=dim),
            ),
            axis=dim,
        )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
        self.attrs = {'dim': dim, 'exclusive': True}

    # test backward.
    def test_check_grad(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in self.zero_nums:
                self.prepare_inputs_outputs_attrs(dim, zero_num)
                self.init_grad_input_output(dim)
                if self.dtype == np.float64:
                    self.check_grad(
                        ['X'], 'Out', check_pir=True, max_relative_error=2e-7
                    )
                else:
                    self.check_grad(
                        ['X'],
                        'Out',
                        user_defined_grads=[self.grad_x],
                        user_defined_grad_outputs=[self.grad_out],
                        check_pir=True,
                    )


# # test function.
class TestCumprodExclusiveAndReverse(TestCumprod):
    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype) + 0.5
        )
        if zero_num > 0:
            zero_num = min(zero_num, self.x.size)
            shape = self.x.shape
            self.x = self.x.flatten()
            indices = random.sample(range(self.x.size), zero_num)
            for i in indices:
                self.x[i] = 0
            self.x = np.reshape(self.x, self.shape)
        ones_shape = list(self.shape)
        ones_shape[dim] = 1
        if dim == -4 or dim == 0:
            x_temp = self.x[1:, :, :, :]
        elif dim == -3 or dim == 1:
            x_temp = self.x[:, 1:, :, :]
        elif dim == -2 or dim == 2:
            x_temp = self.x[:, :, 1:, :]
        elif dim == -1 or dim == 3:
            x_temp = self.x[:, :, :, 1:]
        self.out = np.flip(
            np.concatenate(
                (
                    np.ones(ones_shape, dtype=self.dtype),
                    np.flip(x_temp, axis=dim).cumprod(axis=dim),
                ),
                axis=dim,
            ),
            axis=dim,
        )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
        self.attrs = {'dim': dim, 'exclusive': True, 'reverse': True}

    def init_grad_input_output(self, dim):
        reshape_x = self.x.reshape(self.x.size)
        self.grad_out = np.ones(self.x.size, self.val_dtype)
        self.grad_x = np.zeros(self.x.size, self.val_dtype)
        out_data = self.out.reshape(self.x.size)
        if self.dtype == np.complex128 or self.dtype == np.complex64:
            reshape_x = np.conj(reshape_x)
            out_data = np.conj(out_data)
        cumprod_grad(
            reshape_x,
            out_data,
            self.grad_out,
            self.grad_x,
            self.shape,
            dim,
            exclusive=True,
            reverse=True,
        )
        if self.dtype == np.uint16:
            self.grad_x = convert_float_to_uint16(
                self.grad_x.reshape(self.shape)
            )
            self.grad_out = convert_float_to_uint16(
                self.grad_out.reshape(self.shape)
            )
        else:
            self.grad_x = self.grad_x.reshape(self.shape)
            self.grad_out = self.grad_out.reshape(self.shape)


class TestCumprodExclusiveAndReverseCase1(TestCumprodExclusiveAndReverse):
    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def init_params(self):
        self.shape = (120,)
        self.zero_nums = [0, 1, 10]

    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype) + 0.5
        )
        if zero_num > 0:
            zero_num = min(zero_num, self.x.size)
            shape = self.x.shape
            self.x = self.x.flatten()
            indices = random.sample(range(self.x.size), zero_num)
            for i in indices:
                self.x[i] = 0
            self.x = np.reshape(self.x, self.shape)
        ones_shape = list(self.shape)
        ones_shape[dim] = 1
        if dim == -1 or dim == 0:
            x_temp = self.x[1:]
        self.out = np.flip(
            np.concatenate(
                (
                    np.ones(ones_shape, dtype=self.dtype),
                    np.flip(x_temp, axis=dim).cumprod(axis=dim),
                ),
                axis=dim,
            ),
            axis=dim,
        )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
        self.attrs = {'dim': dim, 'exclusive': True, 'reverse': True}


class TestCumprodExclusiveAndReverseCase2(TestCumprodExclusiveAndReverse):
    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def init_params(self):
        self.shape = (12, 10)
        self.zero_nums = [0, 1, 10]

    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype) + 0.5
        )
        if zero_num > 0:
            zero_num = min(zero_num, self.x.size)
            shape = self.x.shape
            self.x = self.x.flatten()
            indices = random.sample(range(self.x.size), zero_num)
            for i in indices:
                self.x[i] = 0
            self.x = np.reshape(self.x, self.shape)
        ones_shape = list(self.shape)
        ones_shape[dim] = 1
        if dim == -2 or dim == 0:
            x_temp = self.x[1:, :]
        elif dim == -1 or dim == 1:
            x_temp = self.x[:, 1:]
        self.out = np.flip(
            np.concatenate(
                (
                    np.ones(ones_shape, dtype=self.dtype),
                    np.flip(x_temp, axis=dim).cumprod(axis=dim),
                ),
                axis=dim,
            ),
            axis=dim,
        )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
        self.attrs = {'dim': dim, 'exclusive': True, 'reverse': True}


class TestCumprodExclusiveAndReverseCase3(TestCumprodExclusiveAndReverse):
    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def init_params(self):
        self.shape = (3, 4, 10)
        self.zero_nums = [0, 1, 10]

    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype) + 0.5
        )
        if zero_num > 0:
            zero_num = min(zero_num, self.x.size)
            shape = self.x.shape
            self.x = self.x.flatten()
            indices = random.sample(range(self.x.size), zero_num)
            for i in indices:
                self.x[i] = 0
            self.x = np.reshape(self.x, self.shape)
        ones_shape = list(self.shape)
        ones_shape[dim] = 1
        if dim == -3 or dim == 0:
            x_temp = self.x[1:, :, :]
        elif dim == -2 or dim == 1:
            x_temp = self.x[:, 1:, :]
        elif dim == -1 or dim == 2:
            x_temp = self.x[:, :, 1:]
        self.out = np.flip(
            np.concatenate(
                (
                    np.ones(ones_shape, dtype=self.dtype),
                    np.flip(x_temp, axis=dim).cumprod(axis=dim),
                ),
                axis=dim,
            ),
            axis=dim,
        )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
        self.attrs = {'dim': dim, 'exclusive': True, 'reverse': True}


class TestCumprodExclusiveAndReverseCase4(TestCumprodExclusiveAndReverse):
    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def init_params(self):
        self.shape = (2, 3, 4, 5, 2)
        self.zero_nums = [0, 1, 10]

    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype) + 0.5
        )
        if zero_num > 0:
            zero_num = min(zero_num, self.x.size)
            shape = self.x.shape
            self.x = self.x.flatten()
            indices = random.sample(range(self.x.size), zero_num)
            for i in indices:
                self.x[i] = 0
            self.x = np.reshape(self.x, self.shape)
        ones_shape = list(self.shape)
        ones_shape[dim] = 1
        if dim == -5 or dim == 0:
            x_temp = self.x[1:, :, :, :, :]
        elif dim == -4 or dim == 1:
            x_temp = self.x[:, 1:, :, :, :]
        elif dim == -3 or dim == 2:
            x_temp = self.x[:, :, 1:, :, :]
        elif dim == -2 or dim == 3:
            x_temp = self.x[:, :, :, 1:, :]
        elif dim == -1 or dim == 4:
            x_temp = self.x[:, :, :, :, 1:]
        self.out = np.flip(
            np.concatenate(
                (
                    np.ones(ones_shape, dtype=self.dtype),
                    np.flip(x_temp, axis=dim).cumprod(axis=dim),
                ),
                axis=dim,
            ),
            axis=dim,
        )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
        self.attrs = {'dim': dim, 'exclusive': True, 'reverse': True}


# # test function.
class TestCumprodOuter1AndInner1(OpTest):  # used to pass ci-coverage
    def init_params(self):
        self.shape = (1, 100, 1)

    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def setUp(self):
        paddle.enable_static()
        self.init_params()
        self.init_dtype()
        self.op_type = "cumprod"
        self.python_api = cumprod_wrapper
        self.inputs = {'X': None}
        self.outputs = {'Out': None}
        self.attrs = {'dim': None}

    def prepare_inputs_outputs_attrs(self, reverse):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype) + 0.5
        )
        if reverse:
            self.out = np.flip(
                np.concatenate(
                    (
                        np.ones((1, 1, 1), dtype=self.dtype),
                        np.flip(self.x, axis=1)[:, :-1, :].cumprod(axis=1),
                    ),
                    axis=1,
                ),
                axis=1,
            )
        else:
            self.out = np.concatenate(
                (
                    np.ones((1, 1, 1), dtype=self.dtype),
                    self.x[:, :-1, :].cumprod(axis=1),
                ),
                axis=1,
            )
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            self.outputs = {'Out': convert_float_to_uint16(self.out)}
        else:
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
        self.attrs = {'dim': 1, 'exclusive': True, 'reverse': reverse}

    def init_grad_input_output(self, reverse):
        reshape_x = self.x.reshape(self.x.size)
        self.grad_out = np.ones(self.x.size, self.val_dtype)
        self.grad_x = np.zeros(self.x.size, self.val_dtype)
        out_data = self.out.reshape(self.x.size)
        if self.dtype == np.complex128 or self.dtype == np.complex64:
            reshape_x = np.conj(reshape_x)
            out_data = np.conj(out_data)
        cumprod_grad(
            reshape_x,
            out_data,
            self.grad_out,
            self.grad_x,
            self.shape,
            1,
            exclusive=True,
            reverse=reverse,
        )
        if self.dtype == np.uint16:
            self.grad_x = convert_float_to_uint16(
                self.grad_x.reshape(self.shape)
            )
            self.grad_out = convert_float_to_uint16(
                self.grad_out.reshape(self.shape)
            )
        else:
            self.grad_x = self.grad_x.reshape(self.shape)
            self.grad_out = self.grad_out.reshape(self.shape)

    # test forward.
    def test_check_output(self):
        self.prepare_inputs_outputs_attrs(reverse=True)
        self.check_output(check_pir=True)
        self.prepare_inputs_outputs_attrs(reverse=False)
        self.check_output(check_pir=True)

    # test backward.
    def test_check_grad(self):
        for reverse in [True, False]:
            self.prepare_inputs_outputs_attrs(reverse)
            self.init_grad_input_output(reverse)
            if self.dtype == np.float64:
                self.check_grad(['X'], 'Out', check_pir=True)
            else:
                self.check_grad(
                    ['X'],
                    'Out',
                    user_defined_grads=[self.grad_x],
                    user_defined_grad_outputs=[self.grad_out],
                    check_pir=True,
                )


if __name__ == "__main__":
    unittest.main()
