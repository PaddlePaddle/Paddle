# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

np.random.seed(0)


# define cumprod grad function.
def cumprod_grad(x, y, dy, dx, shape, dim):
    if dim < 0:
        dim += len(shape)
    mid_dim = shape[dim]
    outer_dim = 1
    inner_dim = 1
    for i in range(0, dim):
        outer_dim *= shape[i]
    for i in range(dim + 1, len(shape)):
        inner_dim *= shape[i]
    for i in range(outer_dim):
        for k in range(inner_dim):
            for j in range(mid_dim):
                index = i * mid_dim * inner_dim + j * inner_dim + k
                for n in range(mid_dim):
                    pos = i * mid_dim * inner_dim + n * inner_dim + k
                    elem = 0
                    if j == 0:
                        elem = dy[pos]
                    else:
                        elem = dy[pos] * y[index - inner_dim]
                    if pos > index:
                        for m in range(
                            index + inner_dim, pos + inner_dim, inner_dim
                        ):
                            elem *= x[m]
                    elif pos < index:
                        elem = 0
                    dx[index] += elem


# test function.
class XPUTestCumprodOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'cumprod'
        self.use_dynamic_create_class = False

    class TestCumprod(XPUOpTest):
        def init_params(self):
            self.shape = (2, 3, 4, 5)
            self.zero_nums = [0, 10, 20, 30, int(np.prod(self.shape))]

        def init_dtype(self):
            self.dtype = self.in_type

        def setUp(self):
            paddle.enable_static()
            self.place = paddle.XPUPlace(0)
            self.init_params()
            self.init_dtype()
            self.op_type = "cumprod"
            self.python_api = paddle.cumprod
            self.inputs = {'X': None}
            self.outputs = {'Out': None}
            self.attrs = {'dim': None}

        def prepare_inputs_outputs_attrs(self, dim, zero_num):
            self.x = np.random.random(self.shape).astype(self.dtype) + 0.5
            if zero_num > 0:
                zero_num = min(zero_num, self.x.size)
                shape = self.x.shape
                self.x = self.x.flatten()
                indices = random.sample(range(self.x.size), zero_num)
                for i in indices:
                    self.x[i] = 0
                self.x = np.reshape(self.x, self.shape)
            self.out = np.cumprod(self.x, axis=dim)
            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.out}
            self.attrs = {'dim': dim}

        def init_grad_input_output(self, dim):
            reshape_x = self.x.reshape(self.x.size)
            self.grad_out = np.ones(self.x.size, self.dtype)
            self.grad_x = np.zeros(self.x.size, self.dtype)
            out_data = self.out.reshape(self.x.size)
            if self.dtype == np.complex128 or self.dtype == np.complex64:
                reshape_x = np.conj(reshape_x)
                out_data = np.conj(out_data)
            cumprod_grad(
                reshape_x, out_data, self.grad_out, self.grad_x, self.shape, dim
            )
            self.grad_x = self.grad_x.reshape(self.shape)
            self.grad_out = self.grad_out.reshape(self.shape)

        # test forward.
        def test_check_output(self):
            for dim in range(-len(self.shape), len(self.shape)):
                for zero_num in self.zero_nums:
                    self.prepare_inputs_outputs_attrs(dim, zero_num)
                    self.check_output_with_place(self.place)

        # test backward.
        def test_check_grad(self):
            pass

    # test api.
    class TestCumprodAPI(unittest.TestCase):
        def init_dtype(self):
            self.dtype = 'float32'
            self.shape = [2, 3, 10, 10]

        def setUp(self):
            paddle.enable_static()
            self.init_dtype()
            self.x = (np.random.rand(2, 3, 10, 10) + 0.5).astype(self.dtype)
            self.place = [paddle.XPUPlace(0)]

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


support_types = get_xpu_op_support_types('cumprod')
for stype in support_types:
    create_test_class(globals(), XPUTestCumprodOP, stype)

if __name__ == "__main__":
    unittest.main()
