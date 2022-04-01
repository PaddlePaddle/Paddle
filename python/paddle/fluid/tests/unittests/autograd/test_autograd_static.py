# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from utils import _compute_numerical_jacobian, _compute_numerical_batch_jacobian


def approx_jacobian(f, xs, dtype, eps=1e-5, batch=False):
    r"""Computes an approximate Jacobian matrix of a multi-valued function 
    using finite differences.
    
    The function input is required to be an np array or a list of list of np 
    arrays. 
    """

    def flatten(x):
        if len(x.shape) > 0:
            to = [x.shape[0], -1] if batch else [-1]
            return x.reshape(to)
        else:
            return x

    def flatten_all(xs):
        if isinstance(xs, list):
            flattened = np.concatenate([flatten(x) for x in xs], axis=-1)
        else:
            flattened = flatten(xs)
        return flattened

    def x_like(x, orig_x):
        return x.reshape(orig_x.shape)

    def _f(x):
        if multi_inps:
            _xs = np.split(x, splits, axis=-1)
            _xs = [x_like(_x, _o) for _x, _o in zip(_xs, xs)]
            outs = f(_xs)
        else:
            outs = f(x)
        return flatten_all(outs)

    multi_inps = False if isinstance(xs, np.ndarray) else True
    x = flatten_all(xs)
    xdim = x.shape[-1]
    splits = []

    if multi_inps:
        split = 0
        for inp in xs:
            split += flatten(inp).shape[-1]
            splits.append(split)

    ds = eps * np.eye(xdim, dtype=dtype)

    fprimes_by_x = [(0.5 * (_f(x + d) - _f(x - d)) / eps) for d in ds]
    fprimes_by_y = np.stack(fprimes_by_x, axis=-1)
    return np.transpose(fprimes_by_y, [1, 0, 2]) if batch else fprimes_by_y


def make_tensors(inps):
    if isinstance(inps, list):
        xs = [
            paddle.static.data(
                f'x{i}', inp.shape, dtype=inp.dtype)
            for i, inp in enumerate(inps)
        ]
    else:
        xs = paddle.static.data(name='x', shape=inps.shape, dtype=inps.dtype)
    return xs


all_data_shapes = {
    'A': [[1., 2.]],
    'B': [[1., 2.], [2., 1.]],
    'C': [[2., 2.], [2., 1.]],
    'D': [[[2., 2.], [2., 1.]], [[1., 2.], [2., 1.]]],
    'E': [[[3., 4.], [2., 3.]], [[2., 1.], [1., 3.]]],
}


def prepare_data(test, input_shapes, dtype):
    for name, shape in input_shapes.items():
        setattr(test, name, np.array(shape, dtype=dtype))


class TestJacobianFloat32(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        paddle.enable_static()
        if fluid.core.is_compiled_with_cuda():
            self.place = fluid.CUDAPlace(0)
        else:
            self.place = fluid.CPUPlace()
        self.dtype = 'float32'
        prepare_data(self, all_data_shapes, self.dtype)
        self.eps = 1e-4
        self.rtol = 1e-2
        self.atol = 1e-2

    def run_test_by_fullmatrix(self, pd_f, np_f, inps, batch=False):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            xs = make_tensors(inps)
            JJ = paddle.autograd.functional.Jacobian(pd_f, xs, batch=batch)
            nrow, ncol = JJ.shape()
            full_jacobian = JJ[:]
        exe = fluid.Executor(self.place)
        exe.run(startup)
        if isinstance(inps, list):
            feeds = {f'x{i}': x for i, x in enumerate(inps)}
        else:
            feeds = {'x': inps}
        pd_jacobians = exe.run(main, feed=feeds, fetch_list=[full_jacobian])[0]
        np_jacobians = approx_jacobian(
            np_f, inps, self.dtype, self.eps, batch=batch)
        self.assertTrue(
            np.allclose(pd_jacobians, np_jacobians, self.rtol, self.atol))

    def run_test_by_rows(self, pd_f, np_f, inps, batch=False):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            xs = make_tensors(inps)
            JJ = paddle.autograd.functional.Jacobian(pd_f, xs, batch=batch)
            nrow, ncol = JJ.shape()
            rows = [JJ[i] for i in range(nrow)]
        exe = fluid.Executor(self.place)
        exe.run(startup)
        if isinstance(inps, list):
            feeds = {f'x{i}': x for i, x in enumerate(inps)}
        else:
            feeds = {'x': inps}
        pd_jac = exe.run(main, feed=feeds, fetch_list=[rows])
        np_jac = approx_jacobian(np_f, inps, self.dtype, self.eps, batch=batch)
        for i in range(nrow):
            self.assertTrue(
                np.allclose(pd_jac[i], np_jac[i], self.rtol, self.atol))

    def run_test_by_entries(self, pd_f, np_f, inps, batch=False):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            xs = make_tensors(inps)
            JJ = paddle.autograd.functional.Jacobian(pd_f, xs, batch=batch)
            nrow, ncol = JJ.shape()
            entries = [JJ[i, j] for i in range(nrow) for j in range(ncol)]
        exe = fluid.Executor(self.place)
        exe.run(startup)
        if isinstance(inps, list):
            feeds = {f'x{i}': x for i, x in enumerate(inps)}
        else:
            feeds = {'x': inps}
        pd_entries = exe.run(main, feed=feeds, fetch_list=[entries])
        np_jac = approx_jacobian(np_f, inps, self.dtype, self.eps, batch=batch)
        np_entries = [
            np_jac[i, ..., j] for i in range(nrow) for j in range(ncol)
        ]
        for pd_entry, np_entry in zip(pd_entries, np_entries):
            self.assertTrue(
                np.allclose(pd_entry, np_entry, self.rtol, self.atol))

    def test_square(self):
        def pd_f(x):
            return paddle.multiply(x, x)

        def np_f(x):
            return np.multiply(x, x)

        self.run_test_by_fullmatrix(pd_f, np_f, self.A)
        self.run_test_by_rows(pd_f, np_f, self.A)
        self.run_test_by_entries(pd_f, np_f, self.A)

    def test_mul(self):
        def pd_f(xs):
            x, y = xs
            return paddle.multiply(x, y)

        def np_f(xs):
            x, y = xs
            return np.multiply(x, y)

        self.run_test_by_fullmatrix(
            pd_f,
            np_f,
            [self.B, self.C], )
        self.run_test_by_rows(pd_f, np_f, [self.B, self.C])
        self.run_test_by_entries(pd_f, np_f, [self.B, self.C])

    def test_matmul(self):
        def pd_f(xs):
            x, y = xs
            return paddle.matmul(x, y)

        def np_f(xs):
            x, y = xs
            return np.matmul(x, y)

        self.run_test_by_fullmatrix(pd_f, np_f, [self.B, self.C])
        self.run_test_by_rows(pd_f, np_f, [self.B, self.C])
        self.run_test_by_entries(pd_f, np_f, [self.B, self.C])

    def test_batch_matmul(self):
        def pd_f(xs):
            x, y = xs
            return paddle.matmul(x, y)

        def np_f(xs):
            x, y = xs
            return np.matmul(x, y)

        self.run_test_by_fullmatrix(pd_f, np_f, [self.D, self.E], batch=True)
        self.run_test_by_rows(pd_f, np_f, [self.D, self.E], batch=True)
        self.run_test_by_entries(pd_f, np_f, [self.D, self.E], batch=True)


class TestJacobianFloat64(TestJacobianFloat32):
    @classmethod
    def setUpClass(self):
        paddle.enable_static()
        if fluid.core.is_compiled_with_cuda():
            self.place = fluid.CUDAPlace(0)
        else:
            self.place = fluid.CPUPlace()
        self.dtype = 'float64'
        prepare_data(self, all_data_shapes, self.dtype)
        self.eps = 1e-7
        self.rtol = 1e-6
        self.atol = 1e-6


class TestHessianFloat64(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        paddle.enable_static()
        if fluid.core.is_compiled_with_cuda():
            self.place = fluid.CUDAPlace(0)
        else:
            self.place = fluid.CPUPlace()
        self.dtype = 'float64'
        prepare_data(self, all_data_shapes, self.dtype)
        self.eps = 1e-7
        self.rtol = 1e-6
        self.atol = 1e-6

    def run_test_by_fullmatrix(self, pd_f, inps, np_hess, batch=False):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            xs = make_tensors(inps)
            HH = paddle.autograd.functional.Hessian(pd_f, xs, batch=batch)
            nrow, ncol = HH.shape()
            full_hessian = HH[:]
        exe = fluid.Executor(self.place)
        exe.run(startup)
        if isinstance(inps, list):
            feeds = {f'x{i}': x for i, x in enumerate(inps)}
        else:
            feeds = {'x': inps}
        pd_hess = exe.run(main, feed=feeds, fetch_list=[full_hessian])[0]
        self.assertTrue(np.allclose(pd_hess, np_hess, self.rtol, self.atol))

    def test_square(self):
        def pd_f(x):
            """Input is a square matrix."""
            return paddle.matmul(x, x.T)

        def np_hess(x):
            dim = x.shape[0]
            f_xx_upperleft = 2 * np.eye(dim, dtype=self.dtype)
            f_xx = np.zeros([dim * dim, dim * dim], dtype=self.dtype)
            f_xx[:dim, :dim] = f_xx_upperleft
            return f_xx

        self.run_test_by_fullmatrix(pd_f, self.B, np_hess(self.B))

        def test_batch_square(self):
            def pd_f(x):
                """Input is a square matrix."""
                return paddle.matmul(x, paddle.transpose(x, [0, 2, 1]))

            def np_hess(x):
                bat, dim, _ = x.shape
                f_xx_upperleft = 2 * np.eye(dim, dtype=self.dtype)
                f_xx = np.zeros([bat, dim * dim, dim * dim], dtype=self.dtype)
                f_xx[..., :dim, :dim] = f_xx_upperleft
                return f_xx

            self.run_test_by_fullmatrix(
                pd_f, self.E, np_hess(self.E), batch=True)


if __name__ == "__main__":
    unittest.main()
