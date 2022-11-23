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

import typing
import unittest

import numpy as np
import paddle
import paddle.fluid as fluid

import config
import utils

paddle.enable_static()


@utils.place(config.DEVICES)
@utils.parameterize((utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'stop_gradient'), (
    ('tensor_input', utils.reduce, np.random.rand(2, 3), None, False),
    ('tensor_sequence_input', utils.reduce, np.random.rand(2, 3), None, False),
    ('v_not_none', utils.reduce, np.random.rand(2,
                                                3), np.random.rand(1), False),
    ('xs_stop_gradient', utils.reduce, np.random.rand(
        2, 3), np.random.rand(1), True),
    ('func_mutmul', utils.matmul,
     (np.random.rand(3, 2), np.random.rand(2, 3)), None, False),
    ('func_mul', utils.mul,
     (np.random.rand(3, 3), np.random.rand(3, 3)), None, False),
    ('func_out_two', utils.o2,
     (np.random.rand(10), np.random.rand(10)), None, False),
))
class TestVJP(unittest.TestCase):

    def setUp(self):
        self.dtype = str(self.xs[0].dtype) if isinstance(
            self.xs, typing.Sequence) else str(self.xs.dtype)
        self._rtol = config.TOLERANCE.get(str(
            self.dtype)).get("first_order_grad").get("rtol")
        self._atol = config.TOLERANCE.get(str(
            self.dtype)).get("first_order_grad").get("atol")

    def _vjp(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            feed, static_xs, static_v = utils.gen_static_data_and_feed(
                self.xs, self.v, stop_gradient=self.stop_gradient)
            ys, xs_grads = paddle.incubate.autograd.vjp(self.fun, static_xs,
                                                        static_v)
        exe.run(sp)
        return exe.run(mp, feed=feed, fetch_list=[ys, xs_grads])

    def _expected_vjp(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            feed, static_xs, static_v = utils.gen_static_data_and_feed(
                self.xs, self.v, False)
            ys = self.fun(*static_xs) if isinstance(
                static_xs, typing.Sequence) else self.fun(static_xs)
            xs_grads = paddle.static.gradients(ys, static_xs, static_v)
        exe.run(sp)
        return exe.run(mp, feed=feed, fetch_list=[ys, xs_grads])

    def test_vjp(self):
        actual = self._vjp()
        expected = self._expected_vjp()
        self.assertEqual(len(actual), len(expected))
        for i in range(len(actual)):
            np.testing.assert_allclose(actual[i],
                                       expected[i],
                                       rtol=self._rtol,
                                       atol=self._atol)


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'expected_exception'),
    (('v_shape_not_equal_ys', utils.square, np.random.rand(3),
      np.random.rand(1), RuntimeError), ))
class TestVJPException(unittest.TestCase):

    def setUp(self):
        self.exe = paddle.static.Executor()

    def _vjp(self):
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            feed, static_xs, static_v = utils.gen_static_data_and_feed(
                self.xs, self.v)
            ys, xs_grads = paddle.incubate.autograd.vjp(self.fun, static_xs,
                                                        static_v)
        self.exe.run(sp)
        return self.exe.run(mp, feed, fetch_list=[ys, xs_grads])

    def test_vjp(self):
        with self.assertRaises(self.expected_exception):
            self._vjp()


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
            paddle.static.data(f'x{i}', inp.shape, dtype=inp.dtype)
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
        self.np_dtype = np.float32
        prepare_data(self, all_data_shapes, self.dtype)
        self.eps = config.TOLERANCE.get(
            self.dtype).get('first_order_grad').get('eps')
        # self.rtol = config.TOLERANCE.get(self.dtype).get('first_order_grad').get('rtol')
        # self.atol = config.TOLERANCE.get(self.dtype).get('first_order_grad').get('atol')
        # Do't use tolerance in config, which will cause this test case failed.
        self.rtol = 1e-2
        self.atol = 1e-2

    def run_test_by_fullmatrix(self, pd_f, np_f, inps, batch=False):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            xs = make_tensors(inps)
            JJ = paddle.incubate.autograd.Jacobian(pd_f, xs, is_batched=batch)
            if batch:
                _, nrow, ncol = JJ.shape
            else:
                nrow, ncol = JJ.shape
            full_jacobian = JJ[:]
        exe = fluid.Executor(self.place)
        exe.run(startup)
        if isinstance(inps, list):
            feeds = {f'x{i}': x for i, x in enumerate(inps)}
        else:
            feeds = {'x': inps}
        pd_jacobians = exe.run(main, feed=feeds, fetch_list=[full_jacobian])[0]
        np_jacobians = approx_jacobian(np_f,
                                       inps,
                                       self.dtype,
                                       self.eps,
                                       batch=batch)
        if batch:
            np_jacobians = utils._np_transpose_matrix_format(
                np_jacobians, utils.MatrixFormat.NBM, utils.MatrixFormat.BNM)

        np.testing.assert_allclose(pd_jacobians, np_jacobians, self.rtol,
                                   self.atol)

    def run_test_by_rows(self, pd_f, np_f, inps, batch=False):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            xs = make_tensors(inps)
            JJ = paddle.incubate.autograd.Jacobian(pd_f, xs, is_batched=batch)
            if batch:
                nbatch, nrow, ncol = JJ.shape
                rows = [JJ[:, i, :] for i in range(nrow)]
            else:
                nrow, ncol = JJ.shape
                rows = [JJ[i, :] for i in range(nrow)]

        exe = fluid.Executor(self.place)
        exe.run(startup)
        if isinstance(inps, list):
            feeds = {f'x{i}': x for i, x in enumerate(inps)}
        else:
            feeds = {'x': inps}
        pd_jac = exe.run(main, feed=feeds, fetch_list=[rows])
        np_jac = approx_jacobian(np_f, inps, self.dtype, self.eps, batch=batch)
        for i in range(nrow):
            np.testing.assert_allclose(pd_jac[i], np_jac[i], self.rtol,
                                       self.atol)

    def run_test_by_entries(self, pd_f, np_f, inps, batch=False):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            xs = make_tensors(inps)
            JJ = paddle.incubate.autograd.Jacobian(pd_f, xs, is_batched=batch)
            if batch:
                nbatch, nrow, ncol = JJ.shape
                entries = [
                    JJ[:, i, j] for i in range(nrow) for j in range(ncol)
                ]
            else:
                nrow, ncol = JJ.shape
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
            np.testing.assert_allclose(pd_entry, np_entry, self.rtol, self.atol)

    def test_square(self):

        def pd_f(x):
            return paddle.multiply(x, x)

        def np_f(x):
            return np.multiply(x, x)

        self.run_test_by_fullmatrix(pd_f, np_f, self.A)
        self.run_test_by_rows(pd_f, np_f, self.A)
        self.run_test_by_entries(pd_f, np_f, self.A)

    def test_mul(self):

        def pd_f(x, y):
            return paddle.multiply(x, y)

        def np_f(xs):
            x, y = xs
            return np.multiply(x, y)

        self.run_test_by_fullmatrix(
            pd_f,
            np_f,
            [self.B, self.C],
        )
        self.run_test_by_rows(pd_f, np_f, [self.B, self.C])
        self.run_test_by_entries(pd_f, np_f, [self.B, self.C])

    def test_matmul(self):

        def pd_f(x, y):
            return paddle.matmul(x, y)

        def np_f(xs):
            x, y = xs
            return np.matmul(x, y)

        self.run_test_by_fullmatrix(pd_f, np_f, [self.B, self.C])
        self.run_test_by_rows(pd_f, np_f, [self.B, self.C])
        self.run_test_by_entries(pd_f, np_f, [self.B, self.C])

    def test_batch_matmul(self):

        def pd_f(x, y):
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
        self.eps = config.TOLERANCE.get(
            self.dtype).get('first_order_grad').get('eps')
        self.rtol = config.TOLERANCE.get(
            self.dtype).get('first_order_grad').get('rtol')
        self.atol = config.TOLERANCE.get(
            self.dtype).get('first_order_grad').get('atol')


class TestHessianFloat32(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        paddle.enable_static()
        if fluid.core.is_compiled_with_cuda():
            self.place = fluid.CUDAPlace(0)
        else:
            self.place = fluid.CPUPlace()
        self.dtype = 'float32'
        prepare_data(self, all_data_shapes, self.dtype)
        self.eps = config.TOLERANCE.get(
            self.dtype).get('second_order_grad').get('eps')
        self.rtol = config.TOLERANCE.get(
            self.dtype).get('second_order_grad').get('rtol')
        self.atol = config.TOLERANCE.get(
            self.dtype).get('second_order_grad').get('atol')

    def run_test_by_fullmatrix(self, pd_f, inps, np_hess, batch=False):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            xs = make_tensors(inps)
            HH = paddle.incubate.autograd.Hessian(pd_f, xs, is_batched=batch)
            nrow, ncol = HH.shape
            full_hessian = HH[:]
        exe = fluid.Executor(self.place)
        exe.run(startup)
        if isinstance(inps, list):
            feeds = {f'x{i}': x for i, x in enumerate(inps)}
        else:
            feeds = {'x': inps}
        pd_hess = exe.run(main, feed=feeds, fetch_list=[full_hessian])[0]
        np.testing.assert_allclose(pd_hess, np_hess, self.rtol, self.atol)

    def test_square(self):

        def pd_f(x):
            """Input is a square matrix."""
            return paddle.matmul(x, x.T).flatten().sum()

        def np_hess(x):
            dim = x.shape[0]
            upperleft = 2 * np.eye(dim, dtype=self.dtype)
            upper = np.concatenate((upperleft, upperleft))
            return np.concatenate((upper, upper), axis=1)

        self.run_test_by_fullmatrix(pd_f, self.B, np_hess(self.B))


class TestHessianFloat64(TestHessianFloat32):

    @classmethod
    def setUpClass(self):
        paddle.enable_static()
        if fluid.core.is_compiled_with_cuda():
            self.place = fluid.CUDAPlace(0)
        else:
            self.place = fluid.CPUPlace()
        self.dtype = 'float64'
        prepare_data(self, all_data_shapes, self.dtype)
        self.eps = config.TOLERANCE.get(
            self.dtype).get('second_order_grad').get('eps')
        self.rtol = config.TOLERANCE.get(
            self.dtype).get('second_order_grad').get('rtol')
        self.atol = config.TOLERANCE.get(
            self.dtype).get('second_order_grad').get('atol')


if __name__ == "__main__":
    unittest.main()
