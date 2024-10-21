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

import unittest

import numpy as np
from op_test import OpTest

import paddle

paddle.enable_static()


class TestDeterminantOp(OpTest):
    def setUp(self):
        self.python_api = paddle.linalg.det
        self.init_data()
        self.op_type = "determinant"
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['Input'], ['Out'], check_pir=True)

    def init_data(self):
        np.random.seed(0)
        self.case = np.random.rand(3, 3, 3, 5, 5).astype('float64')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)


class TestDeterminantOpCase1(TestDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        self.case = np.random.rand(10, 10).astype('float32')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)


class TestDeterminantOpCase1FP16(TestDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        self.case = np.random.rand(10, 10).astype(np.float16)
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case.astype(np.float32))


class TestDeterminantOpCase2(TestDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        # not invertible matrix
        self.case = np.ones([4, 2, 4, 4]).astype('float64')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)


class TestDeterminantOpCase2FP16(TestDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        # not invertible matrix
        self.case = np.ones([4, 2, 4, 4]).astype(np.float16)
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case.astype(np.float32)).astype(
            np.float16
        )


class TestDeterminantOpCase3(TestDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        self.case = np.vectorize(complex)(
            np.random.rand(10, 10), np.random.rand(10, 10)
        ).astype('complex64')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)


class TestDeterminantOpCase4(TestDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        self.case = np.vectorize(complex)(
            np.random.rand(10, 10), np.random.rand(10, 10)
        ).astype('complex128')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)


class TestDeterminantOpCase5(TestDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        # not invertible matrix
        self.case = np.ones([4, 2, 4, 4]).astype('complex64')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)


class TestDeterminantOpCase6(TestDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        # not invertible matrix
        self.case = np.ones([4, 2, 4, 4]).astype('complex128')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)


class TestDeterminantOpCase7(TestDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        self.case = np.vectorize(complex)(
            np.random.rand(5, 3, 10, 10), np.random.rand(5, 3, 10, 10)
        ).astype('complex64')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)


class TestDeterminantOpCase8(TestDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        self.case = np.vectorize(complex)(
            np.random.rand(5, 3, 10, 10), np.random.rand(5, 3, 10, 10)
        ).astype('complex128')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)


class TestDeterminantAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.dtype = np.float32
        self.shape = [3, 3, 5, 5]
        self.x = np.random.random(self.shape).astype(self.dtype)
        self.place = paddle.CPUPlace()

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape, dtype=self.dtype)
            out_value = paddle.linalg.det(x)
            exe = paddle.static.Executor(self.place)
            (out_np,) = exe.run(feed={'X': self.x}, fetch_list=[out_value])
        out_ref = np.linalg.det(self.x)

        np.testing.assert_allclose(out_np, out_ref, rtol=0.001)
        self.assertEqual(out_np.shape, out_ref.shape)
        self.assertEqual(tuple(out_value.shape), out_ref.shape)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        out = paddle.linalg.det(x_tensor)
        out_ref = np.linalg.det(self.x)
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
        paddle.enable_static()


def determinant_complex_numeric_grad_single_batch(
    x, n, delta=0.005, det_out_grad=np.array(1 + 0j)
):
    # an naive implementation of numeric_grad with single batch input x
    # the output of det for complex matrix is always complex, so det_out_grad
    # should be a+bj, where a and b are arbitrary real numbers
    dx = []
    for i in range(n):
        for j in range(n):
            xp = x.copy()
            xn = x.copy()
            xpj = x.copy()
            xnj = x.copy()
            xp[i, j] += delta
            xn[i, j] -= delta
            xpj[i, j] += delta * 1j
            xnj[i, j] -= delta * 1j
            yp = np.linalg.det(xp)
            yn = np.linalg.det(xn)
            ypj = np.linalg.det(xpj)
            ynj = np.linalg.det(xnj)
            df_over_dr = (yp - yn) / delta / 2
            df_over_di = (ypj - ynj) / delta / 2
            dl_over_du, dl_over_dv = det_out_grad.real, det_out_grad.imag
            du_over_dr, dv_over_dr = df_over_dr.real, df_over_dr.imag
            du_over_di, dv_over_di = df_over_di.real, df_over_di.imag
            dl_over_dr = np.sum(
                dl_over_du * du_over_dr + dl_over_dv * dv_over_dr
            )
            dl_over_di = np.sum(
                dl_over_du * du_over_di + dl_over_dv * dv_over_di
            )
            dx.append(dl_over_dr + 1j * dl_over_di)
    return np.array(dx).reshape([n, n])


class TestDeterminantAPIComplex(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.dtype = np.complex64
        self.shape = [2, 1, 4, 3, 6, 6]
        self.x = np.vectorize(complex)(
            np.random.random(self.shape), np.random.random(self.shape)
        ).astype(self.dtype)
        self.place = paddle.CPUPlace()
        self.out_grad = (
            np.array([1 - 0.5j] * 2 * 1 * 4 * 3)
            .reshape(2, 1, 4, 3)
            .astype(self.dtype)
        )
        self.x_grad_ref_dy = self.get_numeric_grad(
            self.x, self.shape, self.out_grad
        )
        self.x_grad_ref_st = self.get_numeric_grad(self.x, self.shape)

    def get_numeric_grad(self, x, shape, out_grad=None):
        n = shape[-1]
        flatten_x = x.reshape([-1, n, n])
        n_batch = flatten_x.shape[0]
        grad = []
        if out_grad is None:
            for b in range(n_batch):
                grad.append(
                    determinant_complex_numeric_grad_single_batch(
                        flatten_x[b], n
                    )
                )
        else:
            flatten_grad = out_grad.reshape([-1])
            for b in range(n_batch):
                grad.append(
                    determinant_complex_numeric_grad_single_batch(
                        flatten_x[b], n, det_out_grad=flatten_grad[b]
                    )
                )
        return np.array(grad).reshape(shape)

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape, dtype=self.dtype)
            x.stop_gradient = False
            out_value = paddle.linalg.det(x)
            x_grad = paddle.static.gradients([out_value], x)
            exe = paddle.static.Executor(self.place)
            (out_np, x_grad_np) = exe.run(
                feed={'X': self.x}, fetch_list=[out_value, x_grad]
            )
        out_ref = np.linalg.det(self.x)

        np.testing.assert_allclose(out_np, out_ref, rtol=0.001)
        self.assertEqual(out_np.shape, out_ref.shape)
        self.assertEqual(tuple(out_value.shape), out_ref.shape)
        np.testing.assert_allclose(x_grad_np, self.x_grad_ref_st, rtol=0.001)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        x_tensor.stop_gradient = False
        out = paddle.linalg.det(x_tensor)
        out.backward(paddle.to_tensor(self.out_grad))
        out_ref = np.linalg.det(self.x)
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
        np.testing.assert_allclose(
            x_tensor.grad.numpy(), self.x_grad_ref_dy, rtol=0.001
        )
        paddle.enable_static()


class TestDeterminantAPIComplex2(TestDeterminantAPIComplex):
    def setUp(self):
        np.random.seed(0)
        self.dtype = np.complex128
        self.shape = [3, 3, 5, 5]
        self.x = np.vectorize(complex)(
            np.random.random(self.shape), np.random.random(self.shape)
        ).astype(self.dtype)
        self.place = paddle.CPUPlace()
        self.out_grad = (
            np.array([0.5 + 1.2j] * 3 * 3).reshape(3, 3).astype(self.dtype)
        )
        self.x_grad_ref_dy = self.get_numeric_grad(
            self.x, self.shape, self.out_grad
        )
        self.x_grad_ref_st = self.get_numeric_grad(self.x, self.shape)


class TestSlogDeterminantOp(OpTest):
    def setUp(self):
        self.op_type = "slogdeterminant"
        self.python_api = paddle.linalg.slogdet
        self.init_data()
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        # the slog det's grad value is always huge
        self.check_grad(
            ['Input'], ['Out'], max_relative_error=0.1, check_pir=True
        )

    def init_data(self):
        np.random.seed(0)
        self.case = np.random.rand(4, 5, 5).astype('float64')
        self.inputs = {'Input': self.case}
        self.target = np.array(np.linalg.slogdet(self.case))


class TestSlogDeterminantOpCase1(TestSlogDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        self.case = np.random.rand(2, 2, 5, 5).astype(np.float32)
        self.inputs = {'Input': self.case}
        self.target = np.array(np.linalg.slogdet(self.case))


class TestSlogDeterminantAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [3, 3, 5, 5]
        self.x = np.random.random(self.shape).astype(np.float32)
        self.place = paddle.CPUPlace()

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            out = paddle.linalg.slogdet(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        out_ref = np.array(np.linalg.slogdet(self.x))
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=0.001)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        out = paddle.linalg.slogdet(x_tensor)
        out_ref = np.array(np.linalg.slogdet(self.x))
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
        paddle.enable_static()


def slogdeterminant_complex_numeric_grad_single_batch(
    x, n, delta=0.005, logabsdet_out_grad=np.array(1 + 0j)
):
    # an naive implementation of numeric_grad with single batch input x
    # the output of logabsdet is always real, so logabsdet_out_grad
    # should be a+0j, where a is an arbitrary real number
    dx = []
    for i in range(n):
        for j in range(n):
            xp = x.copy()
            xn = x.copy()
            xpj = x.copy()
            xnj = x.copy()
            xp[i, j] += delta
            xn[i, j] -= delta
            xpj[i, j] += delta * 1j
            xnj[i, j] -= delta * 1j
            _, yp = np.linalg.slogdet(xp)
            _, yn = np.linalg.slogdet(xn)
            _, ypj = np.linalg.slogdet(xpj)
            _, ynj = np.linalg.slogdet(xnj)
            df_over_dr = (yp - yn) / delta / 2
            df_over_di = (ypj - ynj) / delta / 2
            dl_over_du, dl_over_dv = (
                logabsdet_out_grad.real,
                logabsdet_out_grad.imag,
            )
            du_over_dr, dv_over_dr = df_over_dr.real, df_over_dr.imag
            du_over_di, dv_over_di = df_over_di.real, df_over_di.imag
            dl_over_dr = np.sum(
                dl_over_du * du_over_dr + dl_over_dv * dv_over_dr
            )
            dl_over_di = np.sum(
                dl_over_du * du_over_di + dl_over_dv * dv_over_di
            )
            dx.append(dl_over_dr + 1j * dl_over_di)
    return np.array(dx).reshape([n, n])


class TestSlogDeterminantAPIComplex(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [3, 3, 5, 5]
        self.dtype = np.complex64
        self.x = np.vectorize(complex)(
            np.random.random(self.shape), np.random.random(self.shape)
        ).astype(self.dtype)
        self.places = [paddle.CPUPlace()]
        if paddle.base.core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self.out_grad = (
            np.array([1 + 0j, 1 + 0j] * 3 * 3)
            .reshape(2, 3, 3)
            .astype(self.dtype)
        )
        self.x_grad_ref_dy = self.get_numeric_grad(
            self.x, self.shape, self.out_grad
        )
        self.x_grad_ref_st = self.get_numeric_grad(self.x, self.shape)

    def get_numeric_grad(self, x, shape, out_grad=None):
        n = shape[-1]
        flatten_x = x.reshape([-1, n, n])
        n_batch = flatten_x.shape[0]
        grad = []
        if out_grad is None:
            for b in range(n_batch):
                grad.append(
                    slogdeterminant_complex_numeric_grad_single_batch(
                        flatten_x[b], n
                    )
                )
        else:
            flatten_grad = out_grad.reshape([-1, 2])
            for b in range(n_batch):
                grad.append(
                    slogdeterminant_complex_numeric_grad_single_batch(
                        flatten_x[b], n, logabsdet_out_grad=flatten_grad[b][1]
                    )
                )
        return np.array(grad).reshape(shape)

    def test_api_static(self):
        for place in self.places:
            paddle.enable_static()
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.shape, self.dtype)
                x.stop_gradient = False
                out = paddle.linalg.slogdet(x)
                x_grad = paddle.static.gradients(out, x)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={'X': self.x}, fetch_list=[out, x_grad])
            out_ref = np.array(np.linalg.slogdet(self.x))
            np.testing.assert_allclose(res[0], out_ref, rtol=0.001)
            np.testing.assert_allclose(res[1], self.x_grad_ref_st, rtol=0.001)

    def test_api_dygraph(self):
        for place in self.places:
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x)
            x_tensor.stop_gradient = False
            out = paddle.linalg.slogdet(x_tensor)
            out.backward(paddle.to_tensor(self.out_grad))
            out_ref = np.array(np.linalg.slogdet(self.x))
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
            np.testing.assert_allclose(
                x_tensor.grad.numpy(), self.x_grad_ref_dy, rtol=0.001
            )
            paddle.enable_static()


class TestSlogDeterminantAPIComplex2(TestSlogDeterminantAPIComplex):
    def setUp(self):
        np.random.seed(0)
        self.shape = [6, 5, 5]
        self.dtype = np.complex128
        self.x = np.vectorize(complex)(
            np.random.random(self.shape), np.random.random(self.shape)
        ).astype(self.dtype)
        self.places = [paddle.CPUPlace()]
        if paddle.base.core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self.out_grad = np.array([3 + 0j, 3 + 0j] * 6).reshape(2, 6)
        self.x_grad_ref_dy = self.get_numeric_grad(
            self.x, self.shape, self.out_grad
        )
        self.x_grad_ref_st = self.get_numeric_grad(self.x, self.shape)


if __name__ == '__main__':
    unittest.main()
