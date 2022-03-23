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

import collections
import typing
import unittest

import numpy as np
import paddle
import paddle.compat as cpt
import paddle.nn.functional as F
from paddle.autograd.functional import _as_tensors

import config
import parameterize as param
import utils
from utils import (_compute_numerical_batch_hessian, _compute_numerical_hessian,
                   _compute_numerical_vhp, matmul, mul, nested, o2, pow, reduce,
                   reduce_dim, unuse)


def make_v(f, inputs):
    outputs = _as_tensors(f(*inputs))
    return [paddle.ones_like(x) for x in outputs]


class TestAutogradFunctional(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.RAW_INPUTS = {
            'a': [1.0],
            'b': [1.0, 2.0],
            'c': [3.0, 4.0],
            'd': [[2.0], [3.0]],
            'A': [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            'B': [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
        }

    def setUp(self):
        pass

    def gen_input(self, inp, stop_gradient=False):
        if isinstance(inp, paddle.Tensor):
            return inp
        return paddle.to_tensor(
            self.RAW_INPUTS[inp], stop_gradient=stop_gradient)

    def gen_inputs(self, inputs):
        if isinstance(inputs, list):
            inputs = [self.gen_input(x) for x in inputs]
        else:
            inputs = [self.gen_input(inputs)]
        return inputs

    def gen_test_pairs(self,
                       func,
                       inputs,
                       v=None,
                       create_graph=False,
                       allow_unused=False):
        def vjp_test():
            nonlocal v
            xs = self.gen_inputs(inputs)
            if v is not None:
                v = self.gen_inputs(v)
                outputs, inputs_grad = paddle.autograd.vjp(func, xs, v)
            else:
                outputs, inputs_grad = paddle.autograd.vjp(func, xs)
            return outputs, inputs_grad

        def grad_test():
            nonlocal v
            xs = self.gen_inputs(inputs)
            if v is not None:
                v = self.gen_inputs(v)
            outputs = func(*xs)
            if v is not None:
                inputs_grad = paddle.grad(
                    outputs,
                    xs,
                    v,
                    create_graph=create_graph,
                    allow_unused=allow_unused)
            else:
                inputs_grad = paddle.grad(
                    outputs,
                    xs,
                    create_graph=create_graph,
                    allow_unused=allow_unused)
            return outputs, inputs_grad

        return vjp_test, grad_test

    def gen_jvp_tests(self,
                      func,
                      inputs,
                      v=None,
                      create_graph=False,
                      allow_unused=False):
        def jvp_test():
            nonlocal v
            xs = self.gen_inputs(inputs)
            if v is not None:
                v = self.gen_inputs(v)
                outputs, outputs_grad = paddle.autograd.jvp(
                    func,
                    xs,
                    v,
                    create_graph=create_graph,
                    allow_unused=allow_unused)
            else:
                outputs, outputs_grad = paddle.autograd.jvp(
                    func,
                    xs,
                    create_graph=create_graph,
                    allow_unused=allow_unused)
            return outputs, outputs_grad

        return jvp_test

    def check_results(self, ref, res):
        type_error = 'Result is different than expected in shape or type'
        value_error = 'Result is different than expected values'
        if ref is None:
            self.assertTrue(res is None, type_error)
        elif isinstance(ref, paddle.Tensor):
            self.assertTrue(isinstance(res, paddle.Tensor), type_error)
            np.testing.assert_allclose(res, ref)
        else:
            self.assertTrue(len(res) == len(ref), type_error)
            for i in range(len(ref)):
                self.check_results(ref[i], res[i])
        return True


class TestVJP(TestAutogradFunctional):
    def test_vjp_i1o1(self):
        test_cases = [
            [reduce, 'A'],  # noqa
            [reduce_dim, 'A'],  # noqa
        ]  # noqa
        for f, inputs in test_cases:
            vjp, grad = self.gen_test_pairs(f, inputs)
            vjp_result, grad_result = vjp(), grad()
            self.check_results(grad_result, vjp_result)

    def test_vjp_i2o1(self):
        test_cases = [
            [matmul, ['A', 'B']],  # noqa
            [mul, ['b', 'c']],  # noqa
        ]  # noqa
        for f, inputs in test_cases:
            vjp, grad = self.gen_test_pairs(f, inputs)
            vjp_result, grad_result = vjp(), grad()
            self.check_results(grad_result, vjp_result)

    def test_vjp_i2o2(self):
        test_cases = [
            [o2, ['A', 'A']],  # noqa
        ]  # noqa
        for f, inputs in test_cases:
            inputs = self.gen_inputs(inputs)
            v = make_v(f, inputs)
            vjp, grad = self.gen_test_pairs(f, inputs, v=v)
            vjp_result, grad_result = vjp(), grad()
            self.check_results(grad_result, vjp_result)

    def test_vjp_i2o2_omitting_v(self):
        test_cases = [
            [o2, ['A', 'A']],  # noqa
        ]  # noqa
        for f, inputs in test_cases:
            inputs = self.gen_inputs(inputs)
            vjp, grad = self.gen_test_pairs(f, inputs)
            vjp_result, grad_result = vjp(), grad()
            self.check_results(grad_result, vjp_result)

    def test_vjp_nested(self):
        x = self.gen_input('a')
        test_cases = [
            [nested(x), 'a'],  # noqa
        ]
        for f, inputs in test_cases:
            vjp, grad = self.gen_test_pairs(f, inputs)
            vjp_result, grad_result = vjp(), grad()
            self.check_results(grad_result, vjp_result)

    def test_vjp_aliased_input(self):
        x = self.gen_input('a')
        ref = self.gen_test_pairs(nested(x), 'a')[0]
        aliased = self.gen_test_pairs(nested(x), x)[0]
        ref_result, aliased_result = ref(), aliased()
        self.check_results(ref_result, aliased_result)


def jac(grad_fn, f, inputs):
    assert grad_fn in [paddle.autograd.vjp, paddle.autograd.jvp]
    if grad_fn is paddle.autograd.jvp:
        vs = [paddle.zeros_like(x) for x in inputs]
    else:
        outputs = f(*inputs)
        if isinstance(outputs, paddle.Tensor):
            outputs = [outputs]
        vs = [paddle.zeros_like(y) for y in outputs]
    JJ_cols = []
    for i, v in enumerate(vs):
        v = v.flatten()
        for j in range(len(v)):
            _v = paddle.zeros_like(v).detach()
            _v[j] = 1.0
            _v = _v.reshape(vs[i].shape)
            _vs = vs.copy()
            _vs[i] = _v
            _, grads = grad_fn(f, inputs, _vs)
            d_outs = paddle.concat([d_out.flatten() for d_out in grads])
            JJ_cols.append(d_outs)
    # JJ is the fully unrolled jacobian
    JJ = paddle.stack(JJ_cols)
    if grad_fn is paddle.autograd.vjp:
        JJ = JJ.t()
    return JJ


class TestJVP(TestAutogradFunctional):
    def test_jvp_i1o1(self):
        test_cases = [
            [reduce, 'A'],  # noqa
            [reduce_dim, 'A'],  # noqa
        ]  # noqa
        for f, inputs in test_cases:
            inputs = self.gen_inputs(inputs)
            forward_jac = jac(paddle.autograd.jvp, f, inputs)
            reverse_jac = jac(paddle.autograd.vjp, f, inputs)
            self.check_results(forward_jac, reverse_jac)

    def test_jvp_i2o1(self):
        test_cases = [  # noqa
            [matmul, ['A', 'B']],  # noqa
        ]  # noqa
        for f, inputs in test_cases:
            inputs = self.gen_inputs(inputs)
            forward_jac = jac(paddle.autograd.jvp, f, inputs)
            reverse_jac = jac(paddle.autograd.vjp, f, inputs)
            self.check_results(forward_jac, reverse_jac)

    def test_jvp_i2o2(self):
        test_cases = [  # noqa
            [o2, ['A', 'A']],  # noqa
        ]  # noqa
        for f, inputs in test_cases:
            inputs = self.gen_inputs(inputs)
            forward_jac = jac(paddle.autograd.jvp, f, inputs)
            reverse_jac = jac(paddle.autograd.vjp, f, inputs)
            self.check_results(forward_jac, reverse_jac)

    def test_jvp_i2o2_omitting_v(self):
        test_cases = [  # noqa
            [o2, ['A', 'A']],  # noqa
        ]  # noqa
        for f, inputs in test_cases:
            inputs = self.gen_inputs(inputs)
            results_omitting_v = paddle.autograd.jvp(f, inputs)
            v = [paddle.ones_like(x) for x in inputs]
            results_with_v = paddle.autograd.jvp(f, inputs, v)
            self.check_results(results_omitting_v, results_with_v)


@param.place(config.DEVICES)
@param.parameterize(
    (param.TEST_CASE_NAME, 'func', 'xs'),
    (  # noqa
        ('1d-in-1d-out', utils.square, np.array([2., 3.])),
        ('3d-in-3d-out', utils.square, np.random.rand(2, 3, 4)),
        ('multi-input', utils.square, np.random.rand(10, 20)),
        ('matmul', paddle.matmul,
         (np.random.rand(2, 2), np.random.rand(2, 2))), ))
class TestJacobianClassNoBatch(unittest.TestCase):
    def setUp(self):
        self._dtype = self.xs[0].dtype if isinstance(
            self.xs, typing.Sequence) else self.xs.dtype
        self._eps = config.EPS.get(str(self._dtype))

        self.xs = [paddle.to_tensor(x) for x in self.xs] if isinstance(
            self.xs, typing.Sequence) else paddle.to_tensor(self.xs)
        self._actual = paddle.autograd.Jacobian(self.func, self.xs, False)
        self._expected = self._expected()

    def test_jacobian(self):
        Index = collections.namedtuple('Index', ('type', 'value'))
        indexes = (Index('all', (slice(0, None, None), slice(0, None, None))),
                   Index('row', (0, slice(0, None, None))),
                   Index('col', (slice(0, None, None), 0)),
                   Index('multi-row', (slice(0, 2, 1), slice(0, None, None))))
        self.assertEqual(self._actual[:].dtype, self._expected.dtype)
        for index in indexes:
            np.testing.assert_allclose(
                self._actual.__getitem__(index.value),
                self._expected.__getitem__(index.value),
                rtol=config.RTOL.get(str(self._dtype)),
                atol=config.ATOL.get(str(self._dtype)),
                err_msg=f'Testcase {index.type} index not passed, value is {index.value}'
            )

    def _expected(self):
        # numerical_jacobian return list of list of tensors, need to concat.
        results = utils._compute_numerical_jacobian(self.func, self.xs,
                                                    self._eps, self._dtype)
        rows = []
        for i in range(len(results)):
            rows.append(
                paddle.concat([paddle.to_tensor(x) for x in results[i]], -1))
        return paddle.concat(rows, 0)


@param.place(config.DEVICES)
@param.parameterize((param.TEST_CASE_NAME, 'func', 'xs'), (
    ('1d-in-1d-out', utils.square, np.array([[1., 2., 3.], [3., 4., 3.]])),
    ('3d-in-3d-out', utils.square, np.random.rand(2, 3, 4)),
    ('multi-in-single-out', utils.square, np.random.rand(5, 6)), ))
class TestJacobianClassBatchFirst(unittest.TestCase):
    def setUp(self):
        self._dtype = self.xs[0].dtype if isinstance(
            self.xs, typing.Sequence) else self.xs.dtype
        self._eps = config.EPS.get(str(self._dtype))

        self.xs = [paddle.to_tensor(x) for x in self.xs] if isinstance(
            self.xs, typing.Sequence) else paddle.to_tensor(self.xs)
        self._actual = paddle.autograd.Jacobian(self.func, self.xs, True)
        self._expected = self._expected()

    def test_jacobian(self):
        Index = collections.namedtuple('Index', ('type', 'value'))
        indexes = (
            Index('all', (slice(0, None, None), slice(0, None, None),
                          slice(0, None, None))),
            Index('row', (slice(0, None, None), 0, slice(0, None, None))),
            Index('col',
                  (slice(0, None, None), slice(0, None, None), 0)), Index(
                      'batch', (slice(0, 2, None), slice(0, None, None),
                                slice(0, None, None))),
            Index('multi-row',
                  (slice(0, 1, None), slice(0, 2, 1), slice(0, None, None))))
        self.assertEqual(self._actual[:].dtype, self._expected.dtype)
        for index in indexes:
            np.testing.assert_allclose(
                self._actual.__getitem__(index.value),
                self._expected.__getitem__(index.value),
                rtol=config.RTOL.get(str(self._dtype)),
                atol=config.ATOL.get(str(self._dtype)),
                err_msg=f'Testcase {index.type} index not passed, value is {index.value}'
            )

    def _expected(self):
        results = utils._compute_numerical_batch_jacobian(
            self.func, self.xs, self._eps, self._dtype)
        rows = []
        for i in range(len(results)):
            rows.append(
                paddle.concat([paddle.to_tensor(x) for x in results[i]], -1))
        return paddle.transpose(paddle.concat(rows, 1), (1, 0, 2))


class TestHessianClassNoBatch(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.shape = (2, 2)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = 1e-2
        self.rtol = 1e-2
        self.atol = 1e-2
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)

    def test_single_input(self):
        def func(x):
            return paddle.sum(paddle.matmul(x, x))

        numerical_hessian = utils._compute_numerical_hessian(
            func, self.x, self.numerical_delta, self.np_dtype)
        numerical_hessian = utils.concat_to_matrix(numerical_hessian)

        self.x.stop_gradient = False
        hessian = paddle.autograd.Hessian(func, self.x)
        assert np.allclose(hessian[:].numpy(), numerical_hessian, self.rtol,
                           self.atol)

    def test_multi_input(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, y))

        numerical_hessian = utils._compute_numerical_hessian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        numerical_hessian = utils.concat_to_matrix(numerical_hessian)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        hessian = paddle.autograd.Hessian(func, [self.x, self.y])
        np.testing.assert_allclose(
            hessian[:].numpy(),
            numerical_hessian,
            rtol=self.rtol,
            atol=self.atol)

    def test_allow_unused_true(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, x))

        numerical_hessian = utils._compute_numerical_hessian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        numerical_hessian = utils.concat_to_matrix(numerical_hessian)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        hessian = paddle.autograd.Hessian(func, [self.x, self.y])
        np.testing.assert_allclose(hessian[:].numpy(), numerical_hessian,
                                   self.rtol, self.atol)

    def test_create_graph_true(self):
        def func(x):
            return paddle.sum(F.sigmoid(x))

        numerical_hessian = utils._compute_numerical_hessian(
            func, self.x, self.numerical_delta, self.np_dtype)
        numerical_hessian = utils.concat_to_matrix(numerical_hessian)
        self.x.stop_gradient = False
        hessian = paddle.autograd.Hessian(func, self.x)
        assert hessian[:].stop_gradient == False
        np.testing.assert_allclose(hessian[:].numpy(), numerical_hessian,
                                   self.rtol, self.atol)


class TestHessian(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.shape = (2, 2)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = 1e-2
        self.rtol = 1e-2
        self.atol = 1e-2
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)

    def test_single_input(self):
        def func(x):
            return paddle.sum(paddle.matmul(x, x))

        numerical_hessian = _compute_numerical_hessian(
            func, self.x, self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        hessian = paddle.autograd.hessian(func, self.x)
        assert np.allclose(hessian.numpy(), numerical_hessian[0][0], self.rtol,
                           self.atol)

    def test_multi_input(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, y))

        numerical_hessian = _compute_numerical_hessian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        self.y.stop_gradient = False
        hessian = paddle.autograd.hessian(func, [self.x, self.y])
        for i in range(len(hessian)):
            for j in range(len(hessian[0])):
                assert np.allclose(hessian[i][j].numpy(),
                                   numerical_hessian[i][j], self.rtol,
                                   self.atol)

    def test_allow_unused_false(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, x))

        try:
            self.x.stop_gradient = False
            self.y.stop_gradient = False
            hessian = paddle.autograd.hessian(func, [self.x, self.y])
        except ValueError as e:
            error_msg = cpt.get_exception_message(e)
            assert error_msg.find("allow_unused") > 0

    def test_allow_unused_true(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, x))

        numerical_hessian = _compute_numerical_hessian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        hessian = paddle.autograd.hessian(
            func, [self.x, self.y], allow_unused=True)
        for i in range(len(hessian)):
            for j in range(len(hessian[0])):
                if i == j == 0:
                    assert np.allclose(hessian[i][j].numpy(),
                                       numerical_hessian[i][j], self.rtol,
                                       self.atol)
                else:
                    assert hessian[i][j] is None

    def test_create_graph_false(self):
        def func(x):
            return paddle.sum(paddle.matmul(x, x))

        numerical_hessian = _compute_numerical_hessian(
            func, self.x, self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        hessian = paddle.autograd.hessian(func, self.x)
        assert hessian.stop_gradient == True
        assert np.allclose(hessian.numpy(), numerical_hessian[0][0], self.rtol,
                           self.atol)
        try:
            paddle.grad(hessian, self.x)
        except RuntimeError as e:
            error_msg = cpt.get_exception_message(e)
            assert error_msg.find("has no gradient") > 0

    def test_create_graph_true(self):
        def func(x):
            return paddle.sum(F.sigmoid(x))

        numerical_hessian = _compute_numerical_hessian(
            func, self.x, self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        hessian = paddle.autograd.hessian(func, self.x, create_graph=True)
        assert hessian.stop_gradient == False
        assert np.allclose(hessian.numpy(), numerical_hessian[0][0], self.rtol,
                           self.atol)
        triple_grad = paddle.grad(hessian, self.x)
        assert triple_grad is not None


class TestHessianFloat64(TestHessian):
    @classmethod
    def setUpClass(self):
        self.shape = (2, 2)
        self.dtype = 'float64'
        self.np_dtype = np.float64
        self.numerical_delta = 1e-5
        self.rtol = 1e-5
        self.atol = 1e-5
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)


class TestBatchHessian(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.x_shape = (5, 2)
        self.weight_shape = (2, 4)
        self.y_shape = (5, 2)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = 1e-2
        self.rtol = 1e-3
        self.atol = 1e-3
        self.x = paddle.rand(shape=self.x_shape, dtype=self.dtype)
        self.weight = paddle.rand(shape=self.weight_shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.y_shape, dtype=self.dtype)

    def test_single_input(self):
        def func(x):
            return paddle.matmul(x * x, self.weight)[:, 0:1]

        numerical_hessian = _compute_numerical_batch_hessian(
            func, self.x, self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        hessian = paddle.autograd.batch_hessian(func, self.x, create_graph=True)
        assert np.allclose(hessian, numerical_hessian, self.rtol, self.atol)

    def test_multi_input(self):
        def func(x, y):
            return paddle.matmul(x * x * y * y, self.weight)[:, 0:1]

        numerical_hessian = _compute_numerical_batch_hessian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        self.y.stop_gradient = False
        hessian = paddle.autograd.batch_hessian(func, [self.x, self.y])

        shape_tensor = paddle.to_tensor(numerical_hessian).astype("float64")
        hessian_reshape = np.reshape(hessian, (shape_tensor.shape))
        assert np.allclose(hessian_reshape, numerical_hessian, self.rtol,
                           self.atol)

    def test_allow_unused_false(self):
        def func(x, y):
            return paddle.matmul(x * x, self.weight)[:, 0:1]

        try:
            self.x.stop_gradient = False
            self.y.stop_gradient = False
            hessian = paddle.autograd.batch_hessian(func, [self.x, self.y])
        except ValueError as e:
            error_msg = cpt.get_exception_message(e)
            assert error_msg.find("allow_unused") > 0

    def test_allow_unused_true(self):
        def func(x, y):
            return paddle.matmul(x * x, self.weight)[:, 0:1]

        numerical_hessian = _compute_numerical_batch_hessian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        hessian = paddle.autograd.batch_hessian(
            func, [self.x, self.y], allow_unused=True)

        for i in range(len(hessian)):
            for j in range(len(hessian[0])):
                if i == j == 0:
                    numerical_hessian = np.stack(
                        (numerical_hessian[i][j], numerical_hessian[i][j + 1]),
                        axis=0)
                    assert np.allclose(hessian[i][j], numerical_hessian,
                                       self.rtol, self.atol)
                else:
                    assert hessian[i][j] is None

    def test_create_graph_false(self):
        def func(x):
            return paddle.matmul(x * x, self.weight)[:, 0:1]

        numerical_hessian = _compute_numerical_batch_hessian(
            func, self.x, self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        hessian = paddle.autograd.batch_hessian(func, self.x)
        assert hessian.stop_gradient == True
        assert np.allclose(hessian.numpy(), numerical_hessian, self.rtol,
                           self.atol)
        try:
            paddle.grad(hessian, self.x)
        except RuntimeError as e:
            error_msg = cpt.get_exception_message(e)
            assert error_msg.find("has no gradient") > 0

    def test_create_graph_true(self):
        def func(x):
            return paddle.matmul(x * x, self.weight)[:, 0:1]

        numerical_hessian = _compute_numerical_batch_hessian(
            func, self.x, self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        hessian = paddle.autograd.batch_hessian(func, self.x, create_graph=True)
        assert hessian.stop_gradient == False
        assert np.allclose(hessian.numpy(), numerical_hessian, self.rtol,
                           self.atol)
        triple_grad = paddle.grad(hessian, self.x)
        assert triple_grad is not None


class TestBatchHessianFloat64(TestBatchHessian):
    @classmethod
    def setUpClass(self):
        self.x_shape = (5, 2)
        self.weight_shape = (2, 4)
        self.y_shape = (5, 2)
        self.dtype = 'float64'
        self.np_dtype = np.float64
        self.numerical_delta = 1e-4
        self.rtol = 1e-5
        self.atol = 1e-5
        self.x = paddle.rand(shape=self.x_shape, dtype=self.dtype)
        self.weight = paddle.rand(shape=self.weight_shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.y_shape, dtype=self.dtype)


class TestVHP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.shape = (2, 2)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = 1e-2
        self.rtol = 1e-2
        self.atol = 1e-2
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.vx = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.vy = paddle.rand(shape=self.shape, dtype=self.dtype)

    def test_single_input(self):
        def func(x):
            return paddle.sum(paddle.matmul(x, x))

        numerical_func_output = func(self.x).numpy()
        numerical_vhp = _compute_numerical_vhp(
            func, self.x, self.vx, self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        func_output, vhp = paddle.autograd.vhp(func, self.x, self.vx)
        assert np.allclose(func_output.numpy(), numerical_func_output,
                           self.rtol, self.atol)
        assert np.allclose(vhp[0].numpy(), numerical_vhp[0], self.rtol,
                           self.atol)

    def test_multi_input(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, y))

        numerical_func_output = func(self.x, self.y).numpy()
        numerical_vhp = _compute_numerical_vhp(
            func, [self.x, self.y], [self.vx, self.vy], self.numerical_delta,
            self.np_dtype)

        self.x.stop_gradient = False
        self.y.stop_gradient = False
        func_output, vhp = paddle.autograd.vhp(func, [self.x, self.y],
                                               [self.vx, self.vy])
        assert np.allclose(func_output.numpy(), numerical_func_output,
                           self.rtol, self.atol)
        for i in range(len(vhp)):
            assert np.allclose(vhp[i].numpy(), numerical_vhp[i], self.rtol,
                               self.atol)

    def test_v_default(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, y))

        numerical_func_output = func(self.x, self.y).numpy()
        vx = paddle.ones(self.vx.shape, dtype=self.vx.dtype)
        vy = paddle.ones(self.vy.shape, dtype=self.vy.dtype)
        numerical_vhp = _compute_numerical_vhp(func, [self.x, self.y],
                                               [vx, vy], self.numerical_delta,
                                               self.np_dtype)

        self.x.stop_gradient = False
        self.y.stop_gradient = False
        func_output, vhp = paddle.autograd.vhp(func, [self.x, self.y])
        assert np.allclose(func_output.numpy(), numerical_func_output,
                           self.rtol, self.atol)
        for i in range(len(vhp)):
            assert np.allclose(vhp[i].numpy(), numerical_vhp[i], self.rtol,
                               self.atol)

    def test_allow_unused_true(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, x))

        numerical_func_output = func(self.x, self.y).numpy()
        numerical_vhp = _compute_numerical_vhp(
            func, [self.x, self.y], [self.vx, self.vy], self.numerical_delta,
            self.np_dtype)

        self.x.stop_gradient = False
        self.y.stop_gradient = False
        func_output, vhp = paddle.autograd.vhp(func, [self.x, self.y],
                                               [self.vx, self.vy])
        assert np.allclose(func_output.numpy(), numerical_func_output,
                           self.rtol, self.atol)
        assert np.allclose(vhp[0].numpy(), numerical_vhp[0], self.rtol,
                           self.atol)

    def test_create_graph_true(self):
        def func(x):
            return paddle.sum(F.sigmoid(x))

        numerical_func_output = func(self.x).numpy()
        numerical_vhp = _compute_numerical_vhp(
            func, self.x, self.vx, self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        func_output, vhp = paddle.autograd.vhp(func, self.x, self.vx)
        assert np.allclose(func_output.numpy(), numerical_func_output,
                           self.rtol, self.atol)
        assert vhp[0].stop_gradient == False
        assert np.allclose(vhp[0].numpy(), numerical_vhp[0], self.rtol,
                           self.atol)
        triple_grad = paddle.grad(vhp, self.x)
        assert triple_grad is not None


class TestVHPFloat64(TestVHP):
    @classmethod
    def setUpClass(self):
        self.shape = (2, 2)
        self.dtype = 'float64'
        self.np_dtype = np.float64
        self.numerical_delta = 1e-5
        self.rtol = 1e-5
        self.atol = 1e-5
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.vx = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.vy = paddle.rand(shape=self.shape, dtype=self.dtype)


if __name__ == "__main__":
    unittest.main()
