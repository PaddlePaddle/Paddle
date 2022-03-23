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
from paddle.autograd.functional import _as_tensors, jvp, vjp

import config
import parameterize as param
import utils
from utils import matmul, mul, nested, o2, pow, reduce, reduce_dim, unuse


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
                outputs, inputs_grad = vjp(func, xs, v)
            else:
                outputs, inputs_grad = vjp(func, xs)
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
                outputs, outputs_grad = jvp(func,
                                            xs,
                                            v,
                                            create_graph=create_graph,
                                            allow_unused=allow_unused)
            else:
                outputs, outputs_grad = jvp(func,
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
    assert grad_fn in [vjp, jvp]
    if grad_fn is jvp:
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
            _, grads = grad_fn(f, inputs, vs)
            d_outs = paddle.concat([d_out.flatten() for d_out in grads])
            JJ_cols.append(d_outs)
    # JJ is the fully unrolled jacobian
    JJ = paddle.stack(JJ_cols)
    if grad_fn is vjp:
        JJ = JJ.t()
    return JJ


class TestJVP(TestAutogradFunctional):
    # def test_jvp_i1o1(self):
    #     test_cases = [
    #         [reduce, 'A'],  #noqa
    #         [reduce_dim, 'A'],  #noqa
    #     ]  #noqa
    #     for f, inputs in test_cases:
    #         inputs = self.gen_inputs(inputs)
    #         forward_jac = jac(jvp, f, inputs)
    #         reverse_jac = jac(vjp, f, inputs)
    #         self.check_results(forward_jac, reverse_jac)

    # def test_jvp_i2o1(self):
    #     test_cases = [  #noqa
    #         [matmul, ['A', 'B']],  #noqa
    #     ]  #noqa
    #     for f, inputs in test_cases:
    #         inputs = self.gen_inputs(inputs)
    #         forward_jac = jac(jvp, f, inputs)
    #         reverse_jac = jac(vjp, f, inputs)
    #         self.check_results(forward_jac, reverse_jac)

    def test_jvp_i2o2(self):
        test_cases = [  # noqa
            [o2, ['A', 'A']],  # noqa
        ]  # noqa
        for f, inputs in test_cases:
            inputs = self.gen_inputs(inputs)
            forward_jac = jac(jvp, f, inputs)
            reverse_jac = jac(vjp, f, inputs)
            self.check_results(forward_jac, reverse_jac)

    def test_jvp_i2o2_omitting_v(self):
        test_cases = [  # noqa
            [o2, ['A', 'A']],  # noqa
        ]  # noqa
        for f, inputs in test_cases:
            inputs = self.gen_inputs(inputs)
            results_omitting_v = jvp(f, inputs)
            v = [paddle.ones_like(x) for x in inputs]
            results_with_v = jvp(f, inputs, v)
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
class TestJacobianNoBatch(unittest.TestCase):
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
@param.parameterize(
    (param.TEST_CASE_NAME, 'func', 'xs'),
    (
        ('1d-in-1d-out', utils.square, np.array([[1., 2., 3.], [3., 4., 3.]])),
        ('3d-in-3d-out', utils.square, np.random.rand(2, 3, 4)),
        ('multi-in-single-out', utils.square, np.random.rand(5, 6)),
        # ('multi-in-multi-out', utils.o2, (np.random.rand(2,2), np.random.rand(2,2)))
    ))
class TestJacobianBatchFirst(unittest.TestCase):
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
        return paddle.concat(rows, 1)


if __name__ == "__main__":
    unittest.main()
