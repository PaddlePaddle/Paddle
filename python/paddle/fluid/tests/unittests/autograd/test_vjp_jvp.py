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
import paddle

from paddle.autograd.functional import vjp, jvp, to_tensorlist
from paddle import grad, ones_like, zeros_like


def reduce(x):
    return paddle.sum(x)


def reduce_dim(x):
    return paddle.sum(x, axis=0)


def matmul(x, y):
    return paddle.matmul(x, y)


def mul(x, y):
    return x * y


def pow(x, y):
    return paddle.pow(x, y)


def o2(x, y):
    return paddle.multiply(x, y), paddle.matmul(x, y.t())


def unuse(x, y):
    return paddle.sum(x)


def nested(x):
    def inner(y):
        return x * y

    return inner


def make_v(f, inputs):
    outputs = to_tensorlist(f(*inputs))
    return [ones_like(x) for x in outputs]


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
                outputs, inputs_grad = vjp(func,
                                           xs,
                                           v,
                                           create_graph=create_graph,
                                           allow_unused=allow_unused)
            else:
                outputs, inputs_grad = vjp(func,
                                           xs,
                                           create_graph=create_graph,
                                           allow_unused=allow_unused)
            return outputs, inputs_grad

        def grad_test():
            nonlocal v
            xs = self.gen_inputs(inputs)
            if v is not None:
                v = self.gen_inputs(v)
            outputs = func(*xs)
            if v is not None:
                inputs_grad = grad(
                    outputs,
                    xs,
                    v,
                    create_graph=create_graph,
                    allow_unused=allow_unused)
            else:
                inputs_grad = grad(
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
            self.assertTrue(paddle.allclose(res, ref), value_error)
        else:
            self.assertTrue(len(res) == len(ref), type_error)
            for i in range(len(ref)):
                self.check_results(ref[i], res[i])
        return True


class TestVJP(TestAutogradFunctional):
    def test_vjp_i1o1_no_create_graph(self):
        test_cases = [
            [reduce, 'A'],  #noqa
            [reduce_dim, 'A'],  #noqa
        ]  #noqa
        for f, inputs in test_cases:
            vjp, grad = self.gen_test_pairs(f, inputs)
            vjp_result, grad_result = vjp(), grad()
            self.check_results(grad_result, vjp_result)

    def test_vjp_i2o1_no_create_graph(self):
        test_cases = [
            [matmul, ['A', 'B']],  #noqa
            [mul, ['b', 'c']],  #noqa
        ]  #noqa
        for f, inputs in test_cases:
            vjp, grad = self.gen_test_pairs(f, inputs)
            vjp_result, grad_result = vjp(), grad()
            self.check_results(grad_result, vjp_result)

    def test_vjp_i2o2_no_create_graph(self):
        test_cases = [
            [o2, ['A', 'A']],  #noqa
        ]  #noqa
        for f, inputs in test_cases:
            inputs = self.gen_inputs(inputs)
            v = make_v(f, inputs)
            vjp, grad = self.gen_test_pairs(f, inputs, v=v)
            vjp_result, grad_result = vjp(), grad()
            self.check_results(grad_result, vjp_result)

    def test_vjp_nested_no_create_graph(self):
        x = self.gen_input('a')
        test_cases = [
            [nested(x), 'a'],  #noqa
        ]
        for f, inputs in test_cases:
            vjp, grad = self.gen_test_pairs(f, inputs)
            vjp_result, grad_result = vjp(), grad()
            self.check_results(grad_result, vjp_result)

    def test_vjp_aliased_input_no_create_graph(self):
        x = self.gen_input('a')
        ref = self.gen_test_pairs(nested(x), 'a')[0]
        aliased = self.gen_test_pairs(nested(x), x)[0]
        ref_result, aliased_result = ref(), aliased()
        self.check_results(ref_result, aliased_result)

    def test_vjp_allowunused_no_create_graph(self):
        x, y = self.gen_input('A'), self.gen_input('a')
        vjp, grad = self.gen_test_pairs(unuse, [x, y], allow_unused=True)
        vjp_result, grad_result = vjp(), grad()
        self.check_results(grad_result, vjp_result)


def jac(grad_fn, f, inputs):
    assert grad_fn in [vjp, jvp]
    if grad_fn is jvp:
        vs = [zeros_like(x) for x in inputs]
    else:
        outputs = f(*inputs)
        if isinstance(outputs, paddle.Tensor):
            outputs = [outputs]
        vs = [zeros_like(y) for y in outputs]
    JJ_cols = []
    for i, v in enumerate(vs):
        v = v.flatten()
        for j in range(len(v)):
            _v = zeros_like(v).detach()
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
    def test_jvp_i1o1_no_create_graph(self):
        test_cases = [
            [reduce, 'A'],  #noqa
            [reduce_dim, 'A'],  #noqa
        ]  #noqa
        for f, inputs in test_cases:
            inputs = self.gen_inputs(inputs)
            forward_jac = jac(jvp, f, inputs)
            reverse_jac = jac(vjp, f, inputs)
            self.check_results(forward_jac, reverse_jac)

    def test_jvp_i2o1_no_create_graph(self):
        test_cases = [  #noqa
            [matmul, ['A', 'B']],  #noqa
        ]  #noqa
        for f, inputs in test_cases:
            inputs = self.gen_inputs(inputs)
            forward_jac = jac(jvp, f, inputs)
            reverse_jac = jac(vjp, f, inputs)
            self.check_results(forward_jac, reverse_jac)

    def test_jvp_i2o2_no_create_graph(self):
        test_cases = [  #noqa
            [o2, ['A', 'A']],  #noqa
        ]  #noqa
        for f, inputs in test_cases:
            inputs = self.gen_inputs(inputs)
            forward_jac = jac(jvp, f, inputs)
            reverse_jac = jac(vjp, f, inputs)
            self.check_results(forward_jac, reverse_jac)


if __name__ == "__main__":
    unittest.main()
