# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import operator
import unittest

import numpy as np

import paddle


class TestTensorInplaceOperators(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def _check(self, op, x_np, y, expected):
        x = paddle.to_tensor(x_np)
        alias = x
        out = op(x, y)
        # In-place semantics: the operator returns the very same object, so
        # every alias observes the update.
        self.assertIs(out, alias)
        np.testing.assert_allclose(alias.numpy(), expected, rtol=1e-6)

    def test_float_scalar_and_tensor(self):
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
        y_np = np.array([[2.0, 4.0], [5.0, 8.0]], dtype='float32')
        cases = [
            (operator.iadd, np.add),
            (operator.isub, np.subtract),
            (operator.imul, np.multiply),
            (operator.itruediv, np.true_divide),
        ]
        for op, ref in cases:
            self._check(op, x_np, 2.0, ref(x_np, 2.0))
            self._check(op, x_np, paddle.to_tensor(y_np), ref(x_np, y_np))

    def test_integer_ops(self):
        x_np = np.array([[7, 9], [11, 13]], dtype='int32')
        y_np = np.array([[2, 4], [3, 5]], dtype='int32')
        cases = [
            (operator.ifloordiv, np.floor_divide),
            (operator.imod, np.mod),
            (operator.iand, np.bitwise_and),
            (operator.ior, np.bitwise_or),
            (operator.ixor, np.bitwise_xor),
            (operator.ilshift, np.left_shift),
            (operator.irshift, np.right_shift),
        ]
        for op, ref in cases:
            self._check(op, x_np, 2, ref(x_np, 2))
            self._check(op, x_np, paddle.to_tensor(y_np), ref(x_np, y_np))

    def test_pow(self):
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
        self._check(operator.ipow, x_np, 2, x_np**2)

    def test_alias_in_container(self):
        # The case from the bug report: a /= 2 must also update tensors["a"].
        tensors = {"a": paddle.ones([3, 3])}
        a = tensors["a"]
        a /= 2
        np.testing.assert_allclose(tensors["a"].numpy(), np.full([3, 3], 0.5))

    def test_integer_true_divide_raises(self):
        x = paddle.to_tensor([2, 4], dtype='int32')
        with self.assertRaises(RuntimeError):
            x /= 2

    def test_float_operand_into_integer_raises(self):
        x = paddle.to_tensor([2, 4], dtype='int32')
        with self.assertRaises(RuntimeError):
            x += 1.5
        with self.assertRaises(RuntimeError):
            x += paddle.to_tensor([1.5, 2.5])

    def test_broadcast_larger_than_target_raises(self):
        x = paddle.ones([3, 1])
        with self.assertRaises(ValueError):
            x += paddle.ones([3, 4])

    def test_leaf_requiring_grad_raises(self):
        x = paddle.ones([2, 2])
        x.stop_gradient = False
        with self.assertRaises(ValueError):
            x += 1.0

    def test_non_leaf_requiring_grad(self):
        x = paddle.ones([2, 2])
        x.stop_gradient = False
        y = x * 2
        y += 1.0
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), np.full([2, 2], 2.0))

    def test_tensor_exponent_falls_back_to_out_of_place(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x **= paddle.to_tensor([2.0, 2.0, 2.0])
        np.testing.assert_allclose(x.numpy(), [1.0, 4.0, 9.0])


class TestTensorInplaceOperatorsStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_value_not_patched(self):
        # Static graph Value has no in-place semantics and keeps ``a = a + b``.
        self.assertFalse(hasattr(paddle.pir.Value, '__iadd__'))

    def test_static_graph_keeps_out_of_place_semantics(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data('x', [2, 2], 'float32')
            y = x
            y += 1.0
            # ``y += 1.0`` builds a new Value, exactly as before this change.
            self.assertIsNot(y, x)
        exe = paddle.static.Executor(paddle.CPUPlace())
        (out,) = exe.run(
            main,
            feed={'x': np.ones([2, 2], dtype='float32')},
            fetch_list=[y],
        )
        np.testing.assert_allclose(out, np.full([2, 2], 2.0))


if __name__ == '__main__':
    unittest.main()
