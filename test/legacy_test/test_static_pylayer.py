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

import functools
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core, framework
from paddle.base.backward import append_backward
from paddle.base.framework import Program, program_guard

np.random.seed(123)


class TestStaticPyLayerInputOutput(unittest.TestCase):
    def test_return_single_var(self):
        """
        pseudocode:

        y = 3 * x
        """

        paddle.enable_static()

        def forward_fn(x):
            return 3 * x

        main_program = Program()
        start_program = Program()
        with program_guard(main_program, start_program):
            data = paddle.static.data(name="X", shape=[1], dtype="float32")
            out = paddle.static.nn.static_pylayer(forward_fn, [data])

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        x = np.array([2.0], dtype=np.float32)
        (ret,) = exe.run(main_program, feed={"X": x}, fetch_list=[out.name])
        np.testing.assert_allclose(
            np.asarray(ret), np.array([6.0], np.float32), rtol=1e-05
        )

    # NOTE: Users should not be able to return none when actually using it.
    def test_return_0d_tensor(self):
        """
        pseudocode:

        y = 3 * x
        """

        paddle.enable_static()

        def forward_fn(x):
            return 3 * x

        main_program = Program()
        start_program = Program()
        with program_guard(main_program, start_program):
            data = paddle.full(shape=[], dtype='float32', fill_value=2.0)
            out = paddle.static.nn.static_pylayer(forward_fn, [data])

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        (ret,) = exe.run(main_program, fetch_list=[out.name])
        np.testing.assert_allclose(
            np.asarray(ret), np.array(6.0, np.float32), rtol=1e-05
        )
        self.assertEqual(ret.shape, ())

    def test_0d_tensor_backward(self):
        '''
        pseudocode:

        y = 3 * x
        dx = -5 * dy
        '''

        paddle.enable_static()

        def forward_fn(x):
            return 3 * x

        def backward_fn(dy):
            return -5 * dy

        main_program = Program()
        start_program = Program()
        with program_guard(main_program, start_program):
            data = paddle.full(shape=[], dtype='float32', fill_value=-2.0)
            data.stop_gradient = False
            out = paddle.static.nn.static_pylayer(
                forward_fn, [data], backward_fn
            )
            append_backward(out)

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        ret, x_grad = exe.run(
            main_program, fetch_list=[out.name, data.grad_name]
        )
        np.testing.assert_allclose(np.asarray(ret), np.array(-6.0), rtol=1e-05)
        self.assertEqual(ret.shape, ())

        np.testing.assert_allclose(
            np.asarray(x_grad), np.array(-5.0), rtol=1e-05
        )
        self.assertEqual(x_grad.shape, ())

    def test_return_var_typle(self):
        paddle.enable_static()

        def forward_fn(a, b):
            return 3 * a, -2 * b

        main_program = Program()
        start_program = Program()
        with program_guard(main_program, start_program):
            data_1 = paddle.full(shape=[2, 4], dtype='float32', fill_value=-2.0)
            data_2 = paddle.full(shape=[4, 5], dtype='float32', fill_value=10.0)
            out_1, out_2 = paddle.static.nn.static_pylayer(
                forward_fn, [data_1, data_2]
            )

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        ret_1, ret_2 = exe.run(
            main_program, fetch_list=[out_1.name, out_2.name]
        )
        np.testing.assert_allclose(
            np.asarray(ret_1),
            np.full((2, 4), -6.0, dtype=np.float32),
            rtol=1e-05,
        )

        np.testing.assert_allclose(
            np.asarray(ret_2),
            np.full((4, 5), -20.0, dtype=np.float32),
            rtol=1e-05,
        )

    def test_return_forward_none(self):
        paddle.enable_static()

        input_shape = (1, 3)

        def forward_fn(x):
            y = 3 * x
            return None

        main_program = Program()
        start_program = Program()
        with program_guard(main_program, start_program):
            data = paddle.full(
                shape=input_shape, dtype='float32', fill_value=-2.0
            )
            out = paddle.static.nn.static_pylayer(forward_fn, [data])

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        exe.run(main_program)
        self.assertIsNone(out)

    def test_wrong_structure_exception(self):
        """
        test not all ``stop_gradient`` of inputs is True when ``backward_fn`` is None, and
        wrong number of inputs and outputs returned by ``forward_fn`` and ``backward_fn``
        """

        paddle.enable_static()

        def forward_fn(a, b):
            return 3 * a, -b, paddle.mean(b)

        def backward_fn(daout, dbout):
            return 3 * daout, -dbout

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            data_1 = paddle.static.data(
                name="data_1", shape=[2, 4], dtype="float32"
            )
            data_2 = paddle.static.data(
                name="data_2", shape=[6], dtype="float32"
            )
            data_2.stop_gradient = False
            with self.assertRaises(ValueError) as e:
                out = paddle.static.nn.static_pylayer(
                    forward_fn, [data_1, data_2], backward_fn=None
                )
            self.assertTrue(
                "``stop_gradient`` attr of all inputs to ``forward_fn`` are expected to be True, when ``backward_fn == None``"
                in str(e.exception)
            )

            with self.assertRaises(TypeError) as e:
                out = paddle.static.nn.static_pylayer(
                    forward_fn, [data_1, data_2], backward_fn=backward_fn
                )


class TestControlFlowNestedStaticPyLayer(unittest.TestCase):
    def test_cond_inside_static_pylayer(self):
        """
        forward propagation:
                      _ _ _ _ _ _ _ _
         ---> a ---> |               | -----> out_a ------
        |            | StaticPyLayer |                    |
        i ---------> |_ _ _ _ _ _ _ _| -----> out_i ---> out ---> loss


        pseudocode:
        def forward_fn(i, a):
            if i < 5:
                return i, a + a
            else:
                return i, a - a

        def backward_fn(diout, daout):
            daout_scaled = daout * 3.0
            if diout < 5:
                return daout_scaled, -1 * daout
            else:
                return daout_scaled, daout * daout
        """

        paddle.enable_static()

        def forward_fn(i, a):
            return i, paddle.static.nn.cond(
                i < 5.0, lambda: paddle.add(a, a), lambda: paddle.subtract(a, a)
            )

        def backward_fn(diout, daout):
            daout_scale = daout * 3.0
            return daout_scale, paddle.static.nn.cond(
                diout < 5.0,
                lambda: -1 * daout,
                lambda: daout * daout,
            )

        main_program = Program()
        start_program = Program()
        with program_guard(main_program, start_program):
            i = paddle.static.data(name="i", shape=[1], dtype="float32")
            i.stop_gradient = False
            a = 2.0 * i
            out_i, out_a = paddle.static.nn.static_pylayer(
                forward_fn, [i, a], backward_fn
            )
            out = out_i + out_a
            loss = paddle.exp(out)
            append_backward(loss)

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        for feed_i in range(0, 10):
            expected_a = 2.0 * feed_i
            if feed_i < 5:
                expected_out_i = feed_i
                expected_out_a = expected_a + expected_a
                expected_out = expected_out_a + expected_out_i
                expected_out_grad = np.exp(expected_out)
            else:
                expected_out_i = feed_i
                expected_out_a = expected_a - expected_a
                expected_out = expected_out_a + expected_out_i
                expected_out_grad = np.exp(expected_out)

            if expected_out_grad < 5:
                expected_a_grad = -1 * expected_out_grad
                expected_i_grad = 3 * expected_out_grad + 2 * expected_a_grad
            else:
                expected_a_grad = expected_out_grad * expected_out_grad
                expected_i_grad = 3 * expected_out_grad + 2 * expected_a_grad

            ret = exe.run(
                main_program,
                feed={'i': np.full((1), feed_i, dtype=np.float32)},
                fetch_list=[
                    out.name,
                    out.grad_name,
                    out_i.grad_name,
                    out_a.grad_name,
                    a.grad_name,
                    i.grad_name,
                ],
            )
            np.testing.assert_allclose(
                np.asarray(ret[0]), expected_out, rtol=1e-05
            )
            np.testing.assert_allclose(
                np.asarray(ret[1]), expected_out_grad, rtol=1e-05
            )
            np.testing.assert_allclose(
                np.asarray(ret[2]), expected_out_grad, rtol=1e-05
            )
            np.testing.assert_allclose(
                np.asarray(ret[3]), expected_out_grad, rtol=1e-05
            )
            np.testing.assert_allclose(
                np.asarray(ret[4]), expected_a_grad, rtol=1e-05
            )
            np.testing.assert_allclose(
                np.asarray(ret[5]), expected_i_grad, rtol=1e-05
            )


class TestStaticPyLayerBackward(unittest.TestCase):
    def test_identity_backward(self):
        paddle.enable_static()

        def forward_fn(x):
            return x

        def backward_fn(dy):
            return dy

        main_program = Program()
        start_program = Program()
        input_shape = (2, 4)
        with program_guard(main_program, start_program):
            data = paddle.static.data(
                name="X", shape=input_shape, dtype="float32"
            )
            data.stop_gradient = False
            out = paddle.static.nn.static_pylayer(
                forward_fn, [data], backward_fn
            )
            loss = paddle.mean(out)
            append_backward(loss)

        place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        exe = base.Executor(place)
        randn_x = np.random.random(size=input_shape).astype(np.float32)
        ret, x_grad = exe.run(
            main_program,
            feed={
                'X': randn_x,
            },
            fetch_list=[out.name, data.grad_name],
        )

        np.testing.assert_allclose(
            np.asarray(ret),
            randn_x,
            rtol=1e-05,
        )

        np.testing.assert_allclose(
            np.asarray(x_grad),
            np.full(
                input_shape,
                1.0 / functools.reduce(lambda x, y: x * y, input_shape),
                dtype=np.float32,
            ),
            rtol=1e-05,
        )

    def test_static_pylayer_backward(self):
        '''
        pseudocode:

        y = 3 * x
        dx = tanh(dy)
        '''

        paddle.enable_static()

        def forward_fn(x):
            return 3 * x

        def backward_fn(dy):
            return paddle.tanh(dy)

        main_program = Program()
        start_program = Program()
        input_shape = (3, 4)
        with program_guard(main_program, start_program):
            data = paddle.full(
                shape=input_shape, dtype='float32', fill_value=-2.0
            )
            data.stop_gradient = False
            out = paddle.static.nn.static_pylayer(
                forward_fn, [data], backward_fn
            )
            loss = paddle.mean(out)
            append_backward(loss)

        place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        exe = base.Executor(place)
        ret, x_grad = exe.run(
            main_program, fetch_list=[out.name, data.grad_name]
        )
        np.testing.assert_allclose(
            np.asarray(ret),
            np.full(input_shape, -6.0, dtype=np.float32),
            rtol=1e-05,
        )

        np.testing.assert_allclose(
            np.asarray(x_grad),
            np.full(
                input_shape,
                np.tanh(
                    1.0 / functools.reduce(lambda x, y: x * y, input_shape)
                ),
                dtype=np.float32,
            ),
            rtol=1e-05,
        )


class TestStaticPyLayerBackward(unittest.TestCase):
    def test_identity_backward(self):
        paddle.enable_static()

        def forward_fn(x):
            return x

        def backward_fn(dy):
            return dy

        main_program = Program()
        start_program = Program()
        input_shape = (2, 4)
        with program_guard(main_program, start_program):
            data = paddle.static.data(
                name="X", shape=input_shape, dtype="float32"
            )
            data.stop_gradient = False
            out = paddle.static.nn.static_pylayer(
                forward_fn, [data], backward_fn
            )
            loss = paddle.mean(out)
            append_backward(loss)

        place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        exe = base.Executor(place)
        randn_x = np.random.random(size=input_shape).astype(np.float32)
        ret, x_grad = exe.run(
            main_program,
            feed={
                'X': randn_x,
            },
            fetch_list=[out.name, data.grad_name],
        )

        np.testing.assert_allclose(
            np.asarray(ret),
            randn_x,
            rtol=1e-05,
        )

        np.testing.assert_allclose(
            np.asarray(x_grad),
            np.full(
                input_shape,
                1.0 / functools.reduce(lambda x, y: x * y, input_shape),
                dtype=np.float32,
            ),
            rtol=1e-05,
        )

    def test_static_pylayer_backward(self):
        '''
        pseudocode:

        y = 3 * x
        dx = tanh(dy)
        '''

        paddle.enable_static()

        def forward_fn(x):
            return 3 * x

        def backward_fn(dy):
            return paddle.tanh(dy)

        main_program = Program()
        start_program = Program()
        input_shape = (3, 4)
        with program_guard(main_program, start_program):
            data = paddle.full(
                shape=input_shape, dtype='float32', fill_value=-2.0
            )
            data.stop_gradient = False
            out = paddle.static.nn.static_pylayer(
                forward_fn, [data], backward_fn
            )
            loss = paddle.mean(out)
            append_backward(loss)

        place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        exe = base.Executor(place)
        ret, x_grad = exe.run(
            main_program, fetch_list=[out.name, data.grad_name]
        )
        np.testing.assert_allclose(
            np.asarray(ret),
            np.full(input_shape, -6.0, dtype=np.float32),
            rtol=1e-05,
        )

        np.testing.assert_allclose(
            np.asarray(x_grad),
            np.full(
                input_shape,
                np.tanh(
                    1.0 / functools.reduce(lambda x, y: x * y, input_shape)
                ),
                dtype=np.float32,
            ),
            rtol=1e-05,
        )


class TestStaticPyLayerPrune(unittest.TestCase):
    def net(self):
        def forward_fn(x):
            y = 3 * x
            return y

        def backward_fn(dy):
            grad = paddle.exp(dy)
            return grad

        x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
        x.desc.set_need_check_feed(False)
        hidden = paddle.static.nn.fc(x=[x], size=4, activation="softmax")
        y = paddle.static.nn.static_pylayer(forward_fn, [hidden], backward_fn)
        loss = paddle.mean(y)
        return x, hidden, y, loss

    def net_with_weight(self):
        def forward_fn(x):
            y = 3 * x
            return y

        def backward_fn(dy):
            grad = paddle.exp(dy)
            return grad

        x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
        x.desc.set_need_check_feed(False)
        label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")
        label.desc.set_need_check_feed(False)
        w_param_attrs = base.ParamAttr(
            name="fc_weight",
            learning_rate=0.5,
            initializer=paddle.nn.initializer.Constant(1.0),
            trainable=True,
        )

        y = paddle.static.nn.static_pylayer(forward_fn, [x], backward_fn)
        hidden = paddle.static.nn.fc(
            x=[y], size=4, activation="softmax", weight_attr=w_param_attrs
        )
        loss1 = paddle.nn.functional.cross_entropy(
            input=hidden, label=label, reduction='none', use_softmax=False
        )
        loss1 = paddle.mean(x=loss1)
        loss2 = paddle.nn.functional.cross_entropy(
            input=hidden, label=label, reduction='none', use_softmax=False
        )
        loss2 = paddle.mean(x=loss2)
        loss1.persistable = True
        loss2.persistable = True

        return x, hidden, label, loss1, loss2, w_param_attrs

    def test_prune_with_input(self):
        paddle.enable_static()

        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with base.program_guard(program, startup_program):
            (x, hidden, y, loss) = self.net()

        self.assertEqual(len(block.ops), 5)
        self.assertEqual(
            [op.type for op in block.ops],
            [
                "mul",
                "elementwise_add",
                "softmax",
                "pylayer",
                "reduce_mean",
            ],
        )

        pruned_program = program._prune_with_input(
            feeded_var_names=[hidden.name], targets=[loss]
        )

        self.assertEqual(len(pruned_program.global_block().ops), 2)
        self.assertEqual(
            [op.type for op in pruned_program.global_block().ops],
            ["pylayer", "reduce_mean"],
        )

    def test_prune(self):
        paddle.enable_static()

        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with base.program_guard(program, startup_program):
            (x, hidden, y, loss) = self.net()

        self.assertEqual(len(block.ops), 5)
        self.assertEqual(
            [op.type for op in block.ops],
            [
                "mul",
                "elementwise_add",
                "softmax",
                "pylayer",
                "reduce_mean",
            ],
        )

        pruned_program = program._prune(targets=[loss])

        self.assertEqual(len(pruned_program.global_block().ops), 5)
        self.assertEqual(
            [op.type for op in pruned_program.global_block().ops],
            [
                "mul",
                "elementwise_add",
                "softmax",
                "pylayer",
                "reduce_mean",
            ],
        )

    def test_prune_fetches_without_optimizer(self):
        """
        Prune operators and variables which are not needed to generate 'fetches'.
        """
        paddle.enable_static()

        program = framework.Program()
        startup_program = framework.Program()
        scope = base.Scope()
        with base.scope_guard(scope):
            with base.program_guard(program, startup_program):
                (
                    x,
                    y,
                    label,
                    loss1,
                    loss2,
                    w_param_attrs,
                ) = self.net_with_weight()
                exe = base.Executor(base.CPUPlace())
                exe.run(startup_program)
                weight_init = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor()
                )
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(
                    program,
                    feed={'x': x_np, 'label': label_np},
                    fetch_list=[loss1.name],
                    use_prune=True,
                )
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))  # loss2 is pruned
                weight = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor()
                )
                np.testing.assert_array_equal(
                    weight_init, weight
                )  # weight not changed

    def test_prune_fetches_with_optimizer(self):
        """
        Prune operators and operators which are not needed to generate 'fetches'.
        In train mode, the operators and operators in backward and optimization should be kept.
        """
        paddle.enable_static()

        program = framework.Program()
        startup_program = framework.Program()
        scope = base.Scope()
        with base.scope_guard(scope):
            with base.program_guard(program, startup_program):
                (
                    x,
                    y,
                    label,
                    loss1,
                    loss2,
                    w_param_attrs,
                ) = self.net_with_weight()
                sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.5)
                sgd_optimizer.minimize(loss1)
                exe = base.Executor(base.CPUPlace())
                exe.run(startup_program)
                weight_init = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor()
                )
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(
                    program,
                    feed={'x': x_np, 'label': label_np},
                    fetch_list=[loss1.name],
                    use_prune=True,
                )
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))  # loss2 is pruned
                weight = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor()
                )
                self.assertFalse(
                    np.array_equal(weight_init, weight)
                )  # weight changed

    def test_prune_compiled_program(self):
        paddle.enable_static()

        program = framework.Program()
        startup_program = framework.Program()
        scope = base.Scope()
        with base.scope_guard(scope):
            with base.program_guard(program, startup_program):
                (
                    x,
                    y,
                    label,
                    loss1,
                    loss2,
                    w_param_attrs,
                ) = self.net_with_weight()
                sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.5)
                sgd_optimizer.minimize(loss1)
                exe = base.Executor(base.CPUPlace())
                exe.run(startup_program)
                compiled_prog = base.CompiledProgram(program)
                weight_init = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor()
                )
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(
                    compiled_prog,
                    feed={'x': x_np, 'label': label_np},
                    fetch_list=[loss1.name],
                    use_prune=True,
                )
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))
                weight = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor()
                )
                self.assertFalse(
                    np.array_equal(weight_init, weight)
                )  # weight changed


class TestLoadModel(unittest.TestCase):
    ...


if __name__ == '__main__':
    unittest.main()
