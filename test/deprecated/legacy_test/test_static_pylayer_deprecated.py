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
import sys
import unittest

sys.path.append(".")
import numpy as np
from test_prune_deprecated import (
    TestExecutorRunAutoPrune,
    TestPruneBase,
)

import paddle
from paddle import base
from paddle.base import core
from paddle.base.backward import append_backward

np.random.seed(123)


class TestStaticPyLayerInputOutput(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def test_return_single_var(self):
        """
        pseudocode:

        y = 3 * x
        """

        def forward_fn(x):
            return 3 * x

        main_program = paddle.static.Program()
        start_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, start_program):
            data = paddle.static.data(name="X", shape=[1], dtype="float32")
            out = paddle.static.nn.static_pylayer(forward_fn, [data])

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        x = np.array([2.0], dtype=np.float32)
        (ret,) = exe.run(main_program, feed={"X": x}, fetch_list=[out])
        np.testing.assert_allclose(
            np.asarray(ret), np.array([6.0], np.float32), rtol=1e-05
        )

    # NOTE: Users should not be able to return none when actually using it.

    def test_return_0d_tensor(self):
        """
        pseudocode:

        y = 3 * x
        """

        def forward_fn(x):
            return 3 * x

        main_program = paddle.static.Program()
        start_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, start_program):
            data = paddle.full(shape=[], dtype='float32', fill_value=2.0)
            out = paddle.static.nn.static_pylayer(forward_fn, [data])

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        (ret,) = exe.run(main_program, fetch_list=[out])
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

        def forward_fn(x):
            return 3 * x

        def backward_fn(dy):
            return -5 * dy

        main_program = paddle.static.Program()
        start_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, start_program):
            data = paddle.full(shape=[], dtype='float32', fill_value=-2.0)
            data.stop_gradient = False
            out = paddle.static.nn.static_pylayer(
                forward_fn, [data], backward_fn
            )
            grad_list = append_backward(out, [data])

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)

        if paddle.framework.in_pir_mode():
            for p, g in grad_list:
                if p.is_same(data):
                    data_grad = g
            ret, x_grad = exe.run(
                main_program,
                fetch_list=[out, data_grad],
            )
        else:
            ret, x_grad = exe.run(
                main_program,
                fetch_list=[out.name, data.grad_name],
            )

        np.testing.assert_allclose(np.asarray(ret), np.array(-6.0), rtol=1e-05)
        self.assertEqual(ret.shape, ())

        np.testing.assert_allclose(
            np.asarray(x_grad), np.array(-5.0), rtol=1e-05
        )
        self.assertEqual(x_grad.shape, ())

    def test_return_var_type(self):
        def forward_fn(a, b):
            return 3 * a, -2 * b

        main_program = paddle.static.Program()
        start_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, start_program):
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
        ret_1, ret_2 = exe.run(main_program, fetch_list=[out_1, out_2])
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
        input_shape = (1, 3)

        def forward_fn(x):
            y = 3 * x

        main_program = paddle.static.Program()
        start_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, start_program):
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

        def forward_fn(a, b):
            return 3 * a, -b, paddle.mean(b)

        def backward_fn(daout, dbout):
            return 3 * daout, -dbout

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
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
                append_backward(out, [data_1, data_2])


class TestControlFlowNestedStaticPyLayer(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

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

        main_program = paddle.static.Program()
        start_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, start_program):
            i = paddle.static.data(name="i", shape=[1], dtype="float32")
            i.stop_gradient = False
            a = 2.0 * i
            out_i, out_a = paddle.static.nn.static_pylayer(
                forward_fn, [i, a], backward_fn
            )
            out = out_i + out_a
            loss = paddle.exp(out)
            grad_list = append_backward(loss, [i, a, out_i, out_a, out])

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

            if paddle.framework.in_pir_mode():
                out_grad = None
                out_i_grad = None
                out_a_grad = None
                a_grad = None
                i_grad = None

                for p, g in grad_list:
                    if p.is_same(out_i):
                        out_i_grad = g
                    elif p.is_same(out_a):
                        out_a_grad = g
                    elif p.is_same(a):
                        a_grad = g
                    elif p.is_same(i):
                        i_grad = g
                    elif p.is_same(out):
                        out_grad = g

                ret = exe.run(
                    main_program,
                    feed={'i': np.full((1), feed_i, dtype=np.float32)},
                    fetch_list=[
                        out,
                        out_grad,
                        out_i_grad,
                        out_a_grad,
                        a_grad,
                        i_grad,
                    ],
                )
            else:
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
    def setUp(self):
        paddle.enable_static()

    def test_identity_backward(self):
        def forward_fn(x):
            return x

        def backward_fn(dy):
            return dy

        main_program = paddle.static.Program()
        start_program = paddle.static.Program()
        input_shape = (2, 4)
        with paddle.static.program_guard(main_program, start_program):
            data = paddle.static.data(
                name="X", shape=input_shape, dtype="float32"
            )
            data.stop_gradient = False
            out = paddle.static.nn.static_pylayer(
                forward_fn, [data], backward_fn
            )
            loss = paddle.mean(out)
            grad_list = append_backward(loss, [data])

        place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        exe = base.Executor(place)
        randn_x = np.random.random(size=input_shape).astype(np.float32)

        if paddle.framework.in_pir_mode():
            for p, g in grad_list:
                if p.is_same(data):
                    data_grad = g
            ret, x_grad = exe.run(
                main_program,
                feed={
                    'X': randn_x,
                },
                fetch_list=[out, data_grad],
            )
        else:
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

        def forward_fn(x):
            return 3 * x

        def backward_fn(dy):
            return paddle.tanh(dy)

        main_program = paddle.static.Program()
        start_program = paddle.static.Program()
        input_shape = (3, 4)
        with paddle.static.program_guard(main_program, start_program):
            data = paddle.full(
                shape=input_shape, dtype='float32', fill_value=-2.0
            )
            data.stop_gradient = False
            out = paddle.static.nn.static_pylayer(
                forward_fn, [data], backward_fn
            )
            loss = paddle.mean(out)
            grad_list = append_backward(loss, [data])

        place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        exe = base.Executor(place)

        if paddle.framework.in_pir_mode():
            for p, g in grad_list:
                if p.is_same(data):
                    data_grad = g
            ret, x_grad = exe.run(
                main_program,
                fetch_list=[out, data_grad],
            )
        else:
            ret, x_grad = exe.run(
                main_program,
                fetch_list=[out.name, data.grad_name],
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


class TestStaticPyLayerPrune(TestPruneBase):
    def setUp(self):
        paddle.enable_static()

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
        ops_before_pruned = [
            "mul",
            "elementwise_add",
            "softmax",
            "pylayer",
            "reduce_mean",
        ]

        ops_after_pruned = ["pylayer", "reduce_mean"]

        (x, hidden, y, loss), program = self.run_net(self.net)

        self.check_prune_with_input(
            program, [hidden.name], [loss], ops_before_pruned, ops_after_pruned
        )

    def test_prune(self):
        ops_before_pruned = [
            "mul",
            "elementwise_add",
            "softmax",
            "pylayer",
            "reduce_mean",
        ]

        ops_after_pruned = [
            "mul",
            "elementwise_add",
            "softmax",
            "pylayer",
            "reduce_mean",
        ]

        (x, hidden, y, loss), program = self.run_net(self.net)

        self.check_prune(program, [loss], ops_before_pruned, ops_after_pruned)

    def test_prune_target_not_list(self):
        ops_before_pruned = [
            "mul",
            "elementwise_add",
            "softmax",
            "pylayer",
            "reduce_mean",
        ]

        ops_after_pruned = [
            "mul",
            "elementwise_add",
            "softmax",
            "pylayer",
            "reduce_mean",
        ]

        (x, hidden, y, loss), program = self.run_net(self.net)
        self.check_prune_target_not_list(
            program, loss, ops_before_pruned, ops_after_pruned
        )

    def test_prune_target_none(self):
        ops_before_pruned = [
            "mul",
            "elementwise_add",
            "softmax",
            "pylayer",
            "reduce_mean",
        ]

        (x, hidden, y, loss), program = self.run_net(self.net)
        self.check_prune_target_none(program, ops_before_pruned)


def net_with_weight1():
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


def net_with_weight2():
    def forward_fn(x):
        y = 3 * x
        return y

    def backward_fn(dy):
        grad = paddle.exp(dy)
        return grad

    x1 = paddle.static.data(name='x1', shape=[-1, 2], dtype='float32')
    x1.desc.set_need_check_feed(False)
    x2 = paddle.static.data(name='x2', shape=[-1, 2], dtype='float32')
    x2.desc.set_need_check_feed(False)
    label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")
    label.desc.set_need_check_feed(False)
    w1_param_attrs = base.ParamAttr(
        name="fc_weight1",
        learning_rate=0.5,
        initializer=paddle.nn.initializer.Constant(1.0),
        trainable=True,
    )
    w2_param_attrs = base.ParamAttr(
        name="fc_weight2",
        learning_rate=0.5,
        initializer=paddle.nn.initializer.Constant(1.0),
        trainable=True,
    )

    y1 = paddle.static.nn.static_pylayer(forward_fn, [x1], backward_fn)
    hidden1 = paddle.static.nn.fc(
        x=[y1], size=4, activation="softmax", weight_attr=w1_param_attrs
    )
    y2 = paddle.static.nn.static_pylayer(forward_fn, [x2], backward_fn)
    hidden2 = paddle.static.nn.fc(
        x=[y2], size=4, activation="softmax", weight_attr=w2_param_attrs
    )

    loss1 = paddle.nn.functional.cross_entropy(
        input=hidden1, label=label, reduction='none', use_softmax=False
    )
    loss1 = paddle.mean(x=loss1)
    loss2 = paddle.nn.functional.cross_entropy(
        input=hidden2, label=label, reduction='none', use_softmax=False
    )
    loss2 = paddle.mean(x=loss2)
    loss1.persistable = True
    loss2.persistable = True

    return x1, x2, y1, y2, label, loss1, loss2, w1_param_attrs, w2_param_attrs


class TestStaticPyLayerExecutorAutoPrune(TestExecutorRunAutoPrune):
    def setUp(self):
        paddle.enable_static()
        self.net1 = net_with_weight1
        self.net2 = net_with_weight2


if __name__ == '__main__':
    unittest.main()
