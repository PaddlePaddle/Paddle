#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from simple_nets import (
    batchnorm_fc_with_inputs,
    simple_fc_net_with_inputs,
)
from utils import compare_legacy_with_pt

import paddle
from paddle import base
from paddle.base import core, framework
from paddle.base.backward import append_backward

np.random.seed(123)


class TestCondInputOutput(unittest.TestCase):
    @compare_legacy_with_pt
    def test_return_single_var(self):
        """
        pseudocode:

        if 0.23 < 0.1:
            return 2
        else:
            return -1
        """

        paddle.enable_static()

        def true_func():
            return paddle.tensor.fill_constant(
                shape=[2, 3], dtype='int32', value=2
            )

        def false_func():
            return paddle.tensor.fill_constant(
                shape=[3, 2], dtype='int32', value=-1
            )

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.1
            )
            y = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.23
            )
            pred = paddle.less_than(y, x)
            out = paddle.static.nn.cond(pred, true_func, false_func)
            # out is one tensor

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        if paddle.framework.in_pir_mode():
            (ret,) = exe.run(main_program, fetch_list=[out])
        else:
            (ret,) = exe.run(main_program, fetch_list=[out.name])

        np.testing.assert_allclose(
            np.asarray(ret), np.full((3, 2), -1, np.int32), rtol=1e-05
        )

    @compare_legacy_with_pt
    def test_return_0d_tensor(self):
        """
        pseudocode:

        if 0.23 >= 0.1:
            return 2
        else:
            return -1
        """

        paddle.enable_static()

        def true_func():
            return paddle.full(shape=[], dtype='int32', fill_value=2)

        def false_func():
            return paddle.full(shape=[], dtype='int32', fill_value=-1)

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.full(shape=[1], dtype='float32', fill_value=0.1)
            y = paddle.full(shape=[1], dtype='float32', fill_value=0.23)
            pred = paddle.greater_equal(y, x)
            out = paddle.static.nn.cond(pred, true_func, false_func)
            # out is one tensor

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        if paddle.framework.in_pir_mode():
            (ret,) = exe.run(main_program, fetch_list=[out])
        else:
            (ret,) = exe.run(main_program, fetch_list=[out.name])
        np.testing.assert_allclose(np.asarray(ret), np.array(2), rtol=1e-05)
        self.assertEqual(ret.shape, ())

    @compare_legacy_with_pt
    def test_0d_tensor_as_cond(self):
        """
        pseudocode:

        if 0.23 >= 0.1:
            return 2
        else:
            return -1
        """

        paddle.enable_static()

        def true_func():
            return paddle.full(shape=[3, 3], dtype='int32', fill_value=2)

        def false_func():
            return paddle.full(shape=[3, 3], dtype='int32', fill_value=-1)

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.full(shape=[], dtype='float32', fill_value=0.1)
            y = paddle.full(shape=[], dtype='float32', fill_value=0.23)
            pred = paddle.greater_equal(y, x)
            out = paddle.static.nn.cond(pred, true_func, false_func)
            # out is a tensor

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        if paddle.framework.in_pir_mode():
            (ret,) = exe.run(main_program, fetch_list=[out])
        else:
            (ret,) = exe.run(main_program, fetch_list=[out.name])
        np.testing.assert_allclose(
            np.asarray(ret), np.full((3, 3), 2, np.int32), rtol=1e-05
        )

    def test_0d_tensor_backward(self):
        """
        pseudocode:

        a = -2.0
        if a >= 0:
            return a
        else:
            return -a
        """

        paddle.enable_static()

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            a = paddle.full(shape=[], dtype='float32', fill_value=-2.0)
            a.stop_gradient = False
            a.persistable = True
            out = paddle.static.nn.cond(a >= 0, lambda: a, lambda: -a)
            grad_list = append_backward(out)

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )

        exe = base.Executor(place)
        if paddle.framework.in_pir_mode():
            for p, g in grad_list:
                if p.is_same(a):
                    da = g
            ret = exe.run(main_program, fetch_list=[out, da])
        else:
            ret = exe.run(main_program, fetch_list=[out.name, a.grad_name])
        np.testing.assert_allclose(
            np.asarray(ret[0]), np.array(2.0), rtol=1e-05
        )
        self.assertEqual(ret[0].shape, ())
        np.testing.assert_allclose(
            np.asarray(ret[1]), np.array(-1.0), rtol=1e-05
        )
        self.assertEqual(ret[1].shape, ())

    def test_0d_tensor_dygraph(self):
        """
        pseudocode:

        a = -2.0
        if a >= 0:
            return a
        else:
            return -a
        """
        paddle.disable_static()
        a = paddle.full(shape=[], dtype='float32', fill_value=-2.0)
        a.stop_gradient = False
        out = paddle.static.nn.cond(a >= 0, lambda: a, lambda: -a)
        out.backward()

        np.testing.assert_allclose(np.asarray(out), np.array(2.0), rtol=1e-05)
        self.assertEqual(out.shape, [])

        np.testing.assert_allclose(
            np.asarray(a.grad), np.array(-1.0), rtol=1e-05
        )
        self.assertEqual(a.grad.shape, [])

    @compare_legacy_with_pt
    def test_return_var_tuple(self):
        """
        pseudocode:

        if True:
            return 1, True
        else:
            return 3, 2
        """

        paddle.enable_static()

        def true_func():
            return paddle.tensor.fill_constant(
                shape=[1, 2], dtype='int32', value=1
            ), paddle.tensor.fill_constant(
                shape=[2, 3], dtype='bool', value=True
            )

        def false_func():
            return paddle.tensor.fill_constant(
                shape=[3, 4], dtype='int32', value=3
            ), paddle.tensor.fill_constant(
                shape=[4, 5], dtype='bool', value=False
            )

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            pred = paddle.tensor.fill_constant(
                shape=[1], dtype='bool', value=True
            )
            out = paddle.static.nn.cond(pred, true_func, false_func)
            # out is a tuple containing 2 tensors

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        ret = exe.run(main_program, fetch_list=out)
        np.testing.assert_allclose(
            np.asarray(ret[0]), np.full((1, 2), 1, np.int32), rtol=1e-05
        )
        np.testing.assert_allclose(
            np.asarray(ret[1]), np.full((2, 3), True, bool), rtol=1e-05
        )

    @compare_legacy_with_pt
    def test_pass_and_modify_var(self):
        """
        pseudocode:
        for i in range(5):
            a = 7
            if i % 2 == 0:
                a = a * (i + 1)
            else:
                a = a - (i - 1)
        """

        paddle.enable_static()

        def true_func(a, i):
            a = a * (i + 1)
            return a

        def false_func(a, i):
            a = a - (i - 1)
            return a

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            a = paddle.tensor.fill_constant(
                shape=[3, 2, 1], dtype='int32', value=7
            )
            i = paddle.static.data(name="i", shape=[1], dtype='int32')
            pred = paddle.equal((i % 2), 0)
            a = paddle.static.nn.cond(
                pred, lambda: true_func(a, i), lambda: false_func(a, i)
            )
        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        for feed_i in range(5):
            expected_a = 7 * (feed_i + 1) if feed_i % 2 == 0 else 8 - feed_i
            (ret,) = exe.run(
                main_program,
                feed={'i': np.full((1), feed_i, np.int32)},
                fetch_list=[a],
            )
            np.testing.assert_allclose(
                np.asarray(ret),
                np.full((3, 2, 1), expected_a, np.int32),
                rtol=1e-05,
            )

    def test_return_none(self):
        """
        pseudocode: test doing nothing in branches
        for i in range(5):
            if i % 2 == 0:
                pass
            else:
                pass
        """

        paddle.enable_static()

        def true_func():
            pass

        def false_func():
            return None

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.static.data(name="i", shape=[1], dtype='int32')
            pred = paddle.equal((i % 2), 0)
            out1 = paddle.static.nn.cond(pred, true_func, false_func)
            out2 = paddle.static.nn.cond(pred, None, false_func)
            out3 = paddle.static.nn.cond(pred, true_func, None)
        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        for feed_i in range(5):
            # Test that output is None is runnable
            exe.run(main_program, feed={'i': np.full((1), feed_i, np.int32)})
            self.assertIsNone(out1)
            self.assertIsNone(out2)
            self.assertIsNone(out3)

    @compare_legacy_with_pt
    def test_wrong_structure_exception(self):
        """
        test returning different number of tensors cannot merge into output
        """

        paddle.enable_static()

        def func_return_none():
            return None

        def func_return_one_tensor():
            return paddle.tensor.fill_constant(
                shape=[2, 7], dtype='int32', value=3
            )

        def func_return_two_tensors():
            return paddle.tensor.fill_constant(
                shape=[3, 1], dtype='int32', value=7
            ), paddle.tensor.fill_constant(shape=[3, 1], dtype='int32', value=8)

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.static.data(name="i", shape=[1], dtype='int32')
            pred = paddle.equal((i % 2), 0)
            with self.assertRaises(TypeError):
                out = paddle.static.nn.cond(pred, i, func_return_one_tensor)

            with self.assertRaises(TypeError):
                out = paddle.static.nn.cond(
                    pred, func_return_one_tensor, np.asarray([3])
                )

            with self.assertRaises(Exception) as e:
                out = paddle.static.nn.cond(
                    pred, func_return_none, func_return_one_tensor
                )
            self.assertTrue(
                "Incompatible return values of true_fn and false_fn in cond"
                in str(e.exception)
            )

            with self.assertRaises(Exception) as e:
                out = paddle.static.nn.cond(
                    pred, func_return_two_tensors, func_return_none
                )
            self.assertTrue(
                "Incompatible return values of true_fn and false_fn in cond"
                in str(e.exception)
            )

            with self.assertRaises(Exception) as e:
                out = paddle.static.nn.cond(
                    pred, func_return_one_tensor, func_return_two_tensors
                )
            self.assertTrue(
                "true fn returns 1 vars, but false fn returns 2 vars, which is not equals"
                in str(e.exception)
            )

    def test_extremely_simple_net_with_op_in_condition(self):
        paddle.enable_static()
        main_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(main_program, startup_program):
            a = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=1.23
            )
            a.stop_gradient = False
            a.persistable = True
            b = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=1.25
            )
            b.stop_gradient = False
            b.persistable = True
            out = paddle.static.nn.cond(a - b < -1.0, lambda: a, lambda: b)
            grad_list = append_backward(out)

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        if paddle.framework.in_pir_mode():
            for p, g in grad_list:
                if p.is_same(a):
                    da = g
                if p.is_same(b):
                    db = g
            ret = exe.run(main_program, fetch_list=[out, b, da, db])
        else:
            ret = exe.run(
                main_program, fetch_list=[out, b, a.grad_name, b.grad_name]
            )
        # Note: fill_constant has loss of precision, you have to assertEqual
        # with values doesn't lose precision in float-point number.
        self.assertEqual(ret[0][0], ret[1][0])
        self.assertEqual(ret[2][0], 0.0)
        self.assertEqual(ret[3][0], 1.0)


class TestCondNestedControlFlow(unittest.TestCase):

    def test_cond_inside_cond(self):
        """
        pseudocode:
        for i in range(1, 10):
            a = 2 * i
            if i < 5:
                if i >= 3:
                    return a + a
                else:
                    return a - a
            else:
                if i < 8:
                    return a * a
                else:
                    return a / a
        """
        paddle.enable_static()

        def less_than_branch(i, a):
            return paddle.static.nn.cond(
                i >= 3.0,
                lambda: paddle.add(a, a),
                lambda: paddle.subtract(a, a),
            )

        def greater_equal_branch(i, a):
            return paddle.static.nn.cond(
                i < 8.0,
                lambda: paddle.multiply(a, a),
                lambda: paddle.divide(a, a),
            )

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.static.data(name="i", shape=[1], dtype='float32')
            i.stop_gradient = False
            a = 2.0 * i
            a.persistable = True
            out = paddle.static.nn.cond(
                i < 5.0,
                lambda: less_than_branch(i, a),
                lambda: greater_equal_branch(i, a),
            )
            mean = paddle.mean(out)
            grad_list = append_backward(mean)

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        for feed_i in range(0, 10):
            expected_a = 2.0 * feed_i
            if feed_i < 5:
                expected_ret = expected_a + expected_a if feed_i >= 3 else 0.0
                expected_a_grad = 2.0 if feed_i >= 3 else 0.0
            else:
                expected_ret = expected_a * expected_a if feed_i < 8 else 1.0
                expected_a_grad = 2.0 * expected_a if feed_i < 8 else 0.0
            if paddle.framework.in_pir_mode():
                for p, g in grad_list:
                    if p.is_same(a):
                        da = g
                ret = exe.run(
                    main_program,
                    feed={'i': np.full((1), feed_i, np.float32)},
                    fetch_list=[out, da],
                )
            else:
                ret = exe.run(
                    main_program,
                    feed={'i': np.full((1), feed_i, np.float32)},
                    fetch_list=[out.name, a.grad_name],
                )
            self.assertEqual(ret[0][0], expected_ret)
            self.assertEqual(ret[1][0], expected_a_grad)

    def test_cond_inside_cond_0d_tensor(self):
        """
        pseudocode:
            i = 3.0
            a = 2 * i
            if i < 5:
                if i >= 3:
                    return a + 1
                else:
                    return 1 - a
            else:
                if i < 8:
                    return a * 2
                else:
                    return a / 2
        """

        paddle.enable_static()

        def less_than_branch(i, a):
            return paddle.static.nn.cond(
                i >= 3.0,
                lambda: a + 1,
                lambda: 1 - a,
            )

        def greater_equal_branch(i, a):
            return paddle.static.nn.cond(
                i < 8.0,
                lambda: a * 2,
                lambda: a / 2,
            )

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.full(fill_value=3.0, shape=[], dtype='float32')
            i.stop_gradient = False
            i.persistable = True
            a = 2.0 * i
            a.persistable = True
            out = paddle.static.nn.cond(
                i < 5.0,
                lambda: less_than_branch(i, a),
                lambda: greater_equal_branch(i, a),
            )
            mean = paddle.mean(out)
            grad_list = append_backward(mean)

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        if paddle.framework.in_pir_mode():
            for p, g in grad_list:
                if p.is_same(i):
                    di = g
            ret = exe.run(main_program, fetch_list=[out, di])
        else:
            ret = exe.run(
                main_program,
                fetch_list=[out.name, i.grad_name],
            )
        np.testing.assert_allclose(
            np.asarray(ret[0]), np.array(7.0), rtol=1e-05
        )
        self.assertEqual(ret[0].shape, ())
        np.testing.assert_allclose(
            np.asarray(ret[1]), np.array(2.0), rtol=1e-05
        )
        self.assertEqual(ret[1].shape, ())

    def test_cond_op_in_condition(self):
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        with paddle.static.program_guard(main_program, startup_program):
            a = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=1.23
            )
            a.stop_gradient = False
            a.persistable = True
            b = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=1.24
            )
            b.stop_gradient = False
            b.persistable = True
            out = paddle.static.nn.cond(
                a < b,
                lambda: paddle.static.nn.cond(
                    a - b < -1.0,
                    lambda: paddle.add(a, b),
                    lambda: paddle.multiply(a, b),
                ),
                lambda: paddle.static.nn.cond(
                    paddle.equal(a, b),
                    lambda: paddle.subtract(a, b),
                    lambda: paddle.pow(a, b),
                ),
            )
            grad_list = append_backward(out)

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        if paddle.framework.in_pir_mode():
            for p, g in grad_list:
                if p.is_same(a):
                    da = g
                if p.is_same(b):
                    db = g
            ret = exe.run(main_program, fetch_list=[out, da, db])
        else:
            ret = exe.run(
                main_program, fetch_list=[out, a.grad_name, b.grad_name]
            )
        # Note: fill_constant has loss of precision, so we assertAlmostEqual.
        self.assertAlmostEqual(ret[0][0], 1.5252)
        self.assertAlmostEqual(ret[1][0], 1.24)
        self.assertAlmostEqual(ret[2][0], 1.23)


class TestCondBackward(unittest.TestCase):
    def backward_value_helper(self, cond_func, use_cuda):
        """
        Helper function that compares calculated backward value is close to dy/dx
        """
        paddle.enable_static()
        main_program = paddle.static.Program()
        main_program.random_seed = 123
        startup_program = paddle.static.Program()
        startup_program.random_seed = 123
        with paddle.static.scope_guard(paddle.static.Scope()):
            with paddle.static.program_guard(main_program, startup_program):
                img = paddle.static.data(
                    name='image', shape=[-1, 9], dtype='float32'
                )
                img.stop_gradient = False
                if paddle.framework.in_pir_mode():
                    img.persistable = True
                label = paddle.static.data(
                    name='label', shape=[-1, 1], dtype='int64'
                )
                i = paddle.static.data(name="i", shape=[1], dtype='int32')
                loss = cond_func(i, img, label)
                grad_list = append_backward(loss)
            place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
            exe = base.Executor(place)
            exe.run(startup_program)

            num_devices = 1

            delta = 0.005
            for feed_i in range(0, 10):
                feed_img = np.random.random(size=[1, 9]).astype(np.float32)
                feed_label = np.random.randint(
                    low=0, high=10, size=[1, 1], dtype=np.int64
                )
                if paddle.framework.in_pir_mode():
                    for p, g in grad_list:
                        if p.is_same(img):
                            dimg = g
                    img_grad, loss_value = exe.run(
                        main_program,
                        feed={
                            'i': np.full((1), feed_i, np.int32),
                            'image': feed_img,
                            'label': feed_label,
                        },
                        fetch_list=[dimg, loss],
                    )
                else:
                    img_grad, loss_value = exe.run(
                        main_program,
                        feed={
                            'i': np.full((1), feed_i, np.int32),
                            'image': feed_img,
                            'label': feed_label,
                        },
                        fetch_list=[img.grad_name, loss.name],
                    )

                numerical_grad = np.zeros(
                    shape=[num_devices, 9], dtype=np.float32
                )
                feed_img_delta = np.copy(feed_img)
                for j in range(9):
                    feed_img_delta[0][j] = feed_img[0][j] + delta
                    if paddle.framework.in_pir_mode():
                        for p, g in grad_list:
                            if p.is_same(img):
                                dimg = g
                        _, loss_delta = exe.run(
                            main_program,
                            feed={
                                'i': np.full((1), feed_i, np.int32),
                                'image': feed_img_delta,
                                'label': feed_label,
                            },
                            fetch_list=[dimg, loss],
                        )
                    else:
                        loss_delta = exe.run(
                            main_program,
                            feed={
                                'i': np.full((1), feed_i, np.int32),
                                'image': feed_img_delta,
                                'label': feed_label,
                            },
                            fetch_list=[loss],
                        )
                    numerical_grad[0][j] = (loss_delta - loss_value) / delta
                    feed_img_delta[0][j] = feed_img[0][j]
                np.testing.assert_allclose(
                    img_grad, numerical_grad, rtol=0.05, atol=0.05
                )

    def add_optimizer_helper(self, cond_func, use_cuda):
        """
        Test that program is runnable when add optimizer
        """
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.scope_guard(paddle.static.Scope()):
            with paddle.static.program_guard(main_program, startup_program):
                img = paddle.static.data(
                    name='image', shape=[16, 784], dtype='float32'
                )
                img.stop_gradient = False
                if paddle.framework.in_pir_mode():
                    img.persistable = True
                label = paddle.static.data(
                    name='label', shape=[16, 1], dtype='int64'
                )
                i = paddle.static.data(name="i", shape=[1], dtype='int32')
                loss = cond_func(i, img, label)
                optimizer = paddle.optimizer.SGD(learning_rate=0.1)
                optimizer.minimize(loss)

            place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
            exe = base.Executor(place)
            exe.run(startup_program)

            for feed_i in range(0, 10):
                feed_img = np.random.random(size=[16, 784]).astype(np.float32)
                feed_label = np.random.randint(
                    low=0, high=10, size=[16, 1], dtype=np.int64
                )
                exe.run(
                    main_program,
                    feed={
                        'i': np.full((1), feed_i, np.int32),
                        'image': feed_img,
                        'label': feed_label,
                    },
                    fetch_list=[loss],
                )

    @compare_legacy_with_pt
    def test_cond_backward(self):
        paddle.enable_static()

        def cond_func(i, img, label):
            predicate = paddle.equal((i % 2), 0)
            return paddle.static.nn.cond(
                predicate,
                lambda: simple_fc_net_with_inputs(img, label, class_num=10),
                lambda: batchnorm_fc_with_inputs(img, label, class_num=10),
            )

        self.backward_value_helper(cond_func, core.is_compiled_with_cuda())
        self.add_optimizer_helper(cond_func, core.is_compiled_with_cuda())

    def test_half_nested_cond_backward(self):
        paddle.enable_static()
        np.random.seed(2023)
        paddle.seed(2023)

        def branch(i, img, label):
            return paddle.static.nn.cond(
                paddle.equal((i % 2), 0),
                lambda: simple_fc_net_with_inputs(img, label, class_num=10),
                lambda: batchnorm_fc_with_inputs(img, label, class_num=10),
            )

        def cond_func_simple_net_at_true(i, img, label):
            return paddle.static.nn.cond(
                i < 5, lambda: branch(i, img, label), lambda: paddle.mean(img)
            )

        def cond_func_simple_net_at_false(i, img, label):
            return paddle.static.nn.cond(
                i < 5, lambda: paddle.mean(img), lambda: branch(i, img, label)
            )

        self.backward_value_helper(
            cond_func_simple_net_at_true,
            core.is_compiled_with_cuda(),
        )

        self.backward_value_helper(
            cond_func_simple_net_at_false,
            core.is_compiled_with_cuda(),
        )
        self.add_optimizer_helper(
            cond_func_simple_net_at_true,
            core.is_compiled_with_cuda(),
        )
        self.add_optimizer_helper(
            cond_func_simple_net_at_false,
            core.is_compiled_with_cuda(),
        )

    def test_nested_cond_backward(self):
        paddle.enable_static()
        np.random.seed(2023)
        paddle.seed(2023)

        def branch(i, img, label, mod_two):
            if mod_two:
                predicate = paddle.equal((i % 2), 0)
            else:
                predicate = (i % 2) != 0
            return paddle.static.nn.cond(
                predicate,
                lambda: simple_fc_net_with_inputs(img, label, class_num=10),
                lambda: batchnorm_fc_with_inputs(img, label, class_num=10),
            )

        def cond_func(i, img, label):
            return paddle.static.nn.cond(
                i < 5,
                lambda: branch(i, img, label, True),
                lambda: branch(i, img, label, False),
            )

        self.backward_value_helper(cond_func, core.is_compiled_with_cuda())
        self.add_optimizer_helper(cond_func, core.is_compiled_with_cuda())


class TestCondWithError(unittest.TestCase):
    @compare_legacy_with_pt
    def test_input_type_error(self):
        paddle.enable_static()
        main_program = framework.Program()
        startup_program = framework.Program()
        with framework.program_guard(main_program, startup_program):
            pred = paddle.static.data(name='y', shape=[1], dtype='bool')

            def func():
                return pred

            with self.assertRaises(TypeError):
                paddle.static.nn.cond(None, func, func)

            with self.assertRaises(TypeError):
                paddle.static.nn.cond(pred, func, set())

            with self.assertRaises(TypeError):
                paddle.static.nn.cond(pred, set(), func)

            with self.assertRaises(TypeError):
                paddle.static.nn.cond(pred, func, func, set())


class TestCondWithDict(unittest.TestCase):

    @compare_legacy_with_pt
    def test_input_with_dict(self):
        paddle.enable_static()
        main_program = framework.Program()
        startup_program = framework.Program()
        with framework.program_guard(main_program, startup_program):

            def true_func():
                return {
                    '1': paddle.full(shape=[3, 2], dtype='int32', fill_value=1),
                    '2': paddle.full(
                        shape=[2, 3], dtype='bool', fill_value=True
                    ),
                }

            def false_func():
                return {
                    '1': paddle.full(shape=[3, 4], dtype='int32', fill_value=3),
                    '2': paddle.full(shape=[4, 5], dtype='bool', fill_value=2),
                }

            x = paddle.full(shape=[1], dtype='float32', fill_value=0.1)
            y = paddle.full(shape=[1], dtype='float32', fill_value=0.23)
            pred = paddle.less_than(x=x, y=y, name=None)
            ret = paddle.static.nn.cond(pred, true_func, false_func)

            if paddle.framework.in_pir_mode():
                self.assertEqual(
                    ret['1'].shape,
                    [3, -1],
                    f"The shape is not correct, expects [3, -1] but gets {ret['1'].shape}.",
                )
                self.assertEqual(
                    ret['2'].shape,
                    [-1, -1],
                    f"The shape is not correct, expects [-1, -1] but gets {ret['2'].shape}.",
                )
            else:
                self.assertEqual(
                    ret['1'].shape,
                    (3, -1),
                    f"The shape is not correct, expects (3, -1) but gets {ret['1'].shape}.",
                )
                self.assertEqual(
                    ret['2'].shape,
                    (-1, -1),
                    f"The shape is not correct, expects (-1, -1) but gets {ret['2'].shape}.",
                )


if __name__ == '__main__':
    unittest.main()
