#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from functools import partial

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.base.backward import append_backward
from paddle.base.framework import Program, program_guard

paddle.enable_static()


class TestAPICase(unittest.TestCase):

    def test_return_single_var(self):
        def fn_1():
            return paddle.tensor.fill_constant(
                shape=[4, 2], dtype='int32', value=1
            )

        def fn_2():
            return paddle.tensor.fill_constant(
                shape=[4, 2], dtype='int32', value=2
            )

        def fn_3():
            return paddle.tensor.fill_constant(
                shape=[4, 3], dtype='int32', value=3
            )

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.3
            )
            y = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.1
            )
            z = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.2
            )
            pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
            pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3

            # call fn_1
            out_0 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(pred_1, fn_1), (pred_1, fn_2)], default=fn_3
            )

            # call fn_2
            out_1 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(pred_2, fn_1), (pred_1, fn_2)], default=fn_3
            )

            # call default fn_3
            out_2 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=((pred_2, fn_1), (pred_2, fn_2)), default=fn_3
            )

            # no default, call fn_2
            out_3 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(pred_1, fn_2)]
            )

            # no default, call fn_2. but pred_2 is false
            out_4 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(pred_2, fn_2)]
            )

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = base.Executor(place)

            res = exe.run(
                main_program, fetch_list=[out_0, out_1, out_2, out_3, out_4]
            )

            np.testing.assert_allclose(res[0], 1, rtol=1e-05)
            np.testing.assert_allclose(res[1], 2, rtol=1e-05)
            np.testing.assert_allclose(res[2], 3, rtol=1e-05)
            np.testing.assert_allclose(res[3], 2, rtol=1e-05)
            np.testing.assert_allclose(res[4], 2, rtol=1e-05)

    def test_0d_tensor(self):
        def fn_1():
            return paddle.full(shape=[], dtype='int32', fill_value=1)

        def fn_2():
            return paddle.full(shape=[], dtype='int32', fill_value=2)

        def fn_3():
            return paddle.full(shape=[], dtype='int32', fill_value=3)

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.full(shape=[], dtype='float32', fill_value=0.3)
            y = paddle.full(shape=[], dtype='float32', fill_value=0.1)
            z = paddle.full(shape=[], dtype='float32', fill_value=0.2)
            pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
            pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3

            # call fn_1
            out_0 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(pred_1, fn_1), (pred_1, fn_2)], default=fn_3
            )

            # call fn_2
            out_1 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(pred_2, fn_1), (pred_1, fn_2)], default=fn_3
            )

            # call default fn_3
            out_2 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=((pred_2, fn_1), (pred_2, fn_2)), default=fn_3
            )

            # no default, call fn_2
            out_3 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(pred_1, fn_2)]
            )

            # no default, call fn_2. but pred_2 is false
            out_4 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(pred_2, fn_2)]
            )

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = base.Executor(place)

            res = exe.run(
                main_program, fetch_list=[out_0, out_1, out_2, out_3, out_4]
            )

            np.testing.assert_allclose(res[0], 1, rtol=1e-05)
            self.assertEqual(res[0].shape, ())
            np.testing.assert_allclose(res[1], 2, rtol=1e-05)
            self.assertEqual(res[1].shape, ())
            np.testing.assert_allclose(res[2], 3, rtol=1e-05)
            self.assertEqual(res[2].shape, ())
            np.testing.assert_allclose(res[3], 2, rtol=1e-05)
            self.assertEqual(res[3].shape, ())
            np.testing.assert_allclose(res[4], 2, rtol=1e-05)
            self.assertEqual(res[4].shape, ())

    def test_0d_tensor_backward(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.full(shape=[], dtype='float32', fill_value=-2.0)
            x.stop_gradient = False
            x.persistable = True
            pred = paddle.full(shape=[], dtype='bool', fill_value=0)
            # pred is False, so out = -x
            out = paddle.static.nn.case(
                pred_fn_pairs=[(pred, lambda: x)], default=lambda: -x
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
                if p.is_same(x):
                    dx = g
            res = exe.run(main_program, fetch_list=[out, dx])
        else:
            res = exe.run(main_program, fetch_list=[out.name, x.grad_name])

        np.testing.assert_allclose(
            np.asarray(res[0]), np.array(2.0), rtol=1e-05
        )
        self.assertEqual(res[0].shape, ())
        np.testing.assert_allclose(
            np.asarray(res[1]), np.array(-1.0), rtol=1e-05
        )
        self.assertEqual(res[1].shape, ())

    def test_0d_tensor_dygraph(self):
        paddle.disable_static()

        def fn_1():
            return paddle.full(shape=[], dtype='int32', fill_value=1)

        def fn_2():
            return paddle.full(shape=[], dtype='int32', fill_value=2)

        def fn_3():
            return paddle.full(shape=[], dtype='int32', fill_value=3)

        x = paddle.full(shape=[], dtype='float32', fill_value=0.3)
        y = paddle.full(shape=[], dtype='float32', fill_value=0.1)
        z = paddle.full(shape=[], dtype='float32', fill_value=0.2)
        pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
        pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3

        # call fn_1
        out_0 = paddle.static.nn.control_flow.case(
            pred_fn_pairs=[(pred_1, fn_1), (pred_1, fn_2)], default=fn_3
        )

        # call fn_2
        out_1 = paddle.static.nn.control_flow.case(
            pred_fn_pairs=[(pred_2, fn_1), (pred_1, fn_2)], default=fn_3
        )

        # call default fn_3
        out_2 = paddle.static.nn.control_flow.case(
            pred_fn_pairs=((pred_2, fn_1), (pred_2, fn_2)), default=fn_3
        )

        # no default, call fn_2
        out_3 = paddle.static.nn.control_flow.case(
            pred_fn_pairs=[(pred_1, fn_2)]
        )

        # no default, call fn_2. but pred_2 is false
        out_4 = paddle.static.nn.control_flow.case(
            pred_fn_pairs=[(pred_2, fn_2)]
        )

        np.testing.assert_allclose(out_0, 1, rtol=1e-05)
        self.assertEqual(out_0.shape, [])
        np.testing.assert_allclose(out_1, 2, rtol=1e-05)
        self.assertEqual(out_1.shape, [])
        np.testing.assert_allclose(out_2, 3, rtol=1e-05)
        self.assertEqual(out_2.shape, [])
        np.testing.assert_allclose(out_3, 2, rtol=1e-05)
        self.assertEqual(out_3.shape, [])
        np.testing.assert_allclose(out_4, 2, rtol=1e-05)
        self.assertEqual(out_4.shape, [])

        paddle.enable_static()

    def test_return_var_tuple(self):
        def fn_1():
            return paddle.tensor.fill_constant(
                shape=[1, 2], dtype='int32', value=1
            ), paddle.tensor.fill_constant(
                shape=[2, 3], dtype='float32', value=2
            )

        def fn_2():
            return paddle.tensor.fill_constant(
                shape=[3, 4], dtype='int32', value=3
            ), paddle.tensor.fill_constant(
                shape=[4, 5], dtype='float32', value=4
            )

        def fn_3():
            return paddle.tensor.fill_constant(
                shape=[5, 6], dtype='int32', value=5
            ), paddle.tensor.fill_constant(
                shape=[5, 6], dtype='float32', value=6
            )

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=1)
            y = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=1)
            z = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=3)

            pred_1 = paddle.equal(x, y)  # true
            pred_2 = paddle.equal(x, z)  # false

            out = paddle.static.nn.control_flow.case(
                ((pred_1, fn_1), (pred_2, fn_2)), fn_3
            )

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
                np.asarray(ret[1]), np.full((2, 3), 2, np.float32), rtol=1e-05
            )


class TestAPICase_Nested(unittest.TestCase):

    def test_nested_case(self):
        def fn_1(x=1):
            var_5 = paddle.tensor.fill_constant(
                shape=[1], dtype='int32', value=5
            )
            var_6 = paddle.tensor.fill_constant(
                shape=[1], dtype='int32', value=6
            )
            out = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[
                    (
                        var_5 < var_6,
                        partial(
                            paddle.tensor.fill_constant,
                            shape=[1],
                            dtype='int32',
                            value=x,
                        ),
                    ),
                    (
                        var_5 == var_6,
                        partial(
                            paddle.tensor.fill_constant,
                            shape=[2],
                            dtype='int32',
                            value=x,
                        ),
                    ),
                ]
            )
            return out

        def fn_2(x=2):
            var_5 = paddle.tensor.fill_constant(
                shape=[1], dtype='int32', value=5
            )
            var_6 = paddle.tensor.fill_constant(
                shape=[1], dtype='int32', value=6
            )
            out = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[
                    (var_5 < var_6, partial(fn_1, x=x)),
                    (
                        var_5 == var_6,
                        partial(
                            paddle.tensor.fill_constant,
                            shape=[2],
                            dtype='int32',
                            value=x,
                        ),
                    ),
                ]
            )
            return out

        def fn_3():
            var_5 = paddle.tensor.fill_constant(
                shape=[1], dtype='int32', value=5
            )
            var_6 = paddle.tensor.fill_constant(
                shape=[1], dtype='int32', value=6
            )
            out = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[
                    (var_5 < var_6, partial(fn_2, x=3)),
                    (
                        var_5 == var_6,
                        partial(
                            paddle.tensor.fill_constant,
                            shape=[2],
                            dtype='int32',
                            value=7,
                        ),
                    ),
                ]
            )
            return out

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.3
            )
            y = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.1
            )
            z = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.2
            )
            pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
            pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3

            out_1 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3
            )
            out_2 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(pred_2, fn_1), (pred_1, fn_2)], default=fn_3
            )

            out_3 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(x == y, fn_1), (x == z, fn_2)], default=fn_3
            )

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = base.Executor(place)

            res = exe.run(main_program, fetch_list=[out_1, out_2, out_3])

            np.testing.assert_allclose(res[0], 1, rtol=1e-05)
            np.testing.assert_allclose(res[1], 2, rtol=1e-05)
            np.testing.assert_allclose(res[2], 3, rtol=1e-05)

    def test_nested_0d_tensor(self):
        def fn_1(x=1):
            var_5 = paddle.full(shape=[], dtype='int32', fill_value=5)
            var_6 = paddle.full(shape=[], dtype='int32', fill_value=6)
            out = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[
                    (
                        var_5 < var_6,
                        partial(
                            paddle.full,
                            shape=[],
                            dtype='int32',
                            fill_value=x,
                        ),
                    ),
                    (
                        var_5 == var_6,
                        partial(
                            paddle.full,
                            shape=[],
                            dtype='int32',
                            fill_value=x,
                        ),
                    ),
                ]
            )
            return out

        def fn_2(x=2):
            var_5 = paddle.full(shape=[], dtype='int32', fill_value=5)
            var_6 = paddle.full(shape=[], dtype='int32', fill_value=6)
            out = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[
                    (var_5 < var_6, partial(fn_1, x=x)),
                    (
                        var_5 == var_6,
                        partial(
                            paddle.full,
                            shape=[],
                            dtype='int32',
                            fill_value=x,
                        ),
                    ),
                ]
            )
            return out

        def fn_3():
            var_5 = paddle.full(shape=[], dtype='int32', fill_value=5)
            var_6 = paddle.full(shape=[], dtype='int32', fill_value=6)
            out = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[
                    (var_5 < var_6, partial(fn_2, x=3)),
                    (
                        var_5 == var_6,
                        partial(
                            paddle.full,
                            shape=[],
                            dtype='int32',
                            fill_value=7,
                        ),
                    ),
                ]
            )
            return out

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.full(shape=[], dtype='float32', fill_value=0.3)
            y = paddle.full(shape=[], dtype='float32', fill_value=0.1)
            z = paddle.full(shape=[], dtype='float32', fill_value=0.2)
            pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
            pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3

            out_1 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3
            )

            out_2 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(pred_2, fn_1), (pred_1, fn_2)], default=fn_3
            )

            out_3 = paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(x == y, fn_1), (x == z, fn_2)], default=fn_3
            )

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = base.Executor(place)

            res = exe.run(main_program, fetch_list=[out_1, out_2, out_3])

            np.testing.assert_allclose(res[0], 1, rtol=1e-05)
            self.assertEqual(res[0].shape, ())
            np.testing.assert_allclose(res[1], 2, rtol=1e-05)
            self.assertEqual(res[1].shape, ())
            np.testing.assert_allclose(res[2], 3, rtol=1e-05)
            self.assertEqual(res[2].shape, ())


class TestAPICase_Error(unittest.TestCase):

    def test_error(self):
        def fn_1():
            return paddle.tensor.fill_constant(
                shape=[4, 2], dtype='int32', value=1
            )

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.23
            )
            z = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.2
            )
            pred_1 = paddle.less_than(z, x)  # true

            # The type of 'pred_fn_pairs' in case must be list or tuple
            def type_error_pred_fn_pairs():
                paddle.static.nn.control_flow.case(
                    pred_fn_pairs=1, default=fn_1
                )

            self.assertRaises(TypeError, type_error_pred_fn_pairs)

            # The elements' type of 'pred_fn_pairs' in Op(case) must be tuple
            def type_error_pred_fn_1():
                paddle.static.nn.control_flow.case(
                    pred_fn_pairs=[1], default=fn_1
                )

            self.assertRaises(TypeError, type_error_pred_fn_1)

            # The tuple's size of 'pred_fn_pairs' in Op(case) must be 2
            def type_error_pred_fn_2():
                paddle.static.nn.control_flow.case(
                    pred_fn_pairs=[(1, 2, 3)], default=fn_1
                )

            self.assertRaises(TypeError, type_error_pred_fn_2)

            # The pred's type of 'pred_fn_pairs' in Op(case) must be bool Variable
            def type_error_pred():
                paddle.static.nn.control_flow.case(
                    pred_fn_pairs=[(1, fn_1)], default=fn_1
                )

            self.assertRaises(TypeError, type_error_pred)

            # The function of pred_fn_pairs in case must be callable
            def type_error_fn():
                paddle.static.nn.control_flow.case(
                    pred_fn_pairs=[(pred_1, 2)], default=fn_1
                )

            self.assertRaises(TypeError, type_error_fn)

            # The default in Op(case) must be callable
            def type_error_default():
                paddle.static.nn.control_flow.case(
                    pred_fn_pairs=[(pred_1, fn_1)], default=fn_1()
                )

            self.assertRaises(TypeError, type_error_default)


# when optimizer in case
class TestMultiTask(unittest.TestCase):

    def test_optimizer_in_case(self):
        BATCH_SIZE = 1
        INPUT_SIZE = 784
        EPOCH_NUM = 2
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                name='x', shape=[BATCH_SIZE, INPUT_SIZE], dtype='float32'
            )
            y = paddle.static.data(
                name='y', shape=[BATCH_SIZE, INPUT_SIZE], dtype='float32'
            )
            x.stop_gradient = False
            y.stop_gradient = False
            switch_id = paddle.static.data(
                name='switch_id', shape=[1], dtype='int32'
            )

            one = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=1)
            adam = paddle.optimizer.Adam(learning_rate=0.001)
            adagrad = paddle.optimizer.Adagrad(learning_rate=0.001)

            def fn_1():
                sum = paddle.multiply(x, y)
                loss = paddle.mean(sum, name="f_1_loss")
                adam.minimize(loss)

            def fn_2():
                sum = paddle.multiply(x, y)
                loss = paddle.mean(sum, name="f_2_loss")
                adagrad.minimize(loss)

            paddle.static.nn.control_flow.case(
                pred_fn_pairs=[(switch_id == one, fn_1)], default=fn_2
            )

            exe = base.Executor(base.CPUPlace())
            exe.run(startup_program)

            for epoch in range(EPOCH_NUM):
                np.random.seed(epoch)
                feed_image = np.random.random(
                    size=[BATCH_SIZE, INPUT_SIZE]
                ).astype('float32')
                out = exe.run(
                    main_program,
                    feed={
                        'x': feed_image,
                        'y': feed_image,
                        'switch_id': np.array([epoch]).astype('int32'),
                    },
                    fetch_list=[],
                )


if __name__ == '__main__':
    unittest.main()
