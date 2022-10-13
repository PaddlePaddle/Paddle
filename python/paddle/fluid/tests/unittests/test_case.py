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

import numpy as np
import unittest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from paddle.fluid.framework import Program, program_guard
from functools import partial
import paddle.fluid.optimizer as optimizer


class TestAPICase(unittest.TestCase):

    def test_return_single_var(self):

        def fn_1():
            return layers.fill_constant(shape=[4, 2], dtype='int32', value=1)

        def fn_2():
            return layers.fill_constant(shape=[4, 2], dtype='int32', value=2)

        def fn_3():
            return layers.fill_constant(shape=[4, 3], dtype='int32', value=3)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = layers.fill_constant(shape=[1], dtype='float32', value=0.3)
            y = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
            z = layers.fill_constant(shape=[1], dtype='float32', value=0.2)
            pred_2 = layers.less_than(x, y)  # false: 0.3 < 0.1
            pred_1 = layers.less_than(z, x)  # true: 0.2 < 0.3

            # call fn_1
            out_0 = layers.case(pred_fn_pairs=[(pred_1, fn_1), (pred_1, fn_2)],
                                default=fn_3)

            # call fn_2
            out_1 = layers.case(pred_fn_pairs=[(pred_2, fn_1), (pred_1, fn_2)],
                                default=fn_3)

            # call default fn_3
            out_2 = layers.case(pred_fn_pairs=((pred_2, fn_1), (pred_2, fn_2)),
                                default=fn_3)

            # no default, call fn_2
            out_3 = layers.case(pred_fn_pairs=[(pred_1, fn_2)])

            # no default, call fn_2. but pred_2 is false
            out_4 = layers.case(pred_fn_pairs=[(pred_2, fn_2)])

            place = fluid.CUDAPlace(
                0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
            exe = fluid.Executor(place)

            res = exe.run(main_program,
                          fetch_list=[out_0, out_1, out_2, out_3, out_4])

            np.testing.assert_allclose(res[0], 1, rtol=1e-05)
            np.testing.assert_allclose(res[1], 2, rtol=1e-05)
            np.testing.assert_allclose(res[2], 3, rtol=1e-05)
            np.testing.assert_allclose(res[3], 2, rtol=1e-05)
            np.testing.assert_allclose(res[4], 2, rtol=1e-05)

    def test_return_var_tuple(self):

        def fn_1():
            return layers.fill_constant(shape=[1, 2], dtype='int32',
                                        value=1), layers.fill_constant(
                                            shape=[2, 3],
                                            dtype='float32',
                                            value=2)

        def fn_2():
            return layers.fill_constant(shape=[3, 4], dtype='int32',
                                        value=3), layers.fill_constant(
                                            shape=[4, 5],
                                            dtype='float32',
                                            value=4)

        def fn_3():
            return layers.fill_constant(shape=[5], dtype='int32',
                                        value=5), layers.fill_constant(
                                            shape=[5, 6],
                                            dtype='float32',
                                            value=6)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = layers.fill_constant(shape=[1], dtype='float32', value=1)
            y = layers.fill_constant(shape=[1], dtype='float32', value=1)
            z = layers.fill_constant(shape=[1], dtype='float32', value=3)

            pred_1 = layers.equal(x, y)  # true
            pred_2 = layers.equal(x, z)  # false

            out = layers.case(((pred_1, fn_1), (pred_2, fn_2)), fn_3)

            place = fluid.CUDAPlace(
                0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
            exe = fluid.Executor(place)
            ret = exe.run(main_program, fetch_list=out)

            np.testing.assert_allclose(np.asarray(ret[0]),
                                       np.full((1, 2), 1, np.int32),
                                       rtol=1e-05)
            np.testing.assert_allclose(np.asarray(ret[1]),
                                       np.full((2, 3), 2, np.float32),
                                       rtol=1e-05)


class TestAPICase_Nested(unittest.TestCase):

    def test_nested_case(self):

        def fn_1(x=1):
            var_5 = layers.fill_constant(shape=[1], dtype='int32', value=5)
            var_6 = layers.fill_constant(shape=[1], dtype='int32', value=6)
            out = layers.case(pred_fn_pairs=[
                (var_5 < var_6,
                 partial(
                     layers.fill_constant, shape=[1], dtype='int32', value=x)),
                (var_5 == var_6,
                 partial(
                     layers.fill_constant, shape=[2], dtype='int32', value=x))
            ])
            return out

        def fn_2(x=2):
            var_5 = layers.fill_constant(shape=[1], dtype='int32', value=5)
            var_6 = layers.fill_constant(shape=[1], dtype='int32', value=6)
            out = layers.case(pred_fn_pairs=[
                (var_5 < var_6, partial(fn_1, x=x)),
                (var_5 == var_6,
                 partial(
                     layers.fill_constant, shape=[2], dtype='int32', value=x))
            ])
            return out

        def fn_3():
            var_5 = layers.fill_constant(shape=[1], dtype='int32', value=5)
            var_6 = layers.fill_constant(shape=[1], dtype='int32', value=6)
            out = layers.case(pred_fn_pairs=[
                (var_5 < var_6, partial(fn_2, x=3)),
                (var_5 == var_6,
                 partial(
                     layers.fill_constant, shape=[2], dtype='int32', value=7))
            ])
            return out

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = layers.fill_constant(shape=[1], dtype='float32', value=0.3)
            y = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
            z = layers.fill_constant(shape=[1], dtype='float32', value=0.2)
            pred_2 = layers.less_than(x, y)  # false: 0.3 < 0.1
            pred_1 = layers.less_than(z, x)  # true: 0.2 < 0.3

            out_1 = layers.case(pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)],
                                default=fn_3)

            out_2 = layers.case(pred_fn_pairs=[(pred_2, fn_1), (pred_1, fn_2)],
                                default=fn_3)

            out_3 = layers.case(pred_fn_pairs=[(x == y, fn_1), (x == z, fn_2)],
                                default=fn_3)

            place = fluid.CUDAPlace(
                0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
            exe = fluid.Executor(place)

            res = exe.run(main_program, fetch_list=[out_1, out_2, out_3])

            np.testing.assert_allclose(res[0], 1, rtol=1e-05)
            np.testing.assert_allclose(res[1], 2, rtol=1e-05)
            np.testing.assert_allclose(res[2], 3, rtol=1e-05)


class TestAPICase_Error(unittest.TestCase):

    def test_error(self):

        def fn_1():
            return layers.fill_constant(shape=[4, 2], dtype='int32', value=1)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = layers.fill_constant(shape=[1], dtype='float32', value=0.23)
            z = layers.fill_constant(shape=[1], dtype='float32', value=0.2)
            pred_1 = layers.less_than(z, x)  # true

            # The type of 'pred_fn_pairs' in case must be list or tuple
            def type_error_pred_fn_pairs():
                layers.case(pred_fn_pairs=1, default=fn_1)

            self.assertRaises(TypeError, type_error_pred_fn_pairs)

            # The elements' type of 'pred_fn_pairs' in Op(case) must be tuple
            def type_error_pred_fn_1():
                layers.case(pred_fn_pairs=[1], default=fn_1)

            self.assertRaises(TypeError, type_error_pred_fn_1)

            # The tuple's size of 'pred_fn_pairs' in Op(case) must be 2
            def type_error_pred_fn_2():
                layers.case(pred_fn_pairs=[(1, 2, 3)], default=fn_1)

            self.assertRaises(TypeError, type_error_pred_fn_2)

            # The pred's type of 'pred_fn_pairs' in Op(case) must be bool Variable
            def type_error_pred():
                layers.case(pred_fn_pairs=[(1, fn_1)], default=fn_1)

            self.assertRaises(TypeError, type_error_pred)

            # The function of pred_fn_pairs in case must be callable
            def type_error_fn():
                layers.case(pred_fn_pairs=[(pred_1, 2)], default=fn_1)

            self.assertRaises(TypeError, type_error_fn)

            # The default in Op(case) must be callable
            def type_error_default():
                layers.case(pred_fn_pairs=[(pred_1, fn_1)], default=fn_1())

            self.assertRaises(TypeError, type_error_default)


# when optimizer in case
class TestMutiTask(unittest.TestCase):

    def test_optimizer_in_case(self):
        BATCH_SIZE = 1
        INPUT_SIZE = 784
        EPOCH_NUM = 2

        x = fluid.data(name='x',
                       shape=[BATCH_SIZE, INPUT_SIZE],
                       dtype='float32')
        y = fluid.data(name='y',
                       shape=[BATCH_SIZE, INPUT_SIZE],
                       dtype='float32')

        switch_id = fluid.data(name='switch_id', shape=[1], dtype='int32')

        one = layers.fill_constant(shape=[1], dtype='int32', value=1)
        adam = optimizer.Adam(learning_rate=0.001)
        adagrad = optimizer.Adagrad(learning_rate=0.001)

        def fn_1():
            sum = layers.elementwise_mul(x, y)
            loss = paddle.mean(sum, name="f_1_loss")
            adam.minimize(loss)

        def fn_2():
            sum = layers.elementwise_mul(x, y)
            loss = paddle.mean(sum, name="f_2_loss")
            adagrad.minimize(loss)

        layers.case(pred_fn_pairs=[(switch_id == one, fn_1)], default=fn_2)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        for epoch in range(EPOCH_NUM):
            np.random.seed(epoch)
            feed_image = np.random.random(
                size=[BATCH_SIZE, INPUT_SIZE]).astype('float32')
            main_program = fluid.default_main_program()
            out = exe.run(main_program,
                          feed={
                              'x': feed_image,
                              'y': feed_image,
                              'switch_id': np.array([epoch]).astype('int32')
                          },
                          fetch_list=[])


if __name__ == '__main__':
    unittest.main()
