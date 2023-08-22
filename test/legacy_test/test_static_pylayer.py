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

import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.base.backward import append_backward
from paddle.base.framework import Program, program_guard

np.random.seed(123)


class TestStatocPyLayerInputOutput(unittest.TestCase):
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


# TODO(MarioLulab): Disable now. We will refine and add testcases later.
class _TestControlFlowNestedStaticPyLayer(unittest.TestCase):
    # TODO(MarioLulab): failed when i >= 5, fix it later
    def _test_cond_inside_static_pylayer(self):
        """
        forward pass:
                      _ _ _ _ _ _ _ _
         ---> a ---> |               | -----> out_i
        |            | StaticPyLayer |
        i ---------> |_ _ _ _ _ _ _ _| -----> out ---> loss


        pseudocode:
        def forward_fn(i, a):
            if i < 5:
                return i, a + a
            else:
                return i, a - a

        def backward_fn(diout, daout):
            if diout < 5:
                return 2 * diout, daout * daout
            else:
                return 2 * diout, cos(daout)
        """

        paddle.enable_static()

        def forward_fn(i, a):
            return i, paddle.static.nn.cond(
                i < 5.0, lambda: paddle.add(a, a), lambda: paddle.subtract(a, a)
            )

        def backward_fn(diout, daout):
            return 2 * diout, paddle.static.nn.cond(
                diout < 5.0,
                lambda: paddle.multiply(daout, daout),
                lambda: paddle.cos(daout),
            )

        main_program = Program()
        start_program = Program()
        with program_guard(main_program, start_program):
            i = paddle.static.data(name="i", shape=[1], dtype="float32")
            i.stop_gradient = False
            a = 2.0 * i
            out_i, out = paddle.static.nn.static_pylayer(
                forward_fn, [i, a], backward_fn
            )
            loss = paddle.exp(out)
            append_backward(loss)

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        for feed_i in range(0, 10):
            print(feed_i)
            expected_a = 2.0 * feed_i
            if feed_i < 5:
                expected_out_i = feed_i
                expected_out = expected_a + expected_a
                expected_out_grad = np.exp(expected_out)
                expected_a_grad = expected_out_grad * expected_out_grad
                expected_i_grad = 2 * expected_a_grad + 0
            else:
                expected_out_i = feed_i
                expected_out = expected_a - expected_a
                expected_out_grad = np.exp(expected_out)
                expected_a_grad = np.cos(expected_out_grad)
                expected_i_grad = 2 * expected_a_grad + 0
            ret = exe.run(
                main_program,
                feed={'i': np.full((1), feed_i, dtype=np.float32)},
                fetch_list=[out.name, out.grad_name, a.grad_name, i.grad_name],
            )
            np.testing.assert_allclose(
                np.asarray(ret[0]), expected_out, rtol=1e-05
            )
            np.testing.assert_allclose(
                np.asarray(ret[1]), expected_out_grad, rtol=1e-05
            )
            np.testing.assert_allclose(
                np.asarray(ret[2]), expected_a_grad, rtol=1e-05
            )
            np.testing.assert_allclose(
                np.asarray(ret[3]), expected_i_grad, rtol=1e-05
            )


if __name__ == '__main__':
    unittest.main()
