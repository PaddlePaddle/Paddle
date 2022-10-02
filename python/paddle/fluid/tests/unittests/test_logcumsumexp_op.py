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

from typing import Optional
import unittest
import itertools
import numpy as np
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.framework import _test_eager_guard
from op_test import OpTest


def np_naive_logcumsumexp(x: np.ndarray, axis: Optional[int] = None):
    return np.log(np.cumsum(np.exp(x), axis=axis))


def np_logcumsumexp(x: np.ndarray,
                    axis: Optional[int] = None,
                    flatten: Optional[bool] = None,
                    reverse: bool = False,
                    exclusive: bool = False):
    # `flatten` aligns with c++ op
    if flatten:
        assert axis in [0, None]
        axis = None

    x = np.copy(x)

    if axis is None:
        x = x.flatten()
        axis = 0

    if reverse:
        x = np.flip(x, axis)

    dimensions = [range(dim) for dim in x.shape[:axis]]

    if exclusive:
        x = np.roll(x, 1, axis)
        for prefix_dim in itertools.product(*dimensions):
            x[prefix_dim][0] = np.finfo(x.dtype).min

    for prefix_dim in itertools.product(*dimensions):
        arr = x[prefix_dim]
        for dim in range(1, arr.shape[0]):
            arr[dim] = np.logaddexp(arr[dim - 1], arr[dim])

    if reverse:
        x = np.flip(x, axis)

    return x


def np_logcumsumexp_grad(
    x: np.ndarray,
    dout: np.ndarray,
    axis: Optional[int] = None,
    flatten: Optional[bool] = None,
    reverse: bool = False,
    exclusive: bool = False,
):
    out = np_logcumsumexp(x, axis, flatten, reverse, exclusive)
    log_grad_positive = np.where(dout > 0, np.log(dout), np.finfo(x.dtype).min)
    log_grad_negative = np.where(dout < 0, np.log(-dout), np.finfo(x.dtype).min)

    output_pos = np.exp(
        np_logcumsumexp(log_grad_positive - out,
                        axis=axis,
                        flatten=flatten,
                        reverse=not reverse,
                        exclusive=exclusive).reshape(x.shape) + x)
    output_neg = np.exp(
        np_logcumsumexp(log_grad_negative - out,
                        axis=axis,
                        flatten=flatten,
                        reverse=not reverse,
                        exclusive=exclusive).reshape(x.shape) + x)

    return output_pos - output_neg


class TestLogcumsumexp(unittest.TestCase):

    def run_imperative(self):
        data_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        data = paddle.to_tensor(data_np)

        y = paddle.logcumsumexp(data)
        z = np_logcumsumexp(data_np)
        np.testing.assert_allclose(z, y.numpy(), rtol=1e-05)

        y = paddle.logcumsumexp(data, axis=0)
        z = np_logcumsumexp(data_np, axis=0)
        np.testing.assert_allclose(z, y.numpy(), rtol=1e-05)

        y = paddle.logcumsumexp(data, axis=-1)
        z = np_logcumsumexp(data_np, axis=-1)
        np.testing.assert_allclose(z, y.numpy(), rtol=1e-05)

        y = paddle.logcumsumexp(data, dtype='float32')
        self.assertTrue(y.dtype == core.VarDesc.VarType.FP32)

        y = paddle.logcumsumexp(data, axis=-2)
        z = np_logcumsumexp(data_np, axis=-2)
        np.testing.assert_allclose(z, y.numpy(), rtol=1e-05)

        with self.assertRaises(IndexError):
            y = paddle.logcumsumexp(data, axis=-3)

        with self.assertRaises(IndexError):
            y = paddle.logcumsumexp(data, axis=2)

        data_np = np.arange(10000, 10024, dtype=np.float32)
        data = paddle.to_tensor(data_np)
        y = paddle.logcumsumexp(data)
        z = np_naive_logcumsumexp(data_np)
        # check that naive algorithm overflows
        self.assertTrue(all(z == np.inf))
        z = np_logcumsumexp(data_np)
        # check that our algorithm doesn't overflow
        self.assertTrue(all(z != np.inf))
        np.testing.assert_allclose(z, y.numpy(), rtol=1e-05)

    def run_static(self, use_gpu=False):
        with fluid.program_guard(fluid.Program()):
            data_np = np.random.random((5, 4)).astype(np.float32)
            x = paddle.static.data('X', [5, 4])
            y = paddle.logcumsumexp(x)
            y2 = paddle.logcumsumexp(x, axis=0)
            y3 = paddle.logcumsumexp(x, axis=-1)
            y4 = paddle.logcumsumexp(x, dtype='float64')
            y5 = paddle.logcumsumexp(x, axis=-2)

            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            out = exe.run(feed={'X': data_np},
                          fetch_list=[
                              y.name,
                              y2.name,
                              y3.name,
                              y4.name,
                              y5.name,
                          ])

            z = np_logcumsumexp(data_np)
            np.testing.assert_allclose(z, out[0], rtol=1e-05)
            z = np_logcumsumexp(data_np, axis=0)
            np.testing.assert_allclose(z, out[1], rtol=1e-05)
            z = np_logcumsumexp(data_np, axis=-1)
            np.testing.assert_allclose(z, out[2], rtol=1e-05)
            self.assertTrue(out[3].dtype == np.float64)
            z = np_logcumsumexp(data_np, axis=-2)
            np.testing.assert_allclose(z, out[4], rtol=1e-05)

    def test_cpu(self):
        paddle.disable_static(paddle.fluid.CPUPlace())
        self.run_imperative()
        paddle.enable_static()

        self.run_static()

    def test_gpu(self):
        if not fluid.core.is_compiled_with_cuda():
            return
        paddle.disable_static(paddle.fluid.CUDAPlace(0))
        self.run_imperative()
        paddle.enable_static()

        self.run_static(use_gpu=True)

    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = paddle.static.data('x', [3, 4])
            y = paddle.logcumsumexp(x, name='out')
            self.assertTrue('out' in y.name)

    def test_type_error(self):
        with fluid.program_guard(fluid.Program()):

            with self.assertRaises(TypeError):
                data_np = np.random.random((100, 100), dtype=np.int32)
                x = paddle.static.data('X', [100, 100], dtype='int32')
                y = paddle.logcumsumexp(x)

                place = fluid.CUDAPlace(0)
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                out = exe.run(feed={'X': data_np}, fetch_list=[y.name])


class BaseTestCases:

    class BaseOpTest(OpTest):

        def setUp(self):
            self.op_type = "logcumsumexp"
            input, attrs = self.input_and_attrs()
            self.inputs = {'X': input}
            self.attrs = attrs
            self.outputs = {'Out': np_logcumsumexp(input, **attrs)}

        def test_check_output(self):
            self.check_output()

        def test_check_grad(self):
            self.check_grad(['X'],
                            'Out',
                            user_defined_grads=[
                                np_logcumsumexp_grad(self.inputs['X'],
                                                     1 / self.inputs['X'].size,
                                                     **self.attrs)
                            ])

        def input_and_attrs(self):
            raise NotImplementedError()


class TestLogcumsumexpOp1(BaseTestCases.BaseOpTest):

    def input_and_attrs(self):
        return np.arange(100, dtype=np.float64).reshape(10, 10), {
            'axis': 0,
            'flatten': True,
            'reverse': True
        }


class TestLogcumsumexpOp2(BaseTestCases.BaseOpTest):

    def input_and_attrs(self):
        return np.arange(100, dtype=np.float64).reshape(10, 10), {
            'axis': 1,
            'reverse': True
        }


class TestLogcumsumexpOp3(BaseTestCases.BaseOpTest):

    def input_and_attrs(self):
        return np.arange(100, dtype=np.float64).reshape(10, 10), {'axis': 1}


class TestLogcumsumexpOp4(BaseTestCases.BaseOpTest):

    def input_and_attrs(self):
        return np.arange(100, dtype=np.float64).reshape(10, 10), {
            'axis': 0,
            'flatten': True,
            'reverse': True,
            'exclusive': True
        }


if __name__ == '__main__':
    unittest.main()
