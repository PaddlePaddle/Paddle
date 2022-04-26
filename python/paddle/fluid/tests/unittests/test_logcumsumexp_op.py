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

from __future__ import print_function

from typing import Optional
import unittest
import itertools
import numpy as np
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.framework import _test_eager_guard


def np_logcumsumexp(x: np.ndarray, axis: Optional[int]=None, reverse: bool=False, exclusive: bool=False):
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


class TestLogcumsumexpOp(unittest.TestCase):
    def run_cases(self):
        data_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        data = paddle.to_tensor(data_np)

        y = paddle.logcumsumexp(data)
        z = np_logcumsumexp(data_np)
        self.assertTrue(np.allclose(z, y.numpy()))

        y = paddle.logcumsumexp(data, axis=0)
        z = np_logcumsumexp(data_np, axis=0)
        self.assertTrue(np.allclose(z, y.numpy()))

        y = paddle.logcumsumexp(data, axis=-1)
        z = np_logcumsumexp(data_np, axis=-1)
        self.assertTrue(np.allclose(z, y.numpy()))

        y = paddle.logcumsumexp(data, dtype='float64')
        self.assertTrue(y.dtype == core.VarDesc.VarType.FP64)

        y = paddle.logcumsumexp(data, axis=-2)
        z = np_logcumsumexp(data_np, axis=-2)
        self.assertTrue(np.allclose(z, y.numpy()))

        with self.assertRaises(IndexError):
            y = paddle.logcumsumexp(data, axis=-3)

        with self.assertRaises(IndexError):
            y = paddle.logcumsumexp(data, axis=2)

    def run_static(self, use_gpu=False):
        with fluid.program_guard(fluid.Program()):
            data_np = np.random.random((100, 100)).astype(np.float32)
            x = paddle.static.data('X', [100, 100])
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
                              y.name, y2.name, y3.name, y4.name, y5.name,
                          ])

            z = np_logcumsumexp(data_np)
            self.assertTrue(np.allclose(z, out[0]))
            z = np_logcumsumexp(data_np, axis=0)
            self.assertTrue(np.allclose(z, out[1]))
            z = np_logcumsumexp(data_np, axis=-1)
            self.assertTrue(np.allclose(z, out[2]))
            self.assertTrue(out[3].dtype == np.float64)
            z = np_logcumsumexp(data_np, axis=-2)
            self.assertTrue(np.allclose(z, out[4]))

    def test_cpu(self):
        paddle.disable_static(paddle.fluid.CPUPlace())
        self.run_cases()
        paddle.enable_static()

        self.run_static()

    def test_gpu(self):
        if not fluid.core.is_compiled_with_cuda():
            return
        paddle.disable_static(paddle.fluid.CUDAPlace(0))
        self.run_cases()
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
                out = exe.run(feed={'X': data_np},
                              fetch_list=[
                                  y.name
                              ])


if __name__ == '__main__':
    unittest.main()
