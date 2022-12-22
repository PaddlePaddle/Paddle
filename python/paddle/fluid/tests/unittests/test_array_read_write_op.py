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

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from paddle.fluid import Program, program_guard
from paddle.fluid.backward import append_backward
from paddle.fluid.executor import Executor
from paddle.fluid.framework import default_main_program


def _test_read_write(x):
    i = layers.zeros(shape=[1], dtype='int64')
    i.stop_gradient = False
    arr = paddle.tensor.array_write(x=x[0], i=i)
    i = paddle.increment(x=i)
    arr = paddle.tensor.array_write(x=x[1], i=i, array=arr)
    i = paddle.increment(x=i)
    arr = paddle.tensor.array_write(x=x[2], i=i, array=arr)

    i = layers.zeros(shape=[1], dtype='int64')
    i.stop_gradient = False
    a0 = paddle.tensor.array_read(array=arr, i=i)
    i = paddle.increment(x=i)
    a1 = paddle.tensor.array_read(array=arr, i=i)
    i = paddle.increment(x=i)
    a2 = paddle.tensor.array_read(array=arr, i=i)

    mean_a0 = paddle.mean(a0)
    mean_a1 = paddle.mean(a1)
    mean_a2 = paddle.mean(a2)

    a_sum = layers.sums(input=[mean_a0, mean_a1, mean_a2])

    mean_x0 = paddle.mean(x[0])
    mean_x1 = paddle.mean(x[1])
    mean_x2 = paddle.mean(x[2])

    x_sum = layers.sums(input=[mean_x0, mean_x1, mean_x2])

    return a_sum, x_sum


class TestArrayReadWrite(unittest.TestCase):
    def test_read_write(self):
        paddle.enable_static()
        x = [
            layers.data(name='x0', shape=[100]),
            layers.data(name='x1', shape=[100]),
            layers.data(name='x2', shape=[100]),
        ]
        for each_x in x:
            each_x.stop_gradient = False

        tensor = np.random.random(size=(100, 100)).astype('float32')
        a_sum, x_sum = _test_read_write(x)

        place = core.CPUPlace()
        exe = Executor(place)
        outs = exe.run(
            feed={'x0': tensor, 'x1': tensor, 'x2': tensor},
            fetch_list=[a_sum, x_sum],
            scope=core.Scope(),
        )
        self.assertEqual(outs[0], outs[1])

        total_sum = layers.sums(input=[a_sum, x_sum])
        total_sum_scaled = paddle.scale(x=total_sum, scale=1 / 6.0)

        append_backward(total_sum_scaled)

        g_vars = list(
            map(
                default_main_program().global_block().var,
                [each_x.name + "@GRAD" for each_x in x],
            )
        )
        g_out = [
            item.sum()
            for item in exe.run(
                feed={'x0': tensor, 'x1': tensor, 'x2': tensor},
                fetch_list=g_vars,
            )
        ]
        g_out_sum = np.array(g_out).sum()

        # since our final gradient is 1 and the neural network are all linear
        # with mean_op.
        # the input gradient should also be 1
        self.assertAlmostEqual(1.0, g_out_sum, delta=0.1)

        with fluid.dygraph.guard(place):
            tensor1 = fluid.dygraph.to_variable(tensor)
            tensor2 = fluid.dygraph.to_variable(tensor)
            tensor3 = fluid.dygraph.to_variable(tensor)
            x_dygraph = [tensor1, tensor2, tensor3]
            for each_x in x_dygraph:
                each_x.stop_gradient = False
            a_sum_dygraph, x_sum_dygraph = _test_read_write(x_dygraph)
            self.assertEqual(a_sum_dygraph, x_sum_dygraph)

            total_sum_dygraph = layers.sums(
                input=[a_sum_dygraph, x_sum_dygraph]
            )
            total_sum_scaled_dygraph = paddle.scale(
                x=total_sum_dygraph, scale=1 / 6.0
            )
            total_sum_scaled_dygraph.backward()
            g_out_dygraph = [
                item._grad_ivar().numpy().sum() for item in x_dygraph
            ]
            g_out_sum_dygraph = np.array(g_out_dygraph).sum()

            self.assertAlmostEqual(1.0, g_out_sum_dygraph, delta=0.1)


class TestArrayReadWriteOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            x1 = np.random.randn(2, 4).astype('int32')
            x2 = paddle.ones(shape=[1], dtype='int32')
            x3 = np.random.randn(2, 4).astype('int32')

            self.assertRaises(
                TypeError, paddle.tensor.array_read, array=x1, i=x2
            )
            self.assertRaises(
                TypeError, paddle.tensor.array_write, array=x1, i=x2, out=x3
            )


class TestArrayReadWriteApi(unittest.TestCase):
    def test_api(self):
        paddle.disable_static()
        arr = paddle.tensor.create_array(dtype="float32")
        x = paddle.full(shape=[1, 3], fill_value=5, dtype="float32")
        i = paddle.zeros(shape=[1], dtype="int32")

        arr = paddle.tensor.array_write(x, i, array=arr)

        item = paddle.tensor.array_read(arr, i)

        np.testing.assert_allclose(x.numpy(), item.numpy(), rtol=1e-05)
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
