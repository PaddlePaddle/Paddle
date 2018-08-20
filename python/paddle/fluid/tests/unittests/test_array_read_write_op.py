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

import unittest
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import default_main_program
import numpy


class TestArrayReadWrite(unittest.TestCase):
    def test_read_write(self):
        x = [
            layers.data(
                name='x0', shape=[100]), layers.data(
                    name='x1', shape=[100]), layers.data(
                        name='x2', shape=[100])
        ]

        for each_x in x:
            each_x.stop_gradient = False

        i = layers.zeros(shape=[1], dtype='int64')
        i.stop_gradient = False
        arr = layers.array_write(x=x[0], i=i)
        i = layers.increment(x=i)
        arr = layers.array_write(x=x[1], i=i, array=arr)
        i = layers.increment(x=i)
        arr = layers.array_write(x=x[2], i=i, array=arr)

        i = layers.zeros(shape=[1], dtype='int64')
        i.stop_gradient = False
        a0 = layers.array_read(array=arr, i=i)
        i = layers.increment(x=i)
        a1 = layers.array_read(array=arr, i=i)
        i = layers.increment(x=i)
        a2 = layers.array_read(array=arr, i=i)

        mean_a0 = layers.mean(a0)
        mean_a1 = layers.mean(a1)
        mean_a2 = layers.mean(a2)

        a_sum = layers.sums(input=[mean_a0, mean_a1, mean_a2])

        mean_x0 = layers.mean(x[0])
        mean_x1 = layers.mean(x[1])
        mean_x2 = layers.mean(x[2])

        x_sum = layers.sums(input=[mean_x0, mean_x1, mean_x2])

        scope = core.Scope()
        cpu = core.CPUPlace()

        exe = Executor(cpu)

        tensor = numpy.random.random(size=(100, 100)).astype('float32')

        outs = exe.run(feed={'x0': tensor,
                             'x1': tensor,
                             'x2': tensor},
                       fetch_list=[a_sum, x_sum],
                       scope=scope)
        self.assertEqual(outs[0], outs[1])

        total_sum = layers.sums(input=[a_sum, x_sum])
        total_sum_scaled = layers.scale(x=total_sum, scale=1 / 6.0)

        append_backward(total_sum_scaled)

        g_vars = list(
            map(default_main_program().global_block().var,
                [each_x.name + "@GRAD" for each_x in x]))
        g_out = [
            item.sum()
            for item in exe.run(
                feed={'x0': tensor,
                      'x1': tensor,
                      'x2': tensor},
                fetch_list=g_vars)
        ]
        g_out_sum = numpy.array(g_out).sum()

        # since our final gradient is 1 and the neural network are all linear
        # with mean_op.
        # the input gradient should also be 1
        self.assertAlmostEqual(1.0, g_out_sum, delta=0.1)


if __name__ == '__main__':
    unittest.main()
