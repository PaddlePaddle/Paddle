# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest

import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, program_guard


class TestWhileLoop(unittest.TestCase):
    def test_simple_net(self):
        def cond(i):
            return layers.less_than(i, ten)

        def body(i):
            return [layers.increment(i)]

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = layers.fill_constant(shape=[1], dtype='int64', value=0)
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)
            out = layers.while_loop(cond, body, [i])

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(main_program, fetch_list=out)
        self.assertTrue(
            np.allclose(np.asarray(res[0]), np.full((1), 10, np.int64)))

    def test_simple_net2(self):
        def cond1(i, j, init, sums):
            return layers.less_than(i, loop_len1)

        def body1(i, j, init, sums):
            def cond2(j, init, sums):
                return layers.less_than(j, loop_len2)

            def body2(j, init, sums):
                init = layers.elementwise_add(x=init, y=ones)
                sums = layers.elementwise_add(x=init, y=sums)
                j = layers.increment(j)
                return [j, init, sums]

            result = layers.while_loop(cond2, body2, [j, init, sums])
            j = result[0]
            init = result[1]
            sums = result[2]
            sums = layers.elementwise_add(x=init, y=sums)
            i = layers.increment(i)
            return [i, j, init, sums]

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = layers.zeros(shape=[1], dtype='int64')
            j = layers.zeros(shape=[1], dtype='int64')
            init = layers.data(name="init", shape=[3, 3], dtype='float32')
            sums = layers.data(name="sums", shape=[3, 3], dtype='float32')
            loop_len1 = layers.fill_constant(shape=[1], dtype='int64', value=2)
            loop_len2 = layers.fill_constant(shape=[1], dtype='int64', value=3)
            ones = layers.fill_constant(shape=[3, 3], dtype='float32', value=1)

            res = layers.while_loop(cond1, body1, [i, j, init, sums])
            #layers.Print(res[0])
            #layers.Print(res[1])
            #layers.Print(res[2])
            #layers.Print(res[3])
            data1 = np.random.rand(3, 3).astype('float32')
            data2 = np.zeros([3, 3]).astype('float32')

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        ret = exe.run(main_program,
                      feed={'init': data1,
                            'sums': data2},
                      fetch_list=res)
        for i in range(3):
            data1 = np.add(data1, 1)
            data2 = np.add(data1, data2)
        for j in range(2):
            data2 = np.add(data1, data2)
        self.assertTrue(np.allclose(np.asarray(ret[3]), data2))


if __name__ == '__main__':
    unittest.main()
