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

import os

os.environ['CPU_NUM'] = '2'

import unittest

import numpy

import paddle
from paddle import base
from paddle.base import core, in_pir_mode
from paddle.base.executor import Executor

paddle.enable_static()
base.core._set_eager_deletion_mode(0.0, 1.0, True)


class TestEagerDeletionWhileOpBase(unittest.TestCase):

    def test_main(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for p in places:
            with base.program_guard(base.Program(), base.Program()):
                with base.scope_guard(base.Scope()):
                    self.run_main(p)

    def run_main(self, place):
        self.place = place

        if not core.is_compiled_with_cuda() and isinstance(
            self.place, core.CUDAPlace
        ):
            return

        device_cnt = 1

        d0 = paddle.static.data("d0", shape=[-1, 10], dtype='float32')
        d1 = paddle.static.data("d1", shape=[-1, 10], dtype='float32')
        d2 = paddle.static.data("d2", shape=[-1, 10], dtype='float32')

        i = paddle.zeros(shape=[1], dtype='int64')
        i.stop_gradient = True

        init = paddle.zeros(shape=[10], dtype='float32')
        mem_array = paddle.tensor.array_write(x=init, i=i)
        data_array = paddle.tensor.array_write(x=d0, i=i)

        i = paddle.increment(i)
        paddle.tensor.array_write(d1, i, array=data_array)

        i = paddle.increment(i)
        paddle.tensor.array_write(d2, i, array=data_array)

        i = paddle.zeros(shape=[1], dtype='int64')
        i.stop_gradient = True

        array_len = paddle.tensor.fill_constant(
            shape=[1], dtype='int64', value=1
        )
        array_len.stop_gradient = True
        cond = paddle.less_than(x=i, y=array_len)

        j = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=1)
        j.stop_gradient = True

        array_len2 = paddle.tensor.fill_constant(
            shape=[1], dtype='int64', value=3
        )
        array_len2.stop_gradient = True
        cond2 = paddle.less_than(x=j, y=array_len2)

        while_op = paddle.static.nn.control_flow.While(cond=cond)
        while_op2 = paddle.static.nn.control_flow.While(cond=cond2)
        with while_op.block():
            d = paddle.tensor.array_read(array=data_array, i=i)
            prev = paddle.tensor.array_read(array=mem_array, i=i)
            d = paddle.reshape(d, shape=[10])
            prev = paddle.reshape(prev, shape=[10])
            result = paddle.add_n([d, prev])

            i = paddle.increment(x=i)
            paddle.tensor.array_write(result, i=i, array=mem_array)
            paddle.assign(paddle.less_than(x=i, y=array_len), cond)
            with while_op2.block():
                d2 = paddle.tensor.array_read(array=data_array, i=j)
                prev2 = paddle.tensor.array_read(array=mem_array, i=j)
                d2 = paddle.reshape(d2, shape=[10])
                prev2 = paddle.reshape(prev2, shape=[10])
                result2 = paddle.add_n([d2, prev2])

                j = paddle.increment(x=j)
                paddle.tensor.array_write(result2, i=j, array=mem_array)
                paddle.assign(paddle.less_than(x=j, y=array_len2), cond2)

        sum_result = paddle.tensor.array_read(array=mem_array, i=j)
        sum_result.persistable = True
        tmp = paddle.unsqueeze(sum_result, axis=[0])
        tmp = paddle.expand(tmp, [10, -1])
        loss = paddle.mean(sum_result)

        optim = paddle.optimizer.Adam(learning_rate=1e-3)
        optim.minimize(loss)

        if not in_pir_mode():
            gc_vars = core._get_eager_deletion_vars(
                base.default_main_program().desc, [loss.name]
            )
            self.assertEqual(len(gc_vars), 3)

        exe = Executor(self.place)
        exe.run(paddle.static.default_startup_program())

        prog = paddle.static.default_main_program()

        for _ in range(5):
            d = []
            for i in range(3):
                tmp = numpy.random.random(size=[10]).astype('float32')
                d.append(numpy.array([tmp] * device_cnt))

            outs = exe.run(
                program=prog,
                feed={'d0': d[0], 'd1': d[1], 'd2': d[2]},
                fetch_list=[sum_result],
            )
            self.assertAlmostEqual(numpy.sum(d), numpy.sum(outs[0]), delta=0.01)


if __name__ == '__main__':
    unittest.main()
