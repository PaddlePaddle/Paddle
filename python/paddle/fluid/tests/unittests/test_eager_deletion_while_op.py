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
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.executor import Executor
import paddle.fluid.core as core
from paddle.fluid.backward import append_backward
import paddle.fluid.compiler as compiler
import numpy
import multiprocessing

import paddle

paddle.enable_static()
fluid.core._set_eager_deletion_mode(0.0, 1.0, True)


class TestEagerDeletionWhileOpBase(unittest.TestCase):

    def test_main(self):
        places = [
            core.CPUPlace(),
        ]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for p in places:
            for with_data_parallel in [False, True]:
                with fluid.program_guard(fluid.Program(), fluid.Program()):
                    with fluid.scope_guard(fluid.Scope()):
                        self.run_main(p, with_data_parallel)

    def run_main(self, place, with_data_parallel):
        self.place = place
        self.with_data_parallel = with_data_parallel

        if not core.is_compiled_with_cuda() and isinstance(
                self.place, core.CUDAPlace):
            return

        if isinstance(self.place, core.CUDAPlace):
            device_cnt = core.get_cuda_device_count(
            ) if self.with_data_parallel else 1
        else:
            device_cnt = int(
                os.environ.get('CPU_NUM', multiprocessing.cpu_count())
            ) if self.with_data_parallel else 1

        d0 = layers.data("d0",
                         shape=[10],
                         append_batch_size=False,
                         dtype='float32')
        d1 = layers.data("d1",
                         shape=[10],
                         append_batch_size=False,
                         dtype='float32')
        d2 = layers.data("d2",
                         shape=[10],
                         append_batch_size=False,
                         dtype='float32')

        i = layers.zeros(shape=[1], dtype='int64')
        i.stop_gradient = True

        init = layers.zeros(shape=[10], dtype='float32')
        mem_array = layers.array_write(x=init, i=i)
        data_array = layers.array_write(x=d0, i=i)

        i = layers.increment(i)
        layers.array_write(d1, i, array=data_array)

        i = layers.increment(i)
        layers.array_write(d2, i, array=data_array)

        i = layers.zeros(shape=[1], dtype='int64')
        i.stop_gradient = True

        array_len = layers.fill_constant(shape=[1], dtype='int64', value=1)
        array_len.stop_gradient = True
        cond = layers.less_than(x=i, y=array_len)

        j = layers.fill_constant(shape=[1], dtype='int64', value=1)
        j.stop_gradient = True

        array_len2 = layers.fill_constant(shape=[1], dtype='int64', value=3)
        array_len2.stop_gradient = True
        cond2 = layers.less_than(x=j, y=array_len2)

        while_op = layers.While(cond=cond)
        while_op2 = layers.While(cond=cond2)
        with while_op.block():
            d = layers.array_read(array=data_array, i=i)
            prev = layers.array_read(array=mem_array, i=i)
            d = layers.reshape(d, shape=[10])
            prev = layers.reshape(prev, shape=[10])
            result = layers.sums(input=[d, prev])

            i = layers.increment(x=i, in_place=True)
            layers.array_write(result, i=i, array=mem_array)
            layers.less_than(x=i, y=array_len, cond=cond)
            with while_op2.block():
                d2 = layers.array_read(array=data_array, i=j)
                prev2 = layers.array_read(array=mem_array, i=j)
                d2 = layers.reshape(d2, shape=[10])
                prev2 = layers.reshape(prev2, shape=[10])
                result2 = layers.sums(input=[d2, prev2])

                j = layers.increment(x=j, in_place=True)
                layers.array_write(result2, i=j, array=mem_array)
                layers.less_than(x=j, y=array_len2, cond=cond2)

        sum_result = layers.array_read(array=mem_array, i=j)
        sum_result.persistable = True
        tmp = layers.unsqueeze(sum_result, axes=[0])
        tmp = layers.expand(tmp, expand_times=[10, 1])
        fc = layers.fc(tmp, size=256)
        loss = paddle.mean(sum_result)

        optim = fluid.optimizer.Adam(learning_rate=1e-3)
        optim.minimize(loss)

        gc_vars = core._get_eager_deletion_vars(
            fluid.default_main_program().desc, [loss.name])
        self.assertEqual(len(gc_vars), 5)

        exe = Executor(self.place)
        exe.run(fluid.default_startup_program())

        prog = fluid.default_main_program()
        if self.with_data_parallel:
            prog = compiler.CompiledProgram(
                fluid.default_main_program()).with_data_parallel(
                    loss_name=loss.name)

        for _ in range(5):
            d = []
            for i in range(3):
                tmp = numpy.random.random(size=[10]).astype('float32')
                if not self.with_data_parallel:
                    d.append(tmp)
                else:
                    d.append(numpy.array([tmp] * device_cnt))

            outs = exe.run(program=prog,
                           feed={
                               'd0': d[0],
                               'd1': d[1],
                               'd2': d[2]
                           },
                           fetch_list=[sum_result])
            self.assertAlmostEqual(numpy.sum(d), numpy.sum(outs[0]), delta=0.01)


if __name__ == '__main__':
    unittest.main()
