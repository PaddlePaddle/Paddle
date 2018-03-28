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
import os
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle.fluid.layers as layers
import paddle.fluid.core as core


class TestProfiler(unittest.TestCase):
    def net_profiler(self, state, profile_path='/tmp/profile'):
        enable_if_gpu = state == 'GPU' or state == "All"
        if enable_if_gpu and not core.is_compiled_with_cuda():
            return
        startup_program = fluid.Program()
        main_program = fluid.Program()

        with fluid.program_guard(main_program, startup_program):
            image = fluid.layers.data(name='x', shape=[784], dtype='float32')
            hidden1 = fluid.layers.fc(input=image, size=64, act='relu')
            i = layers.zeros(shape=[1], dtype='int64')
            counter = fluid.layers.zeros(
                shape=[1], dtype='int64', force_cpu=True)
            until = layers.fill_constant([1], dtype='int64', value=10)
            data_arr = layers.array_write(hidden1, i)
            cond = fluid.layers.less_than(x=counter, y=until)
            while_op = fluid.layers.While(cond=cond)
            with while_op.block():
                hidden_n = fluid.layers.fc(input=hidden1, size=64, act='relu')
                layers.array_write(hidden_n, i, data_arr)
                fluid.layers.increment(x=counter, value=1, in_place=True)
                layers.less_than(x=counter, y=until, cond=cond)

            hidden_n = layers.array_read(data_arr, i)
            hidden2 = fluid.layers.fc(input=hidden_n, size=64, act='relu')
            predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')
            label = fluid.layers.data(name='y', shape=[1], dtype='int64')
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(cost)
            batch_size = fluid.layers.create_tensor(dtype='int64')
            batch_acc = fluid.layers.accuracy(
                input=predict, label=label, total=batch_size)

        optimizer = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        opts = optimizer.minimize(avg_cost, startup_program=startup_program)

        place = fluid.CPUPlace() if state == 'CPU' else fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(startup_program)

        pass_acc_calculator = fluid.average.WeightedAverage()
        with profiler.profiler(state, 'total', profile_path) as prof:
            for iter in range(10):
                if iter == 2:
                    profiler.reset_profiler()
                x = np.random.random((32, 784)).astype("float32")
                y = np.random.randint(0, 10, (32, 1)).astype("int64")

                outs = exe.run(main_program,
                               feed={'x': x,
                                     'y': y},
                               fetch_list=[avg_cost, batch_acc, batch_size])
                acc = np.array(outs[1])
                b_size = np.array(outs[2])
                pass_acc_calculator.add(value=acc, weight=b_size)
                pass_acc = pass_acc_calculator.eval()

    def test_cpu_profiler(self):
        self.net_profiler('CPU')

    def test_cuda_profiler(self):
        self.net_profiler('GPU')

    def test_all_profiler(self):
        self.net_profiler('All', '/tmp/profile_out')
        with open('/tmp/profile_out', 'r') as f:
            self.assertGreater(len(f.read()), 0)


if __name__ == '__main__':
    unittest.main()
