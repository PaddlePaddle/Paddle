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
import os
import tempfile
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import paddle.fluid.proto.profiler.profiler_pb2 as profiler_pb2


class TestProfiler(unittest.TestCase):
    def net_profiler(self, state, use_parallel_executor=False):
        profile_path = os.path.join(tempfile.gettempdir(), "profile")
        open(profile_path, "w").write("")
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
        if use_parallel_executor:
            pe = fluid.ParallelExecutor(
                state != 'CPU',
                loss_name=avg_cost.name,
                main_program=main_program)

        pass_acc_calculator = fluid.average.WeightedAverage()
        with profiler.profiler(state, 'total', profile_path) as prof:
            for iter in range(10):
                if iter == 2:
                    profiler.reset_profiler()
                x = np.random.random((32, 784)).astype("float32")
                y = np.random.randint(0, 10, (32, 1)).astype("int64")

                if use_parallel_executor:
                    pe.run(feed={'x': x, 'y': y}, fetch_list=[avg_cost.name])
                    continue
                outs = exe.run(main_program,
                               feed={'x': x,
                                     'y': y},
                               fetch_list=[avg_cost, batch_acc, batch_size])
                acc = np.array(outs[1])
                b_size = np.array(outs[2])
                pass_acc_calculator.add(value=acc, weight=b_size)
                pass_acc = pass_acc_calculator.eval()
        data = open(profile_path, 'rb').read()
        self.assertGreater(len(data), 0)
        profile_pb = profiler_pb2.Profile()
        profile_pb.ParseFromString(data)
        self.assertGreater(len(profile_pb.events), 0)
        for event in profile_pb.events:
            if event.type == profiler_pb2.Event.GPUKernel:
                if not event.detail_info and not event.name.startswith("MEM"):
                    raise Exception(
                        "Kernel %s missing event. Has this kernel been recorded by RecordEvent?"
                        % event.name)
            elif event.type == profiler_pb2.Event.CPU and (
                    event.name.startswith("Driver API") or
                    event.name.startswith("Runtime API")):
                print("Warning: unregister", event.name)

    def test_cpu_profiler(self):
        self.net_profiler('CPU')
        self.net_profiler('CPU', use_parallel_executor=True)

    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "profiler is enabled only with GPU")
    def test_cuda_profiler(self):
        self.net_profiler('GPU')
        self.net_profiler('GPU', use_parallel_executor=True)

    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "profiler is enabled only with GPU")
    def test_all_profiler(self):
        self.net_profiler('All')
        self.net_profiler('All', use_parallel_executor=True)


if __name__ == '__main__':
    unittest.main()
