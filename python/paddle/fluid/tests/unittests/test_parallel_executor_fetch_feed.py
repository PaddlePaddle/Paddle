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

import math
import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler
import paddle.fluid.core as core
import unittest
import numpy as np
import os


def Lenet(data, class_dim):
    conv1 = fluid.layers.conv2d(data, 4, 5, 1, act=None)
    bn1 = fluid.layers.batch_norm(conv1, act='relu')
    pool1 = fluid.layers.pool2d(bn1, 2, 'max', 2)
    conv2 = fluid.layers.conv2d(pool1, 16, 5, 1, act=None)
    bn2 = fluid.layers.batch_norm(conv2, act='relu')
    pool2 = fluid.layers.pool2d(bn2, 2, 'max', 2)

    fc1 = fluid.layers.fc(pool2, size=50, act='relu')
    fc2 = fluid.layers.fc(fc1, size=class_dim, act='softmax')

    return fc2


class TestFetchAndFeed(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    def parallel_exe(self,
                     use_cuda,
                     run_parallel_exe,
                     use_faster_executor=False,
                     num_threads=4,
                     seed=1):
        main_program = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = seed
        with fluid.program_guard(main_program, startup):
            data = fluid.layers.data(name='image',
                                     shape=[3, 224, 224],
                                     dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            out = Lenet(data, class_dim=102)
            loss = fluid.layers.cross_entropy(input=out, label=label)
            loss = paddle.mean(loss)
            opt = fluid.optimizer.Momentum(
                learning_rate=0.1,
                momentum=0.9,
                regularization=fluid.regularizer.L2Decay(1e-4))
            opt.minimize(loss)

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup)

        #FIXME force disable enable_inplace and memory_optimize to pass the unittest
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = False
        build_strategy.memory_optimize = False
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = use_faster_executor
        exec_strategy.num_threads = num_threads
        train_cp = compiler.CompiledProgram(main_program).with_data_parallel(
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

        run_parallel_exe(train_cp, exe, use_cuda, data, label, loss)

    def run_parallel_exe_with_fetch(self, compiled_program, exe, use_cuda, data,
                                    label, loss):

        def get_data(batch_size=8):
            np.random.seed(5)
            while True:
                img = np.random.random(size=[batch_size, 3, 224, 224]).astype(
                    np.float32)
                l = (np.random.random(size=[batch_size, 1]) * 10).astype(
                    np.int64)
                yield img, l

        fetch_list = []
        all_vars = compiled_program._program.global_block().vars

        for k, v in all_vars.items():
            if ('tmp' not in k) and (
                    k[0] is not '_' or v.persistable
            ) and v.type == core.VarDesc.VarType.LOD_TENSOR:
                fetch_list.append(k)

        for batch_id, img_label in enumerate(get_data()):
            img, l = img_label
            train_inputs = {data.name: img, label.name: l}
            ret = exe.run(compiled_program,
                          fetch_list=fetch_list,
                          feed=train_inputs,
                          return_numpy=True)
            for i in range(len(fetch_list)):
                assert not math.isnan(np.sum(ret[i])) and \
                       not math.isinf(np.sum(ret[i]))
            if batch_id == 2:
                break

    def run_parallel_exe_with_feed(self, compiled_program, exe, use_cuda, data,
                                   label, loss):

        def get_data(batch_size=8):
            np.random.seed(5)
            while True:
                train_data = []
                for _ in range(batch_size):
                    img = np.random.random(size=[1, 3, 224, 224]).astype(
                        np.float32)
                    label = (np.random.random(size=[1, 1]) * 10).astype(
                        np.int64)
                    train_data.append([img, label])
                yield train_data

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
        reader = feeder.decorate_reader(get_data, multi_devices=True)

        for batch_id, data in enumerate(reader()):
            loss_np = exe.run(compiled_program,
                              feed=data,
                              fetch_list=[loss.name])[0]
            print(batch_id, loss_np)
            if batch_id == 2:
                break

    def check_executor(self, use_faster_executor=False, num_threads=4):
        if core.is_compiled_with_cuda():
            self.parallel_exe(use_cuda=True,
                              run_parallel_exe=self.run_parallel_exe_with_fetch,
                              use_faster_executor=use_faster_executor,
                              num_threads=num_threads)
        self.parallel_exe(use_cuda=False,
                          run_parallel_exe=self.run_parallel_exe_with_fetch,
                          use_faster_executor=use_faster_executor,
                          num_threads=num_threads)

    def test_fetch(self):
        for use_faster_executor in {True, False}:
            self.check_executor(use_faster_executor=use_faster_executor,
                                num_threads=4)
            self.check_executor(use_faster_executor=use_faster_executor,
                                num_threads=1)

    def test_feed(self):
        if core.is_compiled_with_cuda():
            self.parallel_exe(use_cuda=True,
                              run_parallel_exe=self.run_parallel_exe_with_feed)
        self.parallel_exe(use_cuda=False,
                          run_parallel_exe=self.run_parallel_exe_with_feed)


if __name__ == '__main__':
    unittest.main()
