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

import math
import paddle.fluid as fluid
from paddle.fluid import compiler
import paddle.fluid.core as core
import unittest
import numpy as np
import os


def model_net(class_dim):
    data = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    conv1 = fluid.layers.conv2d(data, 4, 5, 1, act=None)
    bn1 = fluid.layers.batch_norm(conv1, act='relu')
    pool1 = fluid.layers.pool2d(bn1, 2, 'max', 2)
    conv2 = fluid.layers.conv2d(pool1, 16, 5, 1, act=None)
    bn2 = fluid.layers.batch_norm(conv2, act='relu')
    pool2 = fluid.layers.pool2d(bn2, 2, 'max', 2)

    fc1 = fluid.layers.fc(pool2, size=50, act='relu')
    fc2 = fluid.layers.fc(fc1, size=class_dim, act='softmax')

    loss = fluid.layers.cross_entropy(input=fc2, label=label)
    loss = fluid.layers.mean(loss)
    opt = fluid.optimizer.Momentum(
        learning_rate=0.1,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    opt.minimize(loss)

    return data, label, loss, [conv1, bn1, pool1, conv2, bn2, pool2, fc1, fc2]


class TestFetchAndFeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    def run_tests(self,
                  use_cuda,
                  run_parallel_exe,
                  use_experimental_executor=False,
                  seed=1):
        img_label = self.gen_data()
        #print('image_label : {0}'.format(img_label))

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

        main_program = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = 1
        main_program.random_seed = 1
        with fluid.program_guard(main_program, startup):
            data, label, loss, fetch_list = model_net(class_dim=102)

        exe = fluid.Executor(place)
        exe.run(startup)

        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = use_experimental_executor
        train_cp = compiler.CompiledProgram(main_program).with_data_parallel(
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
            places=[place])

        loss_a = run_parallel_exe(img_label, fetch_list[-1:], train_cp, exe,
                                  data, label, loss)[-1]
        nodes_num_a = len(train_cp._graph.nodes())

        exe.run(startup)
        loss_b = run_parallel_exe(img_label, fetch_list[-1:], train_cp, exe,
                                  data, label, loss)[-1]
        nodes_num_b = len(train_cp._graph.nodes())

        exe.run(startup)
        loss_c = run_parallel_exe(img_label, fetch_list[:], train_cp, exe, data,
                                  label, loss)[-1]
        nodes_num_c = len(train_cp._graph.nodes())

        self.assertEqual(nodes_num_a, nodes_num_b)
        self.assertNotEqual(nodes_num_b, nodes_num_c)

        self.assertTrue(np.allclose(loss_a, loss_b, atol=1e-8))
        self.assertTrue(np.allclose(loss_b, loss_c, atol=1e-8))

    def run_parallel_exe_with_fetch(self, img_label, fetch_list,
                                    compiled_program, exe, data, label, loss):

        img, l = img_label
        train_inputs = {data.name: img, label.name: l}
        ret = exe.run(compiled_program,
                      fetch_list=fetch_list,
                      feed=train_inputs,
                      return_numpy=True)
        for i in range(len(fetch_list)):
            assert not math.isnan(np.sum(ret[i])) and \
                   not math.isinf(np.sum(ret[i]))
        return ret

    def gen_data(self, batch_size=8):
        np.random.seed(5)
        img = np.random.random(
            size=[batch_size, 3, 224, 224]).astype(np.float32)
        l = (np.random.random(size=[batch_size, 1]) * 10).astype(np.int64)
        return img, l

    def test_recompile_gpu(self):
        self.run_tests(
            use_cuda=True, run_parallel_exe=self.run_parallel_exe_with_fetch)

    def test_recompile_cpu(self):
        self.run_tests(
            use_cuda=False, run_parallel_exe=self.run_parallel_exe_with_fetch)


if __name__ == '__main__':
    unittest.main()
