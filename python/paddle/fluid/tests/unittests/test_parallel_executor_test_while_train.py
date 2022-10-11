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

from simple_nets import simple_fc_net
import paddle.fluid as fluid
from paddle.fluid import compiler
import paddle.fluid.core as core
import numpy as np
import unittest
import os
import sys
import math


class ParallelExecutorTestingDuringTraining(unittest.TestCase):

    def check_network_convergence(self, use_cuda, build_strategy=None):
        os.environ['CPU_NUM'] = str(4)
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = simple_fc_net()
            test_program = main.clone(for_test=True)

            opt = fluid.optimizer.SGD(learning_rate=0.001)
            opt.minimize(loss)

            batch_size = 32
            image = np.random.normal(size=(batch_size, 784)).astype('float32')
            label = np.random.randint(0, 10, (batch_size, 1), dtype="int64")

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup)
            feed_dict = {'image': image, 'label': label}

            train_cp = compiler.CompiledProgram(main).with_data_parallel(
                loss_name=loss.name, build_strategy=build_strategy)
            test_cp = compiler.CompiledProgram(test_program).with_data_parallel(
                loss_name=loss.name,
                build_strategy=build_strategy,
                share_vars_from=train_cp)

            for i in range(5):
                exe.run(train_cp, feed=feed_dict, fetch_list=[loss.name])
                test_loss, = exe.run(test_cp,
                                     feed=feed_dict,
                                     fetch_list=[loss.name])
                train_loss, = exe.run(train_cp,
                                      feed=feed_dict,
                                      fetch_list=[loss.name])

                avg_test_loss_val = np.array(test_loss).mean()
                if math.isnan(float(avg_test_loss_val)):
                    sys.exit("got NaN loss, testing failed.")

                avg_train_loss_val = np.array(train_loss).mean()
                if math.isnan(float(avg_train_loss_val)):
                    sys.exit("got NaN loss, training failed.")

                np.testing.assert_allclose(train_loss,
                                           test_loss,
                                           rtol=1e-05,
                                           atol=0.01)

    def test_parallel_testing(self):
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce
        if core.is_compiled_with_cuda():
            self.check_network_convergence(use_cuda=True,
                                           build_strategy=build_strategy)
        self.check_network_convergence(use_cuda=False,
                                       build_strategy=build_strategy)

    def test_parallel_testing_with_new_strategy_gpu(self):
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        if core.is_compiled_with_cuda():
            self.check_network_convergence(use_cuda=True,
                                           build_strategy=build_strategy)

    def test_parallel_testing_with_new_strategy_cpu(self):
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        self.check_network_convergence(use_cuda=False,
                                       build_strategy=build_strategy)


if __name__ == '__main__':
    unittest.main()
