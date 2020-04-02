# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.tests.unittests.simple_nets import simple_fc_net
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler
import numpy as np
import unittest
import os
import sys
import math


class TestPallelExecutorNgraph(unittest.TestCase):
    def check_network_convergence(self, build_strategy=None):
        os.environ['CPU_NUM'] = str(2)
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = simple_fc_net()
            test_program = main.clone(for_test=True)

            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)

            batch_size = 32
            image = np.random.normal(size=(batch_size, 784)).astype('float32')
            label = np.random.randint(0, 10, (batch_size, 1), dtype="int64")

            place = fluid.CPUPlace()
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
                _ = exe.run(train_cp, fetch_list=[loss.name], feed=feed_dict)
                test_loss, = exe.run(test_cp,
                                     fetch_list=[loss.name],
                                     feed=feed_dict)
                train_loss = exe.run(train_cp,
                                     fetch_list=[loss.name],
                                     feed=feed_dict)

                avg_test_loss_val = np.array(test_loss).mean()
                if math.isnan(float(avg_test_loss_val)):
                    sys.exit("got NaN loss, testing failed.")

                avg_train_loss_val = np.array(train_loss).mean()
                if math.isnan(float(avg_train_loss_val)):
                    sys.exit("got NaN loss, training failed.")

                self.assertTrue(
                    np.allclose(
                        train_loss, test_loss, atol=1e-8),
                    "Train loss: " + str(train_loss) + "\n Test loss:" +
                    str(test_loss))

    def test_parallel_testing(self):
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = False
        build_strategy.memory_optimize = False
        self.check_network_convergence(build_strategy=build_strategy)


if __name__ == '__main__':
    unittest.main()
