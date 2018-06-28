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

import paddle.fluid as fluid
import numpy as np
import unittest
import os


def simple_fc_net():
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = img
    for _ in xrange(4):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)))
    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss


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

            train_exe = fluid.ParallelExecutor(
                use_cuda=use_cuda,
                loss_name=loss.name,
                main_program=main,
                build_strategy=build_strategy)

            test_exe = fluid.ParallelExecutor(
                use_cuda=use_cuda,
                main_program=test_program,
                share_vars_from=train_exe,
                build_strategy=build_strategy)

            for i in xrange(5):
                test_loss, = test_exe.run([loss.name], feed=feed_dict)

                train_loss, = train_exe.run([loss.name], feed=feed_dict)

                self.assertTrue(
                    np.allclose(
                        train_loss, test_loss, atol=1e-8),
                    "Train loss: " + str(train_loss) + "\n Test loss:" +
                    str(test_loss))

    def test_parallel_testing(self):
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce
        self.check_network_convergence(
            use_cuda=True, build_strategy=build_strategy)
        self.check_network_convergence(
            use_cuda=False, build_strategy=build_strategy)

    def test_parallel_testing_with_new_strategy(self):
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        self.check_network_convergence(
            use_cuda=True, build_strategy=build_strategy)
        self.check_network_convergence(
            use_cuda=False, build_strategy=build_strategy)


if __name__ == '__main__':
    unittest.main()
