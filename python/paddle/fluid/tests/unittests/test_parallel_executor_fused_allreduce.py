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
import unittest
import numpy


class TestFusedParallelExecutor(unittest.TestCase):
    def run_test(self, fuse_all_reduce):
        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = 1
        with fluid.program_guard(main, startup):
            img = fluid.layers.fill_constant(
                shape=[64, 784], dtype='float32', value=0)

            hidden = img
            for i in xrange(4):
                hidden = fluid.layers.fc(hidden, size=10, act='relu')
            hidden = fluid.layers.fc(input=hidden, size=10, act='softmax')

            lbl = fluid.layers.fill_constant(
                shape=[64, 1], dtype='int64', value=1)
            loss = fluid.layers.cross_entropy(input=hidden, label=lbl)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(learning_rate=1e-3)
            sgd.minimize(loss)

        exe = fluid.Executor(fluid.CUDAPlace(0))
        exe.run(startup)
        build_strategy = fluid.BuildStrategy()
        if fuse_all_reduce:
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.FusedAllReduce
        else:
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce
        pe = fluid.ParallelExecutor(
            use_cuda=True,
            loss_name=loss.name,
            build_strategy=build_strategy,
            main_program=main)
        for i in xrange(10):
            pe.run(fetch_list=[])
        return numpy.array(pe.run(fetch_list=[loss.name])[0])

    def test_main(self):
        loss2 = self.run_test(False)
        loss1 = self.run_test(True)
        assert len(loss1) == len(loss2)
        for l1, l2 in zip(loss1, loss2):
            self.assertAlmostEqual(l1, l2)


if __name__ == '__main__':
    unittest.main()
