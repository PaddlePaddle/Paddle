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


class TestFusedParallelExecutor(unittest.TestCase):
    def test_fused_all_reduce(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            img = fluid.layers.fill_constant(
                shape=[64, 784], dtype='float', value=0)
            lbl = fluid.layers.fill_constant(
                shape=[64, 1], dtype='int64', value=1)
            hidden = fluid.layers.fc(input=img, size=10, act='softmax')
            loss = fluid.layers.cross_entropy(input=hidden, label=lbl)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(learning_rate=1e-3)
            sgd.minimize(loss)

        exe = fluid.Executor(fluid.CUDAPlace(0))
        exe.run(startup)
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.FusedAllReduce
        pe = fluid.ParallelExecutor(
            use_cuda=True,
            loss_name=loss.name,
            build_strategy=build_strategy,
            main_program=main)
        pe.run(fetch_list=[])


if __name__ == '__main__':
    unittest.main()
