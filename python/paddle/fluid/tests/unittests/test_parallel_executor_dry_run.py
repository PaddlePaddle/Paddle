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

import logging
import os
import unittest

import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler

os.environ['CPU_NUM'] = str(4)


class TestBase(unittest.TestCase):
    def main(
        self,
        network_func,
        iter=10,
        iter_per_pe=10,
        use_gpu=True,
        use_experimental_executor=False,
    ):
        if use_gpu and not fluid.core.is_compiled_with_cuda():
            logging.warning(
                "Paddle is not compiled with CUDA, skip GPU unittests"
            )
            return

        main_prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.Scope()
        with fluid.program_guard(main_prog, startup_prog):
            with fluid.scope_guard(scope):
                loss = network_func()
                exe = fluid.Executor(
                    fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
                )
                exe.run(startup_prog)

                exe_strategy = fluid.ExecutionStrategy()
                exe_strategy._dry_run = True
                exe_strategy.use_experimental_executor = (
                    use_experimental_executor
                )
                train_cp = compiler.CompiledProgram(
                    main_prog
                ).with_data_parallel(
                    loss_name=loss.name, exec_strategy=exe_strategy
                )
                for _ in range(iter):
                    for _ in range(iter_per_pe):
                        exe.run(train_cp)


class TestMNISTDryRun(TestBase):
    def test_mnist_dry_run(self):
        for use_gpu in (False, True):
            for use_experimental_executor in (False, True):
                self.main(
                    network_func=TestMNISTDryRun.network_func,
                    use_gpu=use_gpu,
                    use_experimental_executor=use_experimental_executor,
                )

    @staticmethod
    def network_func():
        img = fluid.layers.data(name='img', shape=[784], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        hidden = img
        for _ in range(10):
            hidden = fluid.layers.fc(input=img, size=200, act='tanh')
        prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
        loss = paddle.nn.functional.cross_entropy(
            input=prediction, label=label, reduction='none', use_softmax=False
        )
        avg_loss = paddle.mean(loss)
        fluid.optimizer.Adam().minimize(avg_loss)
        return avg_loss


if __name__ == '__main__':
    unittest.main()
