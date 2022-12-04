# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid


def enable_parallel_ssa_executor(enabled=True):
    if fluid.is_compiled_with_cuda():
        fluid.core.globals()['FLAGS_enable_parallel_graph'] = enabled


class TestParallelExecutorFetchIsolatedVarBase(unittest.TestCase):
    def build_network(self, is_training):
        x = fluid.data(name='x', shape=[-1, 10], dtype='float32')
        y = fluid.data(name='y', shape=[-1, 10], dtype='float32')
        fc = fluid.layers.fc(x, size=30, bias_attr=False)
        loss = paddle.mean(fc)
        if is_training:
            adam = fluid.optimizer.Adam(learning_rate=1e-3)
            adam.minimize(loss)

        return loss, y

    def exec_strategy(self, use_experimental_executor):
        strategy = fluid.ExecutionStrategy()
        strategy.use_experimental_executor = use_experimental_executor
        return strategy

    def places(self, use_gpu, dev_cnt):
        if use_gpu:
            return fluid.cuda_places(list(range(dev_cnt)))
        else:
            return fluid.cpu_places(dev_cnt)

    def test_main(self):
        for use_gpu in [False, True]:
            for dev_cnt in [1, 2]:
                for is_training in [False, True]:
                    for use_experimental_executor in [False, True]:
                        for use_parallel_ssa_executor in [False, True]:
                            func = lambda: self.run_impl(
                                use_gpu,
                                dev_cnt,
                                is_training,
                                use_experimental_executor,
                                use_parallel_ssa_executor,
                            )
                            self.run_func_with_guard(func)

    def run_impl(
        self,
        use_gpu,
        dev_cnt,
        is_training,
        use_experimental_executor,
        use_parallel_ssa_executor,
    ):
        paddle.enable_static()
        enable_parallel_ssa_executor(use_parallel_ssa_executor)

        if fluid.is_compiled_with_cuda():
            if (
                fluid.core.globals()['FLAGS_enable_parallel_graph']
                and not use_gpu
            ):
                return
            # windows has only 1 GPU
            if use_gpu and dev_cnt > 1 and os.name == "nt":
                return
        else:
            if use_gpu:
                return

        loss, isolated_var = self.build_network(is_training)
        loss_name = loss.name if is_training else None

        places = self.places(use_gpu, dev_cnt)
        exe = fluid.Executor(places[0])

        exe.run(fluid.default_startup_program())

        prog = fluid.CompiledProgram(
            fluid.default_main_program()
        ).with_data_parallel(
            loss_name=loss_name,
            exec_strategy=self.exec_strategy(use_experimental_executor),
            places=places,
        )

        BATCH_SIZE = 8 * dev_cnt
        for _ in range(10):
            x_np = np.random.random(size=[BATCH_SIZE, 10]).astype('float32')
            y_np = np.random.random(size=[BATCH_SIZE, 10]).astype('float32')

            _, y_np_fetch = exe.run(
                prog,
                feed={'x': x_np, 'y': y_np},
                fetch_list=[loss, isolated_var],
            )

            np.testing.assert_array_equal(y_np, y_np_fetch)

        enable_parallel_ssa_executor(False)

    def run_func_with_guard(self, func):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            with fluid.unique_name.guard():
                with fluid.scope_guard(fluid.Scope()):
                    func()


if __name__ == '__main__':
    unittest.main()
