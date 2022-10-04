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

import unittest
import paddle
import paddle.fluid as fluid
import numpy
import os


class TestParallelExecutorDropExeScope(unittest.TestCase):

    def check_drop_scope(self, use_cuda=True):
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

        if not use_cuda:
            os.environ['CPU_NUM'] = str(2)

        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = paddle.mean(hidden)
            test_program = fluid.default_main_program().clone(for_test=True)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

        exe = fluid.Executor(place)
        exe.run(startup_program)

        exec_strateg = fluid.ExecutionStrategy()
        exec_strateg.num_iteration_per_drop_scope = 10

        train_exe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                           main_program=train_program,
                                           loss_name=loss.name,
                                           exec_strategy=exec_strateg)
        test_exe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                          main_program=test_program,
                                          share_vars_from=train_exe,
                                          exec_strategy=exec_strateg)

        x = numpy.random.random(size=(10, 1)).astype('float32')
        train_exe.run(feed={"X": x}, fetch_list=[loss.name])
        test_exe.run(feed={"X": x}, fetch_list=[loss.name])

        assert train_exe._need_create_local_exe_scopes() == False
        assert test_exe._need_create_local_exe_scopes() == False

        # drop the local execution scope immediately
        train_exe.drop_local_exe_scopes()
        test_exe.drop_local_exe_scopes()

        assert train_exe._need_create_local_exe_scopes()
        assert test_exe._need_create_local_exe_scopes()

    def test_drop_scope(self):
        self.check_drop_scope(use_cuda=False)
        if fluid.core.is_compiled_with_cuda():
            self.check_drop_scope(use_cuda=True)


if __name__ == '__main__':
    unittest.main()
