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

import paddle.fluid as fluid
from paddle.fluid.framework import Parameter
import numpy as np
from simple_nets import simple_fc_net
import random
import unittest
import os

batch_size = 32

feed_dict = {
    'image': np.random.random([batch_size, 784]).astype('float32'),
    'label': np.random.random_integers(
        low=0, high=9, size=[batch_size, 1]).astype('int64')
}


class InplaceTestBase(unittest.TestCase):
    def initParameter(self):
        self.use_cuda = True
        self.fuse_all_optimizer_ops = False

    def setUp(self):
        self.initParameter()
        if self.use_cuda and fluid.core.is_compiled_with_cuda():
            self.device_count = fluid.core.get_cuda_device_count()
        else:
            self.device_count = 4
        assert batch_size % self.device_count == 0

    def build_program_and_scope(self):
        self.place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()
        startup_program = fluid.Program()
        main_program = fluid.Program()
        startup_program.random_seed = 1
        main_program.random_seed = 1

        scope = fluid.Scope()
        with fluid.program_guard(main_program, startup_program):
            with fluid.unique_name.guard():
                loss = simple_fc_net()
                adam = fluid.optimizer.Adam(learning_rate=1e-3)
                adam.minimize(loss)

                with fluid.scope_guard(scope):
                    exe = fluid.Executor(
                        fluid.CUDAPlace(0)
                        if self.use_cuda else fluid.CPUPlace())
                    exe.run(startup_program)

        return main_program, scope, exe, loss

    def is_invalid_test(self):
        return self.use_cuda and not fluid.core.is_compiled_with_cuda()

    def get_all_vars(self, program):
        all_vars = program.global_block().vars
        all_vars_name = []
        for name, var in all_vars.items():
            if 0 not in var.shape and not var.persistable:
                all_vars_name.append(name)

        return all_vars_name

    def test_single_card_fetch_var(self):
        if self.is_invalid_test():
            return

        prog1, scope1, exe, loss1 = self.build_program_and_scope()
        prog2, scope2, _, loss2 = self.build_program_and_scope()
        prog3, scope3, _, loss3 = self.build_program_and_scope()

        build_strategy2 = fluid.BuildStrategy()
        build_strategy2.memory_optimize = False
        build_strategy2.enable_inplace = True
        build_strategy2.fuse_all_optimizer_ops = self.fuse_all_optimizer_ops

        compiled_prog2 = fluid.CompiledProgram(prog2).with_data_parallel(
            loss_name=loss2.name,
            build_strategy=build_strategy2,
            places=self.place)

        build_strategy3 = fluid.BuildStrategy()
        build_strategy3.memory_optimize = False
        build_strategy3.enable_inplace = False
        build_strategy3.fuse_all_optimizer_ops = self.fuse_all_optimizer_ops
        compiled_prog3 = fluid.CompiledProgram(prog3).with_data_parallel(
            loss_name=loss2.name,
            build_strategy=build_strategy3,
            places=self.place)

        all_vars_name = self.get_all_vars(prog1)
        repeated_var_names = all_vars_name * 4
        random.shuffle(repeated_var_names)  # add some random 

        for fetch_var in repeated_var_names:
            for _ in range(4):
                with fluid.scope_guard(scope1):
                    fetch_val1, = exe.run(prog1,
                                          feed=feed_dict,
                                          fetch_list=[fetch_var])

                with fluid.scope_guard(scope2):
                    fetch_val2, = exe.run(compiled_prog2,
                                          feed=feed_dict,
                                          fetch_list=[fetch_var])

                with fluid.scope_guard(scope3):
                    fetch_val3, = exe.run(compiled_prog3,
                                          feed=feed_dict,
                                          fetch_list=[fetch_var])

                self.assertTrue(np.array_equal(fetch_val1, fetch_val2))
                self.assertTrue(np.array_equal(fetch_val1, fetch_val3))

    def test_multi_card_fetch_var(self):
        if self.is_invalid_test():
            return

        prog1, scope1, exe, loss1 = self.build_program_and_scope()
        prog2, scope2, _, loss2 = self.build_program_and_scope()

        build_strategy1 = fluid.BuildStrategy()
        build_strategy1.memory_optimize = False
        build_strategy1.enable_inplace = True
        build_strategy1.fuse_all_optimizer_ops = self.fuse_all_optimizer_ops

        build_strategy2 = fluid.BuildStrategy()
        build_strategy2.memory_optimize = False
        build_strategy2.enable_inplace = False
        build_strategy2.fuse_all_optimizer_ops = self.fuse_all_optimizer_ops

        if self.use_cuda:
            places = fluid.cuda_places()
        else:
            places = fluid.cpu_places(self.device_count)

        compiled_prog1 = fluid.CompiledProgram(prog1).with_data_parallel(
            loss_name=loss1.name, build_strategy=build_strategy1, places=places)
        compiled_prog2 = fluid.CompiledProgram(prog2).with_data_parallel(
            loss_name=loss2.name, build_strategy=build_strategy2, places=places)

        repeated_var_names = self.get_all_vars(prog1) * 4
        random.shuffle(repeated_var_names)  # add some random 

        for fetch_var in repeated_var_names:
            for _ in range(4):
                with fluid.scope_guard(scope1):
                    fetch_val1, = exe.run(compiled_prog1,
                                          feed=feed_dict,
                                          fetch_list=[fetch_var])

                with fluid.scope_guard(scope2):
                    fetch_val2, = exe.run(compiled_prog2,
                                          feed=feed_dict,
                                          fetch_list=[fetch_var])

                self.assertTrue(np.array_equal(fetch_val1, fetch_val2))


class CPUInplaceTest(InplaceTestBase):
    def initParameter(self):
        self.use_cuda = False


class CUDAInplaceTestWithFuseOptimizationOps(InplaceTestBase):
    def initParameter(self):
        self.use_cuda = True
        # FIXME(zcd): if training is in CPU, fuse_all_optimizer_ops
        # may cause some diff with the origin model, the difference is
        # about 1e-8. This may be related to byte alignment.
        self.fuse_all_optimizer_ops = True


if __name__ == '__main__':
    unittest.main()
