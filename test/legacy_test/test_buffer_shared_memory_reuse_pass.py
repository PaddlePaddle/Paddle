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

import random
import unittest

import numpy as np
from simple_nets import simple_fc_net

import paddle
from paddle import base

batch_size = 32

feed_dict = {
    'image': np.random.random([batch_size, 784]).astype('float32'),
    'label': np.random.random_integers(
        low=0, high=9, size=[batch_size, 1]
    ).astype('int64'),
}


class InplaceTestBase(unittest.TestCase):
    def initParameter(self):
        self.use_cuda = True
        self.fuse_all_optimizer_ops = False

    def setUp(self):
        paddle.enable_static()
        self.initParameter()
        if self.use_cuda and base.core.is_compiled_with_cuda():
            self.device_count = base.core.get_cuda_device_count()
        else:
            self.device_count = 4
        assert batch_size % self.device_count == 0

    def build_program_and_scope(self):
        self.place = base.CUDAPlace(0) if self.use_cuda else base.CPUPlace()
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        startup_program = base.Program()
        main_program = base.Program()

        scope = base.Scope()
        with base.program_guard(main_program, startup_program):
            with base.unique_name.guard():
                loss = simple_fc_net()
                adam = paddle.optimizer.Adam(learning_rate=1e-3)
                adam.minimize(loss)

                with base.scope_guard(scope):
                    exe = base.Executor(
                        base.CUDAPlace(0) if self.use_cuda else base.CPUPlace()
                    )
                    exe.run(startup_program)

        return main_program, scope, exe, loss

    def is_invalid_test(self):
        return self.use_cuda and not base.core.is_compiled_with_cuda()

    def get_all_vars(self, program):
        all_vars = program.global_block().vars
        all_vars_name = []
        for name, var in all_vars.items():
            if 0 not in var.shape and not var.persistable:
                all_vars_name.append(name)

        return all_vars_name

    def check_single_card_fetch_var(self):
        with paddle.pir_utils.OldIrGuard():
            if self.is_invalid_test():
                return

            prog1, scope1, exe, loss1 = self.build_program_and_scope()
            scopes = []
            compiled_programs = []
            for memory_optimize in [False, True]:
                for enable_inplace in [False, True]:
                    prog, scope, _, loss = self.build_program_and_scope()
                    scopes.append(scope)
                    build_strategy = base.BuildStrategy()
                    build_strategy.memory_optimize = memory_optimize
                    build_strategy.enable_inplace = enable_inplace
                    build_strategy.fuse_all_optimizer_ops = (
                        self.fuse_all_optimizer_ops
                    )
                    compiled_prog = base.CompiledProgram(
                        prog, build_strategy=build_strategy
                    )
                    compiled_programs.append(compiled_prog)

            all_vars_name = self.get_all_vars(prog1)
            repeated_var_names = all_vars_name
            random.shuffle(repeated_var_names)  # add some random

            for fetch_var in repeated_var_names[:4]:
                for _ in range(2):
                    with base.scope_guard(scope1):
                        (fetch_val1,) = exe.run(
                            prog1, feed=feed_dict, fetch_list=[fetch_var]
                        )

                    for scope, compiled_prog in zip(scopes, compiled_programs):
                        with base.scope_guard(scope):
                            (fetch_val2,) = exe.run(
                                compiled_prog,
                                feed=feed_dict,
                                fetch_list=[fetch_var],
                            )
                            np.testing.assert_array_equal(
                                fetch_val1,
                                fetch_val2,
                                err_msg=f'error var name: {fetch_var}, fetch_val1: {fetch_val1[~np.equal(fetch_val1, fetch_val2)]}, fetch_val2: {fetch_val2[~np.equal(fetch_val1, fetch_val2)]}',
                            )


class CUDAInplaceTest(InplaceTestBase):
    def initParameter(self):
        self.use_cuda = True
        self.fuse_all_optimizer_ops = False

    def test_single_card_fetch_var(self):
        self.check_single_card_fetch_var()


class CPUInplaceTest(InplaceTestBase):
    def initParameter(self):
        self.use_cuda = False
        self.fuse_all_optimizer_ops = False

    def test_single_card_fetch_var(self):
        self.check_single_card_fetch_var()


if __name__ == '__main__':
    unittest.main()
