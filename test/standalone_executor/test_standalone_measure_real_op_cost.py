# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np

import paddle
from paddle.base import core
from paddle.base.framework import Block
from paddle.distributed.auto_parallel.static.dist_attribute import (
    OperatorDistAttr,
)
from paddle.distributed.auto_parallel.static.dist_context import (
    DistributedContext,
)
from paddle.distributed.auto_parallel.static.utils import (
    measure_real_op_cost_wrt_program_and_place,
)
from paddle.static import Executor, Program, program_guard

paddle.enable_static()


class TestOpProfiling(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _build_startup_program_and_train_program(self):
        startup_program = Program()
        train_program = Program()
        with program_guard(train_program, startup_program):
            data = paddle.static.data(
                name='X', shape=[1024, 1], dtype='float32'
            )
            hidden = paddle.static.nn.fc(data, 10)
            loss = paddle.mean(hidden)
            paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
        return startup_program, train_program, loss

    def _add_feed_op_for_program_input_var(self, program, var_name, var_idx):
        # [in var] X --pack--> [var] feed --'X'-> [op] feed -'Out'-> [var] X
        global_block = program.global_block()
        global_block: Block
        if not global_block.has_var('feed'):
            global_block.create_var(
                name='feed',
                type=core.VarDesc.VarType.FEED_MINIBATCH,
                persistable=True,
            )
        feed_var = global_block.var('feed')
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [global_block.var(var_name)]},
            attrs={'col': var_idx},
        )

    def _init_dist_attr_for_each_op_in_program(self, program):
        dist_context = DistributedContext(serial_main_prog=program)
        global_block = program.global_block()
        global_block: Block
        for op in global_block.ops:
            op_dist_attr = OperatorDistAttr()
            dist_context.set_op_dist_attr_for_program(op, op_dist_attr)

    def _build_program(self):
        (
            startup_program,
            train_program,
            loss,
        ) = self._build_startup_program_and_train_program()
        self._add_feed_op_for_program_input_var(train_program, "X", 0)
        self._init_dist_attr_for_each_op_in_program(train_program)
        return train_program, startup_program, loss

    def _run_op_profiling(self, place, run_profiling=True):
        # enable static build and deterministic feature
        paddle.framework.set_flags(
            {
                'FLAGS_new_executor_static_build': 1,
                'FLAGS_embedding_deterministic': 1,
                'FLAGS_cudnn_deterministic': 1,
            }
        )
        paddle.seed(123)

        # build program
        train_program, startup_program, loss = self._build_program()

        print(startup_program)

        # Run the startup program once and only once.
        exe = Executor(place)
        exe.run(startup_program)

        if run_profiling:
            measure_real_op_cost_wrt_program_and_place(
                train_program, place, verbose=True
            )

        x = np.ones([1024, 1]).astype('float32')
        (loss_data,) = exe.run(
            train_program, feed={"X": x}, fetch_list=[loss.name]
        )
        return loss_data

    def test_op_profiling_cpu(self):
        sys.stdout.write("Running on CPU with profiling enabled.\n")
        loss0 = self._run_op_profiling(paddle.CPUPlace(), run_profiling=True)
        sys.stdout.write("Running on CPU with profiling disabled.\n")
        loss1 = self._run_op_profiling(paddle.CPUPlace(), run_profiling=False)
        loss0_s, loss1_s = "%.6f" % loss0, "%.6f" % loss1
        sys.stdout.write(f'loss comparison: "{loss0_s}" vs "{loss1_s}"\n')
        assert loss0_s == loss1_s, "loss value changed after profiling!"
        sys.stdout.write('PASSED.\n')


if __name__ == "__main__":
    unittest.main()
